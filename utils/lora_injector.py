"""
LoRA Injector for StyleGAN Watermarking
========================================
核心功能：
1. 向 StyleGAN Synthesis 网络注入可训练的 LoRA 层
2. 提供 Toggle 机制（生成水印图 vs 无水印图）
3. 极小的参数开销（<1% 原模型参数量）
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List, Optional


class LoRALayer(nn.Module):
    """
    低秩自适应层（用于卷积权重参数化）
    
    数学原理：
        W_new = W_original + α * (B @ A)
        其中 A ∈ R^(rank × in_features)，B ∈ R^(out_features × rank)
    
    参数量对比：
        原始卷积: out × in × k × k  (例如 512×512×3×3 = 2. 4M)
        LoRA:     (out + in) × rank (例如 (512+512)×8 = 8K，仅 0.3%)
    """
    
    def __init__(
        self,
        original_weight: torch.Tensor,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 获取权重维度
        shape = original_weight.shape
        if len(shape) == 4:  # Conv2d: (out, in, k, k)
            self.out_dim = shape[0]
            self.in_dim = shape[1]
            self.is_conv = True
        elif len(shape) == 2:  # Linear: (out, in)
            self.out_dim = shape[0]
            self.in_dim = shape[1]
            self.is_conv = False
        else:
            raise ValueError(f"Unsupported weight shape: {shape}")
        
        # 初始化 LoRA 矩阵
        # A 使用 Kaiming 初始化，B 使用零初始化
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_dim) / rank)
        self.lora_B = nn.Parameter(torch. zeros(self.out_dim, rank))
        
        # 可选：Dropout（防止过拟合）
        self.dropout = nn. Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 缩放因子
        self.scaling = self.alpha / self.rank
        
        # Toggle 开关
        self.enabled = True
    
    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        参数化前向传播
        
        工作流程：
        1. enabled=False → 返回 W_original（生成无水印图像）
        2. enabled=True  → 返回 W_original + ΔW（生成水印图像）
        """
        if not self.enabled:
            return original_weight
        
        # 计算低秩更新 ΔW = (B @ A) * scaling
        delta_w = self.lora_B @ self.dropout(self.lora_A)
        delta_w = delta_w * self.scaling
        
        # 匹配卷积权重形状 (out, in) → (out, in, 1, 1)
        if self.is_conv:
            delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)
        
        return original_weight + delta_w


class LoRAInjector:
    """
    LoRA 生命周期管理器
    
    职责：
    1. 自动识别并注入 LoRA 到目标层
    2. 管理所有 LoRA 层的状态（enable/disable）
    3. 提供参数访问接口（用于优化器）
    """
    
    def __init__(self):
        self.lora_layers: List[LoRALayer] = []
        self.target_modules: List[nn.Module] = []
        self.layer_names: List[str] = []
    
    def inject(
        self,
        model: nn.Module,
        target_module_types: tuple = (nn.Conv2d,),
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        layer_filter: Optional[callable] = None
    ):
        """
        向模型注入 LoRA
        
        参数：
            model: 目标模型（通常是 generator. G. synthesis）
            target_module_types: 要注入的层类型（默认仅 Conv2d）
            rank: LoRA 秩（建议 4-16）
            alpha: 缩放因子（建议 0.5-2.0）
            dropout: Dropout 概率（建议 0.0-0.1）
            layer_filter: 自定义过滤函数，例如 lambda name: 'torgb' not in name
        
        示例：
            # 仅注入到 StyleGAN2 的 conv0 和 conv1
            injector.inject(
                generator.G.synthesis,
                layer_filter=lambda name: 'conv0' in name or 'conv1' in name
            )
        """
        print(f"\n{'='*70}")
        print(f"  LoRA Injection Report")
        print(f"{'='*70}")
        print(f"Rank: {rank} | Alpha: {alpha} | Dropout: {dropout}")
        print(f"Target Types: {[t.__name__ for t in target_module_types]}")
        print(f"{'-'*70}")
        
        injection_count = 0
        total_lora_params = 0
        
        for name, module in model.named_modules():
            # 类型过滤
            if not isinstance(module, target_module_types):
                continue
            
            # 自定义过滤
            if layer_filter is not None and not layer_filter(name):
                continue
            
            # 创建 LoRA 层
            lora_layer = LoRALayer(
                original_weight=module.weight. data,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            # 使用 parametrize 注册（零侵入式修改）
            nn.utils.parametrize.register_parametrization(
                module, "weight", lora_layer
            )
            
            # 记录
            self.lora_layers.append(lora_layer)
            self.target_modules.append(module)
            self.layer_names.append(name)
            
            # 统计参数量
            lora_params = sum(p. numel() for p in lora_layer.parameters())
            total_lora_params += lora_params
            
            injection_count += 1
            print(f"✓ [{injection_count:2d}] {name:40s} | Params: +{lora_params:,}")
        
        print(f"{'-'*70}")
        print(f"Total Injected Layers: {injection_count}")
        print(f"Total LoRA Parameters: {total_lora_params:,}")
        print(f"{'='*70}\n")
        
        if injection_count == 0:
            print("⚠️  WARNING: No layers were injected!  Check your filter settings.")
    
    def parameters(self) -> List[nn.Parameter]:
        """获取所有 LoRA 可训练参数"""
        params = []
        for lora_layer in self.lora_layers:
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def enable_lora(self):
        """启用所有 LoRA（生成水印图像）"""
        for layer in self.lora_layers:
            layer.enabled = True
    
    def disable_lora(self):
        """禁用所有 LoRA（生成无水印图像）"""
        for layer in self. lora_layers:
            layer.enabled = False
    
    @contextmanager
    def lora_disabled(self):
        """
        上下文管理器：临时禁用 LoRA
        
        用法：
            with injector.lora_disabled():
                x_ref = generator.generate(...)  # 无水印
            x_wm = generator.generate(...)       # 带水印
        """
        self.disable_lora()
        try:
            yield
        finally:
            self.enable_lora()
    
    def get_state_dict(self) -> dict:
        """导出 LoRA 参数（用于保存 checkpoint）"""
        state_dict = {}
        for i, lora_layer in enumerate(self. lora_layers):
            state_dict[f'lora_{i}. A'] = lora_layer.lora_A
            state_dict[f'lora_{i}.B'] = lora_layer.lora_B
            state_dict[f'lora_{i}.alpha'] = lora_layer.alpha
            state_dict[f'lora_{i}.rank'] = lora_layer.rank
        return state_dict
    
    def load_state_dict(self, state_dict: dict):
        """加载 LoRA 参数"""
        for i, lora_layer in enumerate(self.lora_layers):
            lora_layer.lora_A. data = state_dict[f'lora_{i}.A']
            lora_layer.lora_B.data = state_dict[f'lora_{i}. B']
    
    def merge_lora_to_base(self):
        """
        将 LoRA 合并到基础权重（用于推理优化）
        
        注意：此操作不可逆，会永久修改模型权重
        """
        for lora_layer, module in zip(self.lora_layers, self.target_modules):
            # 移除参数化
            nn.utils.parametrize.remove_parametrizations(module, "weight")
            
            # 手动合并权重
            with torch.no_grad():
                delta_w = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                if lora_layer.is_conv:
                    delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)
                module.weight.data += delta_w
        
        print("✅ LoRA merged into base weights.  Model is now standalone.")