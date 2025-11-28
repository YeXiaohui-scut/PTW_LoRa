"""
LoRA Handler Module for Memory-Efficient Fine-tuning
====================================================
本模块实现了基于 torch.nn.utils.parametrize 的 LoRA 注入机制，用于解决以下问题：
1. 显存优化：仅训练低秩矩阵，大幅减少可训练参数量
2. Toggle 机制：单模型实现"有水印/无水印"双状态切换，避免 deepcopy
3. 零侵入：无需修改 StyleGAN 源码，利用参数化机制动态注入
"""

import torch
import torch. nn as nn
from contextlib import contextmanager
from typing import List


class LoRALayer(nn.Module):
    """
    LoRA 低秩分解层
    
    核心思想：将权重更新分解为 ΔW = B @ A，其中：
    - A: (rank, in_features) 高斯初始化
    - B: (out_features, rank) 零初始化
    - 训练初期 ΔW ≈ 0，保证模型从原始状态开始微调
    
    参数:
        original_weight: 原始卷积层权重 (out_channels, in_channels, k, k)
        rank: LoRA 秩，默认 4（通常 4-16 已足够）
        alpha: 缩放因子，用于控制 LoRA 更新的强度
    """
    
    def __init__(self, original_weight: torch.Tensor, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self. rank = rank
        self.alpha = alpha
        
        # 获取原始权重的形状信息
        # 对于 Conv2d: shape = (out_channels, in_channels, kernel_h, kernel_w)
        shape = original_weight.shape
        self.out_features = shape[0]
        self.in_features = shape[1]
        
        # LoRA 低秩矩阵初始化
        # lora_A: 使用 Kaiming 初始化，保证训练初期梯度稳定
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        # lora_B: 零初始化，确保 ΔW_init = 0
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # 计算缩放系数（参考 LoRA 论文）
        self.scaling = self.alpha / self.rank
        
        # Toggle 开关：控制 LoRA 是否生效
        self.enabled = True
    
    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        """
        参数化前向传播
        
        关键逻辑：
        - enabled=False: 返回原始权重 W_0（用于生成参考图像）
        - enabled=True: 返回 W_0 + ΔW（用于生成水印图像）
        
        参数:
            original_weight: 来自 Conv2d. weight 的原始权重
        
        返回:
            参数化后的权重（形状与输入一致）
        """
        if not self.enabled:
            # Reference Pass: 直接返回冻结权重
            return original_weight
        
        # Training Pass: 注入 LoRA 更新
        # 计算 ΔW = (B @ A) * scaling
        # 注意：需要 reshape 以匹配卷积核形状
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        
        # 扩展维度以匹配卷积核 (out, in, k, k)
        # 对于 1x1 卷积或全连接，需要根据实际形状调整
        if len(original_weight.shape) == 4:
            delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)
        
        return original_weight + delta_w


class LoRAHandler:
    """
    LoRA 生命周期管理器
    
    职责：
    1. 自动识别并注入 LoRA 到 StyleGAN 的 Synthesis 网络
    2. 提供 Toggle 接口（enable/disable）
    3. 管理可训练参数的获取
    """
    
    def __init__(self):
        self.lora_layers: List[LoRALayer] = []
        self.injected_modules: List[nn.Module] = []
    
    def inject(self, generator, rank: int = 4, alpha: float = 1.0, target_modules: List[str] = None):
        """
        向 StyleGAN Generator 注入 LoRA
        
        注入策略：
        - 仅处理 synthesis 模块（映射网络不需要微调）
        - 只注入 Conv2d/ConvTranspose2d 层
        - 使用 torch.nn.utils.parametrize 实现零侵入
        
        参数:
            generator: StyleGAN 的 GAN wrapper 对象
            rank: LoRA 秩（建议 4-16，越大表达能力越强但显存越高）
            alpha: 缩放因子
            target_modules: 目标模块名称列表，默认为 None 表示所有卷积层
        """
        print(f"> [LoRAHandler] Starting LoRA injection (rank={rank}, alpha={alpha})")
        
        # 遍历 synthesis 网络的所有子模块
        synthesis = generator.G.synthesis
        injection_count = 0
        
        for name, module in synthesis.named_modules():
            # 识别卷积层
            if isinstance(module, (nn.Conv2d, nn. ConvTranspose2d)):
                # 如果指定了 target_modules，检查名称是否匹配
                if target_modules is not None and not any(t in name for t in target_modules):
                    continue
                
                # 创建 LoRA 层
                lora_layer = LoRALayer(
                    original_weight=module.weight. data,
                    rank=rank,
                    alpha=alpha
                )
                
                # 使用 parametrize 注册（关键：不修改原模型结构）
                # 注册后，module. weight 的访问会自动经过 LoRALayer. forward
                nn.utils.parametrize.register_parametrization(
                    module, "weight", lora_layer
                )
                
                self.lora_layers.append(lora_layer)
                self.injected_modules.append(module)
                injection_count += 1
                
                print(f"  ✓ Injected LoRA into: {name} (weight shape: {module.weight.shape})")
        
        print(f"> [LoRAHandler] Injection complete!  Total layers: {injection_count}")
        print(f"> [LoRAHandler] Trainable params: {sum(p.numel() for p in self.parameters()):,}")
    
    def parameters(self):
        """
        仅返回 LoRA 的可训练参数
        
        关键优化：
        - 原始模型参数保持冻结
        - 优化器只更新 lora_A 和 lora_B
        - 显存占用 = rank * (in + out) << in * out
        """
        params = []
        for lora_layer in self.lora_layers:
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    @contextmanager
    def disable_lora(self):
        """
        上下文管理器：临时禁用所有 LoRA
        
        使用场景（Reference Pass）：
        with lora_handler.disable_lora():
            x_ref = generator.generate(...)  # 此时生成无水印图像
        # 退出后自动恢复，生成水印图像
        
        实现原理：
        - 进入：设置所有 LoRA. enabled = False
        - 退出：恢复 LoRA.enabled = True
        """
        # 保存当前状态
        original_states = [layer.enabled for layer in self.lora_layers]
        
        try:
            # 禁用所有 LoRA
            for layer in self.lora_layers:
                layer. enabled = False
            yield
        finally:
            # 恢复状态（通常恢复为 True）
            for layer, state in zip(self.lora_layers, original_states):
                layer.enabled = state
    
    def enable_lora(self):
        """手动启用所有 LoRA（训练时使用）"""
        for layer in self.lora_layers:
            layer.enabled = True
    
    def get_lora_state_dict(self):
        """
        导出 LoRA 参数字典（用于保存 checkpoint）
        
        注意：PyTorch 的 parametrize 机制会自动将 LoRA 参数
        包含在模型的 state_dict 中，因此通常不需要单独保存
        """
        state_dict = {}
        for i, layer in enumerate(self.lora_layers):
            state_dict[f'lora_{i}. lora_A'] = layer.lora_A
            state_dict[f'lora_{i}.lora_B'] = layer.lora_B
        return state_dict