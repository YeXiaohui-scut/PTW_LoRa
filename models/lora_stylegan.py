"""
LoRA-enabled StyleGAN Wrapper (updated)
======================================

说明：
- 修复 inject_lora 调用，使用与 src/utils/lora_injector.py 兼容的参数签名。
- 不再传入已删除的参数 `target_module_types`，而是通过 param_name_pattern 和 layer_name_filter 控制注入层。
- 保留 generate_with_watermark / generate_without_watermark / get_lora_parameters 等接口。

修改动机：
之前的实现将 `target_module_types=(torch.nn.Conv2d,)` 传给 injector.inject，
但我们改用遍历 named_parameters 的注入器（更稳健），其 inject 接口接受
param_name_pattern 和 layer_name_filter，因此需要在这里做映射与适配。

使用说明（inject_lora 的 target_layers 参数）：
- 'all'       : 注入所有 name 中包含 "weight" 的参数（默认）
- 'conv_only' : 仅注入 conv0/conv1 的 weight（跳过 affine/torgb）
- 'late_layers': 仅注入高分辨率层（b256, b512, b1024）的 conv0/conv1 weight
- 也可直接调用 inject_lora(rank=..., target_layers=<自定义>) 并在代码中扩展过滤逻辑

备注：
- 注入器会打印注入报告（Total Injected Layers / Total LoRA Parameters）。
- 若在同一 Python 进程中多次注入，建议重启进程以避免 parametrization 冲突。
"""
import torch
import torch.nn as nn

from src.models.stylegan import StyleGAN
from src.utils.lora_injector import LoRAInjector
from src.arguments.model_args import ModelArgs
from src.arguments.env_args import EnvArgs


class LoRAStyleGAN(StyleGAN):
    """
    支持 LoRA 的 StyleGAN（wrapper）

    主要职责：
    - 提供一键注入 LoRA 的方法 inject_lora()
    - 管理 LoRAInjector 实例
    - 提供获取 LoRA 参数的接口（get_lora_parameters）
    """

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        super().__init__(model_args, env_args)
        self.lora_injector: LoRAInjector = None

    def inject_lora(
        self,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_layers: str = 'all'
    ):
        """
        向 Synthesis 网络注入 LoRA（兼容当前的 robust injector）

        参数:
            rank: LoRA 秩 (int)
            alpha: 缩放因子 (float)
            dropout: 未使用（保留参数以兼容接口）
            target_layers: 注入策略:
                - 'all' : 所有包含 'weight' 的参数
                - 'conv_only' : 仅注入 conv0/conv1 的 weight（跳过 affine/torgb）
                - 'late_layers' : 仅后半部分高分辨率层 (b256, b512, b1024) 的 conv0/conv1
        """
        self.lora_injector = LoRAInjector()

        # 根据 target_layers 选择匹配 pattern 和可选的过滤器
        if target_layers == 'all':
            # 匹配所有包含 weight 的参数（conv、torgb、affine 等）
            param_name_pattern = r".*weight.*"
            layer_name_filter = None

        elif target_layers == 'conv_only':
            # 仅匹配 conv0/conv1 的 weight（排除 affine / torgb）
            param_name_pattern = r".*conv[01]\.weight$"
            # 不需要额外的 layer_name_filter；pattern 已足够精确
            layer_name_filter = None

        elif target_layers == 'late_layers':
            # 仅匹配高分辨率层 b256/b512/b1024 的 conv0/conv1 的 weight
            param_name_pattern = r".*conv[01]\.weight$"
            layer_name_filter = lambda full_name: any(f"b{res}." in full_name for res in [256, 512, 1024])

        else:
            # 允许用户传入自定义的 pattern 名称（如果 target_layers 是正则串）
            # 例如: target_layers=".*torgb.*" 将直接当作 pattern 使用
            param_name_pattern = str(target_layers)
            layer_name_filter = None

        # 执行注入（注入器会打印详细的注入报告）
        self.lora_injector.inject(
            model=self.G.synthesis,
            rank=rank,
            alpha=alpha,
            param_name_pattern=param_name_pattern,
            layer_name_filter=layer_name_filter
        )

        print(f"✅ LoRA injected into '{target_layers}' layers of synthesis network.")

    def generate_with_watermark(self, batch_size=1, truncation_psi=1.0, w=None):
        """生成带水印图像（开启 LoRA）"""
        if self.lora_injector is not None:
            self.lora_injector.enable_lora()
        return self.generate(batch_size=batch_size, truncation_psi=truncation_psi, w=w)

    def generate_without_watermark(self, batch_size=1, truncation_psi=1.0, w=None):
        """生成无水印图像（关闭 LoRA）"""
        if self.lora_injector is not None:
            self.lora_injector.disable_lora()
        return self.generate(batch_size=batch_size, truncation_psi=truncation_psi, w=w)

    def get_lora_parameters(self):
        """返回 LoRA 的可训练参数（用于传递给优化器）"""
        if self.lora_injector is None:
            raise RuntimeError("LoRA not injected yet. Call inject_lora() first.")
        return self.lora_injector.parameters()

    def save(self, **kwargs) -> dict:
        """保存模型（包含 LoRA 参数信息）"""
        base_dict = super().save(**kwargs)

        if self.lora_injector is not None:
            base_dict['lora_state_dict'] = self.lora_injector.get_state_dict()
            base_dict['lora_injected'] = True
        else:
            base_dict['lora_injected'] = False

        return base_dict