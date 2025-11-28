"""
Embedding Arguments for LoRA Watermarking
==========================================
"""
from dataclasses import dataclass, field


@dataclass
class EmbedArgs:
    CONFIG_KEY = "embed_args"

    # ═══════════════════════════════════════════════════════════
    # Checkpoint 路径
    # ═══════════════════════════════════════════════════════════
    ckpt: str = field(default=None, metadata={
        "help": "Path to save/load the watermarked model checkpoint"
    })

    # ═══════════════════════════════════════════════════════════
    # LoRA 超参数（新增）
    # ═══════════════════════════════════════════════════════════
    lora_rank: int = field(default=8, metadata={
        "help": "LoRA rank (4=lightweight, 8=balanced, 16=high capacity)"
    })

    lora_alpha: float = field(default=1.0, metadata={
        "help": "LoRA scaling factor (controls update strength)"
    })

    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout probability (0. 0-0.1)"
    })

    target_layers: str = field(default='all', metadata={
        "help": "Which layers to inject LoRA: 'all', 'conv_only', 'late_layers'"
    })

    # ═══════════════════════════════════════════════════════════
    # 训练超参数
    # ═══════════════════════════════════════════════════════════
    ptw_lr: float = field(default=1e-3, metadata={
        "help": "Learning rate for LoRA and Decoder"
    })

    # ═══════════════════════════════════════════════════════════
    # 损失函数权重
    # ═══════════════════════════════════════════════════════════
    lambda_lpips: float = field(default=1.0, metadata={
        "help": "Weight for LPIPS perceptual loss"
    })

    lambda_id: float = field(default=0.0, metadata={
        "help": "Weight for face identity loss (set to 0 for non-face datasets)"
    })

    # ═══════════════════════════════════════════════════════════
    # DLWS 动态权重调度（新增）
    # ═══════════════════════════════════════════════════════════
    use_dlws: bool = field(default=True, metadata={
        "help": "Enable Dynamic Loss Weight Scheduling"
    })

    dlws_threshold: float = field(default=0.95, metadata={
        "help": "Bit accuracy threshold for switching DLWS phases"
    })

    dlws_wm_boost: float = field(default=2.0, metadata={
        "help": "Watermark loss multiplier in phase 1 (learning)"
    })

    dlws_lpips_boost: float = field(default=3.0, metadata={
        "help": "LPIPS loss multiplier in phase 2 (quality)"
    })