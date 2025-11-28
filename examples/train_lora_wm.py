#!/usr/bin/env python3
"""
LoRA Watermarking Training Script
==================================

用法：
    python train_lora_watermark.py --config configs/lora_watermark_stylegan2.yaml
    
    或使用命令行参数覆盖：
    python train_lora_watermark.py \
        --config configs/lora_watermark_stylegan2. yaml \
        --lora_rank 16 \
        --batch_size 8 \
        --message "MYLOGO"
"""

import argparse
import sys
import os

# 添加项目根目录到 Python 路径（把项目根加入，而不是 examples 目录）
# 这样 `import src.*` 能在运行该示例脚本时被正确解析。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lora_stylegan import LoRAStyleGAN
from src.watermarking_key. lora_wm_key import LoRAWatermarkingKey
from src.trainer.lora_trainer import LoRAWatermarkTrainer
from src.utils.config_loader import ConfigLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA Watermarking for StyleGAN")
    
    # ═══════════════════════════════════════════════════════════
    # 配置文件
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    
    # ═══════════════════════════════════════════════════════════
    # 可选覆盖参数
    # ═══════════════════════════════════════════════════════════
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--message', type=str, help='Override watermark message')
    parser. add_argument('--lora_rank', type=int, help='Override LoRA rank')
    parser.add_argument('--lora_alpha', type=float, help='Override LoRA alpha')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Override device')
    parser.add_argument('--output_ckpt', type=str, help='Override output checkpoint path')
    
    # ═══════════════════════════════════════════════════════════
    # 其他选项
    # ═══════════════════════════════════════════════════════════
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (disable wandb)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ═══════════════════════════════════════════════════════════
    # 1. 加载配置
    # ═══════════════════════════════════════════════════════════
    print(f"\n🔧 Loading configuration from: {args.config}")
    model_args, wm_key_args, embed_args, env_args = ConfigLoader.load_from_yaml(args.config)
    
    # 应用命令行覆盖
    if args. model_ckpt:
        model_args.model_ckpt = args.model_ckpt
    if args.message:
        wm_key_args.message = args.message
        wm_key_args.bitlen = len(args.message) * 5
    if args.lora_rank:
        embed_args.lora_rank = args.lora_rank
    if args.lora_alpha:
        embed_args.lora_alpha = args.lora_alpha
    if args.batch_size:
        env_args.batch_size = args.batch_size
    if args.lr:
        embed_args.ptw_lr = args.lr
    if args. device:
        env_args. device = args.device
    if args.output_ckpt:
        embed_args.ckpt = args.output_ckpt
    if args.debug:
        env_args.logging_tool = "none"
    
    # 打印配置
    ConfigLoader.print_config(model_args, wm_key_args, embed_args, env_args)
    
    # ═══════════════════════════════════════════════════════════
    # 2. 加载模型
    # ═══════════════════════════════════════════════════════════
    print("📦 Loading StyleGAN model...")
    generator = LoRAStyleGAN(model_args, env_args)
    
    if args.resume and os.path.exists(embed_args.ckpt):
        print(f"♻️  Resuming from checkpoint: {embed_args.ckpt}")
        checkpoint = torch.load(embed_args.ckpt, map_location=env_args.device)
        # 加载 Generator（包含 LoRA）
        generator.G.load_state_dict(checkpoint['G_ema'])
    else:
        print(f"🆕 Loading pretrained model: {model_args.model_ckpt}")
        generator.load_network(model_args.model_ckpt)
    
    # ═══════════════════════════════════════════════════════════
    # 3. 初始化水印 Key
    # ═══════════════════════════════════════════════════════════
    print("🔑 Initializing watermarking key...")
    wm_key = LoRAWatermarkingKey(wm_key_args, env_args)
    
    if args.resume and os.path.exists(embed_args.ckpt):
        print(f"♻️  Loading decoder from checkpoint...")
        checkpoint = torch.load(embed_args.ckpt, map_location=env_args. device)
        if 'decoder_state_dict' in checkpoint:
            wm_key.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # ═══════════════════════════════════════════════════════════
    # 4. 开始训练
    # ═══════════════════════════════════════════════════════════
    print("🚀 Starting LoRA watermarking training.. .\n")
    trainer = LoRAWatermarkTrainer(embed_args, env_args)
    
    try:
        trainer.train(
            generator=generator,
            wm_key=wm_key,
            lora_rank=embed_args.lora_rank,
            lora_alpha=embed_args.lora_alpha,
            target_layers=embed_args.target_layers
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print(f"💾 Saving checkpoint to: {embed_args. ckpt}")
        trainer.save(embed_args.ckpt)
        print("✅ Checkpoint saved successfully.")


if __name__ == "__main__":
    import torch
    main()