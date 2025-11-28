"""
Configuration Loader for LoRA Watermarking
===========================================
"""

import os
import yaml
from dataclasses import asdict
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs


class ConfigLoader:
    """从 YAML 文件加载配置"""
    
    @staticmethod
    def load_from_yaml(config_path: str):
        """
        加载 YAML 配置文件
        
        返回：
            model_args, wm_key_args, embed_args, env_args
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # ═══════════════════════════════════════════════════════════
        # Model Arguments
        # ═══════════════════════════════════════════════════════════
        model_config = config. get('model', {})
        model_args = ModelArgs(
            model_type=model_config.get('model_type', 'stylegan'),
            model_arch=model_config.get('model_arch', 'stylegan2'),
            model_ckpt=model_config.get('model_ckpt')
        )
        
        # ═══════════════════════════════════════════════════════════
        # Watermarking Key Arguments
        # ═══════════════════════════════════════════════════════════
        wm_config = config.get('watermark', {})
        message = wm_config.get('message', 'HELLO')
        bitlen = wm_config.get('bitlen', len(message) * 5)
        
        # ✅ 修复：使用原有 WatermarkDecoder 支持的参数
        wm_key_args = WatermarkingKeyArgs(
            message=message,
            bitlen=bitlen,
            decoder_arch=wm_config.get('decoder_arch', 'resnet18'),  # ← 使用 decoder_arch
            truncation_psi=wm_config.get('truncation_psi', 0.7),
            ir_se50_weights=wm_config.get('ir_se50_weights')
        )
        
        # ═══════════════════════════════════════════════════════════
        # Embedding Arguments
        # ═══════════════════════════════════════════════════════════
        lora_config = config.get('lora', {})
        train_config = config.get('training', {})
        output_config = config.get('output', {})
        
        # 构造 checkpoint 完整路径
        ckpt_dir = output_config.get('ckpt_dir', 'checkpoints')
        ckpt_name = output_config. get('ckpt_name', 'lora_watermark. pt')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        
        embed_args = EmbedArgs(
            ckpt=ckpt_path,
            lora_rank=lora_config.get('rank', 8),
            lora_alpha=lora_config.get('alpha', 1.0),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_layers=lora_config. get('target_layers', 'all'),
            ptw_lr=train_config.get('lr', 1e-3),
            lambda_lpips=train_config. get('lambda_lpips', 1.0),
            lambda_id=train_config.get('lambda_id', 0.0),
            use_dlws=train_config.get('use_dlws', True),
            dlws_threshold=train_config. get('dlws_threshold', 0.95),
            dlws_wm_boost=train_config.get('dlws_wm_boost', 2.0),
            dlws_lpips_boost=train_config.get('dlws_lpips_boost', 3.0)
        )
        
        # ═══════════════════════════════════════════════════════════
        # Environment Arguments
        # ═══════════════════════════════════════════════════════════
        env_config = config.get('environment', {})
        env_args = EnvArgs(
            device=env_config.get('device', 'cuda'),
            batch_size=train_config.get('batch_size', 4),
            eval_batch_size=train_config.get('eval_batch_size', 16),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            log_every=env_config.get('log_every', 100),
            save_every=env_config.get('save_every', 1000),
            logging_tool=env_config.get('logging_tool', 'wandb')
        )
        
        return model_args, wm_key_args, embed_args, env_args
    
    @staticmethod
    def print_config(model_args, wm_key_args, embed_args, env_args):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("  Configuration Summary")
        print("="*70)
        
        print("\n[Model]")
        print(f"  Type: {model_args.model_type}")
        print(f"  Arch: {model_args.model_arch}")
        print(f"  Checkpoint: {model_args.model_ckpt}")
        
        print("\n[Watermark]")
        print(f"  Message: '{wm_key_args.message}' ({wm_key_args.bitlen} bits)")
        print(f"  Decoder: {wm_key_args.decoder_arch}")
        print(f"  Truncation PSI: {wm_key_args.truncation_psi}")
        
        print("\n[LoRA]")
        print(f"  Rank: {embed_args.lora_rank}")
        print(f"  Alpha: {embed_args.lora_alpha}")
        print(f"  Target Layers: {embed_args. target_layers}")
        
        print("\n[Training]")
        print(f"  Learning Rate: {embed_args.ptw_lr}")
        print(f"  Batch Size: {env_args.batch_size}")
        print(f"  Lambda LPIPS: {embed_args.lambda_lpips}")
        print(f"  Lambda ID: {embed_args.lambda_id}")
        print(f"  DLWS Enabled: {embed_args.use_dlws}")
        
        print("\n[Output]")
        print(f"  Checkpoint: {embed_args.ckpt}")
        print(f"  Device: {env_args.device}")
        print(f"  Logging: {env_args.logging_tool}")
        
        print("\n" + "="*70 + "\n")