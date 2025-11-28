"""
LoRA Watermarking Key
=====================
极简水印系统：仅包含 Decoder，无 Mapper
"""

import torch
import torch.nn as nn
from src.watermarking_key.wm_key import WatermarkingKey
from src.arguments. wm_key_args import WatermarkingKeyArgs
from src.arguments.env_args import EnvArgs
from src.models.wm_decoder import WatermarkDecoder  # ← 使用原有的 Decoder


class LoRAWatermarkingKey(WatermarkingKey):
    """
    基于 LoRA 的水印 Key
    
    架构简化：
        原始 PTW: Generator + Mapper + Decoder
        新方案:   Generator (with LoRA) + Decoder
    
    训练流程：
        1. 冻结 Generator 基础权重
        2. 仅训练 LoRA 参数 + Decoder
        3. 通过 Toggle 机制生成参考图/水印图
    """
    
    def __init__(self, wm_key_args: WatermarkingKeyArgs, env_args: EnvArgs = None):
        super().__init__(wm_key_args, env_args)
        
        # ✅ 修复：使用原有 WatermarkDecoder 的正确参数
        # 原始签名：WatermarkDecoder(bitlen: int, decoder_arch: str)
        self.decoder = WatermarkDecoder(
            bitlen=wm_key_args.bitlen,
            decoder_arch=wm_key_args.decoder_arch  # 从配置文件读取：resnet18/resnet50/resnet101
        ). to(self.env_args.device)
        
        print(f"✅ Initialized {wm_key_args.decoder_arch} decoder for {wm_key_args. bitlen} bits")
    
    def extract(self, x: torch.Tensor, sigmoid=True, **kwargs):
        """
        从图像中提取水印
        
        参数：
            x: 图像张量 (B, C, H, W)，范围 [-1, 1]
            sigmoid: 是否应用 sigmoid（训练时 False，推理时 True）
        
        返回：
            msg_pred: 预测的水印 (B, bitlen)
        """
        msg_pred = self.decoder(x)
        if sigmoid:
            msg_pred = torch.sigmoid(msg_pred)
        return msg_pred
    
    def load(self, ckpt: str):
        """加载水印 Key"""
        checkpoint = torch.load(ckpt, map_location=self.env_args.device)
        
        # 加载 Decoder
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"✅ Loaded decoder state from checkpoint")
        
        # 加载配置
        if WatermarkingKeyArgs.WM_KEY_ARGS_KEY in checkpoint:
            self.wm_key_args = checkpoint[WatermarkingKeyArgs. WM_KEY_ARGS_KEY]
            print(f"✅ Loaded watermark config: message='{self.wm_key_args.message}'")
        
        print(f"✅ Loaded LoRA Watermarking Key from {ckpt}")
    
    def save(self, ckpt_fn: str = None) -> dict:
        """保存水印 Key"""
        save_dict = {
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY: self.wm_key_args,
            'decoder_state_dict': self. decoder.state_dict()
        }
        
        if ckpt_fn is not None:
            torch.save(save_dict, ckpt_fn)
            print(f"✅ Saved LoRA Watermarking Key to {ckpt_fn}")
        
        return save_dict