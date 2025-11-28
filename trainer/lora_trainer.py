"""
LoRA Watermarking Trainer - 完整修复版
=======================================
"""

import os
import time
import wandb
import torch
import torchvision
from accelerate import Accelerator
from torch.optim import Adam
from tqdm import tqdm

from src.arguments. embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.criteria.lpips_loss import LPIPSLoss
from src.criteria.id_loss import IDLoss
from src.models.lora_stylegan import LoRAStyleGAN
from src.watermarking_key. lora_wm_key import LoRAWatermarkingKey
from src.trainer.trainer import Trainer
from src.utils.highlited_print import bcolors, print_warning
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc


class LoRAWatermarkTrainer(Trainer):
    
    EMBED_ARGS_KEY = "embed_args"
    WM_KEY_KEY = "wm_key"
    
    def __init__(self, embed_args: EmbedArgs, env_args: EnvArgs):
        self.embed_args = embed_args
        self.env_args = env_args
        self.generator: LoRAStyleGAN = None
        self.wm_key: LoRAWatermarkingKey = None
    
    def save(self, ckpt: str):
        if ckpt is None:
            print_warning("> No checkpoint path provided. Skipping save.")
            return
        
        torch.save({
            **self.generator.save(),
            **self.wm_key.save(),
            self. EMBED_ARGS_KEY: self.embed_args
        }, ckpt)
        
        print(f"✅ Saved checkpoint to '{bcolors.OKGREEN}{os.path.abspath(ckpt)}{bcolors. ENDC}'")
    
    def setup_logging(self):
        run = None
        if self.env_args.logging_tool == "wandb":
            wandb.login()
            run = wandb.init(
                project="lora_watermark",
                config={
                    "lora_rank": getattr(self.embed_args, 'lora_rank', 8),
                    "lr": self.embed_args.ptw_lr,
                    "lambda_lpips": self.embed_args. lambda_lpips,
                    "batch_size": self.env_args.batch_size,
                    "random_message_training": True
                }
            )
        return run
    
    def train(
        self,
        generator: LoRAStyleGAN,
        wm_key: LoRAWatermarkingKey,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        target_layers: str = 'all'
    ):
        print(f"\n{'='*70}")
        print(f"  LoRA Watermarking Training (Random Message Mode)")
        print(f"{'='*70}\n")
        
        generator.inject_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.0,
            target_layers=target_layers
        )
        
        # 固定消息（仅用于评估）
        fixed_msg = LoRAWatermarkingKey. str_to_bits(wm_key.wm_key_args.message). unsqueeze(0)
        fixed_msg = fixed_msg[:, :wm_key.wm_key_args.bitlen]
        fixed_msg = fixed_msg.to(self.env_args.device)
        
        print(f"📝 Configuration:")
        print(f"   Bitlen: {wm_key. wm_key_args.bitlen}")
        print(f"   Fixed Message (for eval): '{wm_key.wm_key_args.message}'")
        print(f"   Training Mode: Random messages per batch")
        
        # 损失函数
        lpips_loss = LPIPSLoss()
        id_loss = IDLoss(ir_se50_weights=wm_key.wm_key_args.ir_se50_weights) \
            if self.embed_args.lambda_id > 0 else None
        
        # 优化器
        trainable_params = generator.get_lora_parameters() + list(wm_key.decoder.parameters())
        opt = Adam(
            trainable_params,
            lr=self.embed_args. ptw_lr,
            betas=(0.9, 0.999)
        )
        
        print(f"\n{'='*70}")
        print(f"  Training Configuration")
        print(f"{'='*70}")
        print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"LoRA Parameters: {sum(p.numel() for p in generator.get_lora_parameters()):,}")
        print(f"Decoder Parameters: {sum(p.numel() for p in wm_key.decoder.parameters()):,}")
        print(f"Learning Rate: {self.embed_args.ptw_lr}")
        print(f"Batch Size: {self.env_args.batch_size}")
        print(f"{'='*70}\n")
        
        # Accelerator
        accelerator = Accelerator()
        generator, opt = accelerator.prepare(generator, opt)
        
        self.generator = generator
        self. wm_key = wm_key
        
        # 训练循环
        bit_acc = SmoothedValue()
        step = 0
        run = self.setup_logging()
        start_time = time.time()
        
        with tqdm(total=self.env_args.log_every, desc="LoRA Watermark Training") as pbar:
            while True:
                opt.zero_grad()
                wm_key.eval()
                generator.G.train()
                
                # ✅ 每个 batch 随机生成消息
                msg = torch.randint(
                    0, 2,
                    size=(self.env_args.batch_size, wm_key.wm_key_args.bitlen),
                    dtype=torch.float32,
                    device=self.env_args.device
                )
                
                # Reference Pass（无水印）
                with torch.no_grad():
                    with generator.lora_injector. lora_disabled():
                        w_frozen, x_ref = generator.generate(
                            batch_size=self.env_args.batch_size,
                            truncation_psi=wm_key.wm_key_args.truncation_psi
                        )
                
                # Training Pass（带水印）
                _, x_wm = generator.generate(w=w_frozen)
                
                # ✅ 修复：正确计算 bit accuracy
                with torch.no_grad():
                    msg_pred = wm_key.extract(x_wm, sigmoid=True)
                    msg_pred_binary = (msg_pred >= 0.5). float()
                    current_bit_acc = (msg_pred_binary == msg).float().mean(). item()
                
                # DLWS
                if self.embed_args.use_dlws:
                    if current_bit_acc < self.embed_args.dlws_threshold:
                        lambda_w = self.embed_args.dlws_wm_boost
                        lambda_lpips = self.embed_args. lambda_lpips * 0.5
                    else:
                        lambda_w = 1.0
                        lambda_lpips = self.embed_args.lambda_lpips * self.embed_args.dlws_lpips_boost
                else:
                    lambda_w = 1.0
                    lambda_lpips = self.embed_args.lambda_lpips
                
                loss_dict = {}
                
                # 水印损失
                loss_wm = lambda_w * wm_key.loss(x_wm, msg)
                loss = loss_wm
                loss_dict['loss_wm'] = float(loss_wm)
                loss_dict['lambda_w'] = lambda_w
                
                # LPIPS 损失
                loss_lpips_val = lambda_lpips * lpips_loss(x_wm, x_ref). mean()
                loss += loss_lpips_val
                loss_dict['loss_lpips'] = float(loss_lpips_val)
                loss_dict['lambda_lpips'] = lambda_lpips
                
                # ID 损失
                if id_loss is not None:
                    loss_id_val = self.embed_args.lambda_id * id_loss(x_wm, x_ref)[0]. mean()
                    loss += loss_id_val
                    loss_dict['loss_id'] = float(loss_id_val)
                
                # 反向传播
                accelerator.backward(loss)
                if (step + 1) % self. env_args.gradient_accumulation_steps == 0:
                    opt.step()
                
                # 更新指标
                bit_acc.update(current_bit_acc)
                loss_dict['bit_acc'] = bit_acc.avg
                loss_dict['capacity'] = max(0, 2 * (bit_acc.avg * wm_key.wm_key_args.bitlen - 0.5 * wm_key.wm_key_args.bitlen))
                
                # 日志记录
                if step % self.env_args.log_every == 0:
                    print()
                    print(f"{'='*70}")
                    print(f"Step: {step} | Time: {time.time() - start_time:.1f}s")
                    print(f"Bit Acc: {bit_acc. avg*100:.2f}% | Capacity: {loss_dict['capacity']:.2f} bits")
                    print(f"DLWS: λ_wm={lambda_w:.2f}, λ_lpips={lambda_lpips:.2f}")
                    print(f"{'='*70}\n")
                    pbar.reset()
                    
                    # WandB 可视化
                    if accelerator.is_local_main_process and run is not None:
                        top = [x for x in x_wm[:3]]
                        middle = [x for x in x_ref[:3]]
                        bottom = [x - y for x, y in zip(x_wm[:3], x_ref[:3])]
                        
                        try:
                            grid = torchvision.utils. make_grid(
                                torch.stack(top + middle + bottom, 0),
                                nrow=3,
                                normalize=True,
                                value_range=(-1, 1)
                            )
                        except TypeError:
                            grid = torchvision.utils.make_grid(
                                torch.stack(top + middle + bottom, 0),
                                nrow=3,
                                normalize=True,
                                range=(-1, 1)
                            )
                        
                        images = wandb.Image(
                            grid,
                            caption="Top: Watermarked | Middle: Reference | Bottom: Diff"
                        )
                        wandb.log({"examples": images})
                
                if accelerator.is_local_main_process and run is not None:
                    wandb.log({**loss_dict, "step": step})
                
                # 评估 & 保存
                if step % self.env_args.save_every == 0 and step > 0:
                    with torch.no_grad():
                        generator.lora_injector.enable_lora()
                        
                        # 评估 1: 随机消息
                        random_acc_list = []
                        for _ in range(5):
                            _, x_eval = generator.generate(
                                batch_size=self.env_args.eval_batch_size,
                                truncation_psi=wm_key.wm_key_args.truncation_psi
                            )
                            msg_random = torch.randint(
                                0, 2,
                                size=(self.env_args.eval_batch_size, wm_key.wm_key_args.bitlen),
                                dtype=torch.float32,
                                device=self.env_args.device
                            )
                            msg_pred = wm_key.extract(x_eval, sigmoid=True)
                            random_acc_list.append(compute_bitwise_acc(msg_random, msg_pred))
                        
                        avg_random_acc = sum(random_acc_list) / len(random_acc_list)
                        
                        # 评估 2: 固定消息
                        _, x_eval_fixed = generator.generate(
                            batch_size=self.env_args.eval_batch_size,
                            truncation_psi=wm_key.wm_key_args.truncation_psi
                        )
                        msg_fixed_batch = fixed_msg. repeat(self.env_args.eval_batch_size, 1)
                        msg_pred_fixed = wm_key. extract(x_eval_fixed, sigmoid=True)
                        fixed_acc = compute_bitwise_acc(msg_fixed_batch, msg_pred_fixed)
                        
                        print()
                        print(f"{'='*70}")
                        print(f"EVALUATION | Step {step}")
                        print(f"Random Messages Avg Acc: {avg_random_acc:.2f}% (5 trials)")  # ← 修复
                        print(f"Fixed Message ('{wm_key.wm_key_args. message}') Acc: {fixed_acc:.2f}%")  # ← 修复
                        print(f"{'='*70}\n")
                        
                        if accelerator.is_local_main_process and run is not None:
                            wandb.log({
                                "eval/random_msg_acc": avg_random_acc,
                                "eval/fixed_msg_acc": fixed_acc,
                                "step": step
                            })
                        
                        self.save(self.embed_args. ckpt)
                
                step += 1
                pbar.update(1)
                pbar.set_description(f"LoRA Training | Acc: {bit_acc.avg*100:.1f}%")