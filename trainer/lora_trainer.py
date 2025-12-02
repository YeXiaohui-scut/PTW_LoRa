#!/usr/bin/env python3
"""
LoRA Watermarking Trainer (with real-time loss display)
======================================================

This file contains the trainer used to train LoRA + Decoder.
Key changes made here:
- Freeze decoder backbone by default (only train head + LoRA) to reduce trainable params.
- Show real-time losses in the tqdm progress bar using set_postfix (loss, wm, lpips, bit_acc).
- Keep previous DLWS, random-message training, and toggle (disable/enable) LoRA logic.
- Detailed comments indicate why each change was introduced.

Usage:
    python examples/train_lora_wm.py --config configs/lora_watermark_stylegan.yml
"""

import os
import time
import wandb
import torch
import torchvision
from accelerate import Accelerator
from torch.optim import Adam
from tqdm import tqdm

from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.criteria.lpips_loss import LPIPSLoss
from src.criteria.id_loss import IDLoss
from src.models.lora_stylegan import LoRAStyleGAN
from src.watermarking_key.lora_wm_key import LoRAWatermarkingKey
from src.trainer.trainer import Trainer
from src.utils.highlited_print import bcolors, print_warning
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc, plot_images


class LoRAWatermarkTrainer(Trainer):
    """
    LoRA Watermarking Trainer

    Real-time loss:
      - We display loss (total), loss_wm, loss_lpips, and bit_acc in the tqdm postfix every step.
      - This is lightweight and does not increase I/O significantly.

    Decoder backbone freezing:
      - We freeze decoder.base_model1 (backbone) and only train its head (dense).
      - This reduces trainable parameters from ~11M (resnet18) to a small head + LoRA.
    """

    EMBED_ARGS_KEY = "embed_args"
    WM_KEY_KEY = "wm_key"

    def __init__(self, embed_args: EmbedArgs, env_args: EnvArgs):
        self.embed_args = embed_args
        self.env_args = env_args
        self.generator: LoRAStyleGAN = None
        self.wm_key: LoRAWatermarkingKey = None

    def save(self, ckpt: str):
        """Save checkpoint (generator + wm_key + config)."""
        if ckpt is None:
            print_warning("> No checkpoint path provided.  Skipping save.")
            return

        torch.save({
            **self.generator.save(),
            **self.wm_key.save(),
            self.EMBED_ARGS_KEY: self.embed_args
        }, ckpt)

        print(f"✅ Saved checkpoint to '{bcolors.OKGREEN}{os.path.abspath(ckpt)}{bcolors.ENDC}'")

    def setup_logging(self):
        """Set up logging (wandb optional)."""
        run = None
        if self.env_args.logging_tool == "wandb":
            wandb.login()
            run = wandb.init(
                project="lora_watermark",
                config={
                    "lora_rank": getattr(self.embed_args, 'lora_rank', 8),
                    "lr": self.embed_args.ptw_lr,
                    "lambda_lpips": self.embed_args.lambda_lpips,
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
        """
        Train LoRA watermarking.

        Real-time loss display:
          - Uses tqdm.set_postfix to update loss values per iteration:
            loss (total), loss_wm, loss_lpips, bit_acc.

        Freezing decoder backbone:
          - Freeze wm_key.decoder.base_model1 if present; only train the decoder head (self.dense).
        """
        print(f"\n{'='*70}")
        print(f"  LoRA Watermarking Training (Random Message Mode)")
        print(f"{'='*70}\n")

        # Inject LoRA into synthesis network (this prints an injection report)
        generator.inject_lora(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.0,
            target_layers=target_layers
        )

        # Fixed message (only used for evaluation, training uses random messages)
        fixed_msg = LoRAWatermarkingKey.str_to_bits(wm_key.wm_key_args.message).unsqueeze(0)
        fixed_msg = fixed_msg[:, :wm_key.wm_key_args.bitlen].to(self.env_args.device)

        print(f"📝 Configuration:")
        print(f"   Bitlen: {wm_key.wm_key_args.bitlen}")
        print(f"   Fixed Message (for eval): '{wm_key.wm_key_args.message}'")
        print(f"   Training Mode: Random messages per batch")

        # Losses
        lpips_loss = LPIPSLoss()
        id_loss = IDLoss(ir_se50_weights=wm_key.wm_key_args.ir_se50_weights) if self.embed_args.lambda_id > 0 else None

        # ----------------------------
        # Freeze decoder backbone (recommended)
        # ----------------------------
        # The original decoder uses a pretrained ResNet backbone (base_model1) which is ~11M params.
        # Freeze it to only train the dense head (small) + LoRA. This reduces trainable params drastically.
        if hasattr(wm_key.decoder, "base_model1"):
            print("> Freezing decoder backbone parameters (only training classification head).")
            for p in wm_key.decoder.base_model1.parameters():
                p.requires_grad = False
            wm_key.decoder.base_model1.eval()

        # Collect trainable parameters: LoRA params + decoder params that require grad (head)
        trainable_decoder_params = [p for p in wm_key.decoder.parameters() if p.requires_grad]
        trainable_params = generator.get_lora_parameters() + trainable_decoder_params

        # Debug print
        total_lora = sum(p.numel() for p in generator.get_lora_parameters())
        total_decoder_head = sum(p.numel() for p in trainable_decoder_params)
        print(f"\n{'='*70}")
        print(f"  Training Configuration")
        print(f"{'='*70}")
        print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"LoRA Parameters: {total_lora:,}")
        print(f"Decoder Head Trainable Parameters: {total_decoder_head:,}")
        print(f"Learning Rate: {self.embed_args.ptw_lr}")
        print(f"Batch Size: {self.env_args.batch_size}")
        print(f"{'='*70}\n")

        # Optimizer: only LoRA params + decoder head
        opt = Adam(
            trainable_params,
            lr=self.embed_args.ptw_lr,
            betas=(0.9, 0.999)
        )

        # Accelerator prepare
        accelerator = Accelerator()
        generator, opt = accelerator.prepare(generator, opt)
        self.generator = generator
        self.wm_key = wm_key

        # Metrics and logging
        bit_acc = SmoothedValue()
        step = 0
        run = self.setup_logging()
        start_time = time.time()

        # Training loop
        with tqdm(total=self.env_args.log_every, desc="LoRA Watermark Training") as pbar:
            while True:
                opt.zero_grad()
                wm_key.eval()
                generator.G.train()

                # Random message per batch (training uses random bits for robustness)
                msg = torch.randint(
                    0, 2,
                    size=(self.env_args.batch_size, wm_key.wm_key_args.bitlen),
                    dtype=torch.float32,
                    device=self.env_args.device
                )

                # Reference Pass (LoRA disabled) -> produce x_ref
                with torch.no_grad():
                    with generator.lora_injector.lora_disabled():
                        w_frozen, x_ref = generator.generate(
                            batch_size=self.env_args.batch_size,
                            truncation_psi=wm_key.wm_key_args.truncation_psi
                        )
                
                # Clamp x_ref to [-1, 1] just in case
                x_ref = torch.clamp(x_ref, -1.0, 1.0)

                # Training Pass (LoRA enabled implicitly) -> produce x_wm
                _, x_wm = generator.generate(w=w_frozen)
                
                # Clamp x_wm to [-1, 1] to prevent explosion and ensure valid input for LPIPS/Decoder
                x_wm = torch.clamp(x_wm, -1.0, 1.0)

                # Compute current bit accuracy for DLWS decision (use decoder predictions)
                with torch.no_grad():
                    msg_pred = wm_key.extract(x_wm, sigmoid=True)
                    msg_pred_binary = (msg_pred >= 0.5).float()
                    current_bit_acc = (msg_pred_binary == msg).float().mean().item()

                # Dynamic Loss Weight Scheduler (DLWS)
                if self.embed_args.use_dlws:
                    if current_bit_acc < self.embed_args.dlws_threshold:
                        lambda_w = self.embed_args.dlws_wm_boost
                        lambda_lpips = self.embed_args.lambda_lpips * 0.5
                    else:
                        lambda_w = 1.0
                        lambda_lpips = self.embed_args.lambda_lpips * self.embed_args.dlws_lpips_boost
                else:
                    lambda_w = 1.0
                    lambda_lpips = self.embed_args.lambda_lpips
                
                # Safety clamp for lambda weights to prevent explosion
                lambda_w = min(lambda_w, 10.0)
                lambda_lpips = min(lambda_lpips, 10.0)

                # Compute losses (keep them as tensors for .item() reading)
                # Use BCEWithLogitsLoss for numerical stability (sigmoid is integrated)
                # wm_key.loss uses BCEWithLogitsLoss internally if implemented correctly,
                # but let's check wm_key.loss implementation.
                # Assuming wm_key.loss calls F.binary_cross_entropy_with_logits(msg_pred_logits, msg)
                
                # Debug: Check raw logits range
                msg_pred_logits = wm_key.extract(x_wm, sigmoid=False)
                if torch.isnan(msg_pred_logits).any() or torch.isinf(msg_pred_logits).any():
                     print(f"   - msg_pred_logits contains NaNs/Infs! Max: {msg_pred_logits.max().item()}, Min: {msg_pred_logits.min().item()}")

                loss_wm = lambda_w * torch.nn.functional.binary_cross_entropy_with_logits(msg_pred_logits, msg)
                loss_lpips_val = lambda_lpips * lpips_loss(x_wm, x_ref).mean()  # tensor
                loss = loss_wm + loss_lpips_val

                if id_loss is not None:
                    loss_id_val = self.embed_args.lambda_id * id_loss(x_wm, x_ref)[0].mean()
                    loss = loss + loss_id_val
                else:
                    loss_id_val = torch.tensor(0.0, device=loss.device)

                # Stability check: Skip step if loss is NaN/Inf
                if not torch.isfinite(loss):
                    print(f"\n⚠️  Warning: Loss is {loss.item()} (infinite/NaN) at step {step}. Skipping step to prevent collapse.")
                    print(f"   - loss_wm: {loss_wm.item()}")
                    print(f"   - loss_lpips: {loss_lpips_val.item()}")
                    print(f"   - loss_id: {loss_id_val.item() if isinstance(loss_id_val, torch.Tensor) else loss_id_val}")
                    
                    # Check for NaNs in generated images
                    if torch.isnan(x_wm).any():
                        print("   - x_wm contains NaNs!")
                    if torch.isnan(x_ref).any():
                        print("   - x_ref contains NaNs!")
                        
                    opt.zero_grad()
                    # Still update pbar to show progress, but don't increment step counter for saving/logging logic if strict
                    # Or just continue. Let's just continue.
                    continue

                # Backprop and step
                accelerator.backward(loss)
                if (step + 1) % self.env_args.gradient_accumulation_steps == 0:
                    # Stability: Clip gradients to prevent explosion
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, 1.0)
                    
                    opt.step()

                # Update smoothed metrics
                bit_acc.update(current_bit_acc)
                loss_dict = {
                    'loss_wm': float(loss_wm.item()),
                    'loss_lpips': float(loss_lpips_val.item()),
                    'loss_id': float(loss_id_val.item()) if isinstance(loss_id_val, torch.Tensor) else float(loss_id_val),
                    'total_loss': float(loss.item()),
                    'bit_acc': bit_acc.avg
                }
                loss_dict['capacity'] = max(0, 2 * (bit_acc.avg * wm_key.wm_key_args.bitlen - 0.5 * wm_key.wm_key_args.bitlen))

                # Real-time display: update tqdm postfix with losses and acc
                # Keep values small (rounded) to avoid long strings.
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'wm': f"{loss_dict['loss_wm']:.4f}",
                    'lpips': f"{loss_dict['loss_lpips']:.4f}",
                    'acc': f"{100*loss_dict['bit_acc']:.2f}%"
                })

                # Periodic logging (every log_every steps) - print a summary and save visualization
                if step % self.env_args.log_every == 0:
                    print()
                    print(f"{'='*70}")
                    print(f"Step: {step} | Time: {time.time() - start_time:.1f}s")
                    print(f"Bit Acc (smoothed): {bit_acc.avg*100:.2f}% | Capacity: {loss_dict['capacity']:.2f} bits")
                    print(f"DLWS: λ_wm={lambda_w:.2f}, λ_lpips={lambda_lpips:.2f}")
                    print(f"Loss (total/wm/lpips/id): {loss_dict['total_loss']:.4f} / {loss_dict['loss_wm']:.4f} / {loss_dict['loss_lpips']:.4f} / {loss_dict['loss_id']:.4f}")
                    print(f"{'='*70}\n")
                    pbar.reset()

                    # Visualization: save local image and upload to wandb (if enabled)
                    top = [x for x in x_wm[:3]]
                    middle = [x for x in x_ref[:3]]
                    bottom = [x - y for x, y in zip(x_wm[:3], x_ref[:3])]

                    # Save visualization locally
                    vis_dir = os.path.join(os.path.dirname(self.embed_args.ckpt), 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f'step_{step:06d}.png')

                    plot_images(
                        torch.stack(top + middle + bottom, 0),
                        n_row=3,
                        title=f"Step {step} | Acc: {bit_acc.avg*100:.1f}%",
                        save_path=vis_path
                    )

                    # Upload to wandb if active
                    if accelerator.is_local_main_process and run is not None:
                        try:
                            grid = torchvision.utils.make_grid(
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

                # Periodic evaluation & save
                if step % self.env_args.save_every == 0 and step > 0:
                    with torch.no_grad():
                        generator.lora_injector.enable_lora()

                        # Eval on random messages (avg of several trials)
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

                        # Eval on fixed message
                        _, x_eval_fixed = generator.generate(
                            batch_size=self.env_args.eval_batch_size,
                            truncation_psi=wm_key.wm_key_args.truncation_psi
                        )
                        msg_fixed_batch = fixed_msg.repeat(self.env_args.eval_batch_size, 1)
                        msg_pred_fixed = wm_key.extract(x_eval_fixed, sigmoid=True)
                        fixed_acc = compute_bitwise_acc(msg_fixed_batch, msg_pred_fixed)

                        print()
                        print(f"{'='*70}")
                        print(f"EVALUATION | Step {step}")
                        print(f"Random Messages Avg Acc: {avg_random_acc:.2f}% (5 trials)")
                        print(f"Fixed Message ('{wm_key.wm_key_args.message}') Acc: {fixed_acc:.2f}%")
                        print(f"{'='*70}\n")

                        if accelerator.is_local_main_process and run is not None:
                            wandb.log({
                                "eval/random_msg_acc": avg_random_acc,
                                "eval/fixed_msg_acc": fixed_acc,
                                "step": step
                            })

                        self.save(self.embed_args.ckpt)

                step += 1
                pbar.update(1)
                pbar.set_description(f"LoRA Training | Acc: {bit_acc.avg*100:.1f}%")