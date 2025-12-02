# import os
# from os.path import dirname, isdir

# import matplotlib
# import numpy as np
# import torch
# import torchvision
# import yaml
# from matplotlib import pyplot as plt
# from tqdm import tqdm


# def rnd_choice(elems, n):
#     a = np.arange(len(elems))
#     np.random.shuffle(a)
#     return torch.from_numpy(a[:n])

# def activate_gradients(wm_gan, args):
#     print(f"Sampling params up to depth: {args.param_depth}!")
#     wm_gan.activate_gradients()

#     ctr, all_params_ctr = 0, 0
#     masks, hooks = [], []

#     # a hook to modify the gradients with a mask.
#     def get_mask_grad(j):
#         def mask_grad(grad):
#             mask = masks[j]
#             return (torch.flatten(grad) * mask).view(*grad.shape)
#         return mask_grad

#     pbar = tqdm(wm_gan.parameters()[:args.param_depth])
#     for i, param in enumerate(pbar):
#         flattened_params = torch.flatten(param)
#         all_params_ctr += len(flattened_params)

#         idx = rnd_choice(flattened_params, n=int(len(flattened_params)))
#         ctr += len(idx)

#         mask = torch.zeros_like(flattened_params)
#         mask[idx] += 1
#         masks.append(mask)

#         hooks.append(param.register_hook(get_mask_grad(i)))
#         pbar.set_description(f"Watermarking {ctr}/{all_params_ctr} ({ctr/all_params_ctr*100:.2f}% parameters!")
#     return hooks

# def remove_hooks(hooks):
#     for hook in hooks:
#         hook.remove()

# def parse_yaml(yaml_path):
# 	""" Loads a yaml file """
# 	with open(yaml_path, 'r') as stream:
# 		data_loaded = yaml.safe_load(stream)
# 	return data_loaded

# def compute_bitwise_acc(message, msg_pred):
#     msg_pred_cpy = msg_pred.clone().detach()
#     msg_pred_cpy[msg_pred_cpy > 0.5] = 1.0
#     msg_pred_cpy[msg_pred_cpy <= 0.5] = 0.0
#     bitwise_acc = (message == msg_pred_cpy).float().mean(dim=1).mean().item()
#     return bitwise_acc * 100

# def norm(x, x_range):
#     """ Norms a data point """
#     x_min, x_max = x_range
#     if x_min == 0 and x_max == 1:
#         return (x + 1) / 2
#     elif x_min == -1 and x_max == 1:
#         return x * 2 - 1
#     else:
#         raise NotImplementedError

# def save_tensor_as_pdf(x, file_name):
#         grid = torchvision.utils.make_grid(x, nrow=1, range=(-1, 1), scale_each=True, normalize=True)
#         plt.imshow(grid.permute(1, 2, 0).numpy())
#         plt.axis('off')
#         plt.savefig(file_name, format="pdf", bbox_inches='tight')


# def normalize_image(tensor):
#     return (tensor - tensor.min()) / (tensor.max() - tensor.min())

# def plot_images(x, n_row=8, title=""):
#     """
#     绘制图像网格（兼容不同 torchvision 版本）
#     """
#     import matplotlib. pyplot as plt
#     import torchvision
#     from packaging import version
    
#     # ✅ 检测 torchvision 版本并使用正确的参数
#     tv_version = version.parse(torchvision.__version__)
    
#     if tv_version >= version.  parse("0.13.0"):
#         # 新版本使用 value_range
#         grid_img = torchvision.utils.make_grid(
#             x,
#             nrow=n_row,
#             value_range=(-1, 1),
#             normalize=True
#         )
#     else:
#         # 旧版本使用 range
#         grid_img = torchvision. utils.make_grid(
#             x,
#             nrow=n_row,
#             range=(-1, 1),
#             scale_each=True,
#             normalize=True
#         )
    
#     # 转换为 numpy 并显示
#     grid_img = grid_img.permute(1, 2, 0).cpu().numpy()
    
#     plt.figure(figsize=(15, 15))
#     plt.imshow(grid_img)
#     plt.title(title)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

import torch
import matplotlib
# Use non-interactive backend for servers without X
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from typing import Optional


def compute_bitwise_acc(msg_true: torch.Tensor, msg_pred: torch.Tensor) -> float:
    """
    计算比特准确率
    msg_true: (B, bitlen) 0/1 tensor
    msg_pred: (B, bitlen) probabilities or logits (if sigmoid applied beforehand)
    返回百分比 [0, 100]
    """
    if msg_pred.dtype != msg_true.dtype:
        # msg_pred expected to be probabilities in [0,1]
        msg_pred_binary = (msg_pred >= 0.5).float()
    else:
        msg_pred_binary = msg_pred
    correct = (msg_pred_binary == msg_true).float()
    accuracy = correct.mean().item() * 100.0
    return accuracy


def plot_images(x: torch.Tensor, n_row: int = 8, title: str = "", save_path: Optional[str] = None):
    """
    绘制图像网格并可选保存到文件（适用于无 GUI 的服务器环境）
    参数:
      x: Tensor, shape (B, C, H, W), range assumed [-1, 1]
      n_row: images per row
      title: optional title (placed in saved figure)
      save_path: if provided, save the image to this path; otherwise do nothing (no blocking show)
    兼容不同 torchvision 版本的 make_grid API（value_range vs range）。
    """
    # Ensure x is on CPU and detached
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
    else:
        raise ValueError("x must be a torch.Tensor")

    # Create grid using torchvision (handle API differences)
    try:
        grid_img = torchvision.utils.make_grid(x_cpu, nrow=n_row, value_range=(-1, 1), normalize=True)
    except TypeError:
        grid_img = torchvision.utils.make_grid(x_cpu, nrow=n_row, range=(-1, 1), scale_each=True, normalize=True)

    # Convert to HWC numpy
    grid_np = grid_img.permute(1, 2, 0).numpy()

    # Plot using matplotlib and save if requested
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path is not None:
        # Ensure parent dir exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=100)
        # print or logging for tracing
        print(f"📸 Saved visualization to: {save_path}")

    plt.close(fig)