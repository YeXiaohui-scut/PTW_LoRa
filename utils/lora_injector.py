"""
Robust LoRA Injector for StyleGAN (compatible with StyleGAN-XL)

修复说明（Key fix）:
- 在遍历 model.named_parameters() 前先把它转为 list(...) 的快照，
  避免在循环中调用 register_parametrization 导致底层 OrderedDict 被修改从而触发
  "OrderedDict mutated during iteration" 错误。

其他说明:
- 该注入器不依赖模块的具体类型，直接在拥有参数的 parent module 上注册 parametrization，
  因此兼容 StyleGAN2/3/XL 自定义 Layer（如 persistence.Decorator 等）。
- 默认匹配参数名包含 "weight"（param_name_pattern=r".*weight.*"），可改为 ".*" 以注入全部参数（调试时使用）。
"""
import re
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List, Optional


class LoRALayer(nn.Module):
    """
    LoRA parameterization: returns original + delta.
    Supports 2D (Linear) and 4D (Conv) weight shapes.
    """
    def __init__(self, original_tensor: torch.Tensor, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        shape = tuple(original_tensor.shape)
        # Determine out/in dims for low-rank matrices
        if len(shape) == 4:  # Conv2d: (out, in, k, k)
            out_dim, in_dim = shape[0], shape[1]
            self.is_conv = True
        elif len(shape) == 2:  # Linear: (out, in)
            out_dim, in_dim = shape[0], shape[1]
            self.is_conv = False
        else:
            # Fallback: flatten trailing dims into "in_dim"
            out_dim = shape[0]
            in_dim = int(torch.tensor(shape[1:]).prod().item()) if len(shape) > 1 else 1
            self.is_conv = False

        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        # LoRA params: A (rank x in_dim), B (out_dim x rank)
        # Initialize A with small random values, B with zeros
        # This ensures that initially delta is zero, so the model behaves exactly like the pretrained model.
        self.lora_A = nn.Parameter(torch.randn(self.rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, self.rank))

        # toggle
        self.enabled = True

    def forward(self, orig: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return orig
        delta = self.lora_B @ self.lora_A  # (out, in)
        delta = delta * self.scaling
        if self.is_conv:
            delta = delta.unsqueeze(-1).unsqueeze(-1)  # (out, in, 1, 1)
        # cast to orig dtype/device
        delta = delta.to(dtype=orig.dtype, device=orig.device)
        return orig + delta


class LoRAInjector:
    """
    Robust injector:
    - model: typically generator.G.synthesis
    - param_name_pattern: regex to match parameter names (default matches 'weight')
    - layer_name_filter: optional callable(full_param_name)->bool to further filter
    """
    def __init__(self):
        self.lora_layers: List[LoRALayer] = []
        self._injected = []  # list of tuples (parent_module, param_name, lora_layer, full_param_name)
        self.layer_names: List[str] = []

    def _get_module_by_path(self, root: nn.Module, path: str) -> nn.Module:
        """Resolve dotted path relative to root ('' -> root)."""
        if path == "":
            return root
        module = root
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module

    def inject(self,
               model: nn.Module,
               rank: int = 4,
               alpha: float = 1.0,
               param_name_pattern: Optional[str] = r".*weight.*",
               layer_name_filter: Optional[callable] = None):
        """
        Inject parametrizations for parameters matching `param_name_pattern`.

        Args:
            model: root module to scan (e.g. generator.G.synthesis)
            rank: LoRA rank
            alpha: scaling factor
            param_name_pattern: regex to match parameter names (relative to module root),
                                default matches any name containing 'weight'
            layer_name_filter: optional callable(full_param_name)->bool to additionally filter by full param name
        """
        print("\n" + "=" * 70)
        print("  LoRA Injection Report (robust)")
        print("=" * 70)
        regex = re.compile(param_name_pattern)

        injected_count = 0
        total_params = 0

        # IMPORTANT: take a snapshot of named_parameters BEFORE registering parametrizations
        # to avoid "OrderedDict mutated during iteration" errors caused by register_parametrization.
        named_params_snapshot = list(model.named_parameters())

        for full_name, param in named_params_snapshot:
            # skip parameters that don't match pattern
            if not regex.search(full_name):
                continue

            # optional filter by full param name
            if layer_name_filter is not None and not layer_name_filter(full_name):
                continue

            # split parent module path and param attribute
            if '.' in full_name:
                module_path, p_name = full_name.rsplit('.', 1)
            else:
                module_path, p_name = "", full_name

            # resolve parent module relative to model
            try:
                parent = self._get_module_by_path(model, module_path)
            except Exception as e:
                print(f"  ⚠️  Could not resolve module for '{full_name}': {e}")
                continue

            # ensure attribute exists on parent
            if not hasattr(parent, p_name):
                print(f"  ⚠️  Parent module '{module_path}' has no attribute '{p_name}', skipping.")
                continue

            orig_tensor = getattr(parent, p_name)
            if not isinstance(orig_tensor, (torch.Tensor, torch.nn.Parameter)):
                print(f"  ⚠️  '{full_name}' is not a Tensor/Parameter (type={type(orig_tensor)}), skipping.")
                continue

            # create LoRA layer and register parametrization
            lora = LoRALayer(orig_tensor.data, rank=rank, alpha=alpha)
            try:
                nn.utils.parametrize.register_parametrization(parent, p_name, lora)
            except Exception as e:
                print(f"  ⚠️  Failed to register parametrization for '{full_name}': {e}")
                continue

            self.lora_layers.append(lora)
            self._injected.append((parent, p_name, lora, full_name))
            self.layer_names.append(full_name)
            cnt = sum(p.numel() for p in lora.parameters())
            injected_count += 1
            total_params += cnt
            print(f"  ✓ Injected LoRA at '{full_name}' (shape={tuple(param.shape)}, +{cnt:,} params)")

        print("-" * 70)
        print(f"Total Injected Layers: {injected_count}")
        print(f"Total LoRA Parameters: {total_params:,}")
        if injected_count == 0:
            print("⚠️  WARNING: No layers were injected! Try param_name_pattern='.*' or inspect named_parameters().")
        print("=" * 70 + "\n")

    def parameters(self) -> List[nn.Parameter]:
        """Return LoRA parameters for optimizer."""
        params = []
        for l in self.lora_layers:
            params.extend([l.lora_A, l.lora_B])
        return params

    @contextmanager
    def lora_disabled(self):
        prev = [l.enabled for l in self.lora_layers]
        try:
            for l in self.lora_layers:
                l.enabled = False
            yield
        finally:
            for l, p in zip(self.lora_layers, prev):
                l.enabled = p

    def enable_lora(self):
        for l in self.lora_layers:
            l.enabled = True

    def disable_lora(self):
        for l in self.lora_layers:
            l.enabled = False

    def get_state_dict(self):
        d = {}
        for i, l in enumerate(self.lora_layers):
            d[f"lora.{i}.A"] = l.lora_A.detach().cpu()
            d[f"lora.{i}.B"] = l.lora_B.detach().cpu()
            d[f"lora.{i}.alpha"] = l.alpha
            d[f"lora.{i}.rank"] = l.rank
        return d

    def merge_lora_to_base(self):
        """Merge deltas into base weights and remove parametrizations (irreversible)."""
        for parent, p_name, l, full_name in self._injected:
            try:
                nn.utils.parametrize.remove_parametrizations(parent, p_name)
            except Exception:
                pass
            with torch.no_grad():
                base = getattr(parent, p_name)
                delta = (l.lora_B @ l.lora_A) * l.scaling
                if l.is_conv:
                    delta = delta.unsqueeze(-1).unsqueeze(-1)
                base.data.add_(delta.to(base.device, dtype=base.dtype))