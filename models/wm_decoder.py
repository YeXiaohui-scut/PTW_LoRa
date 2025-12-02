import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights


class WatermarkDecoder(nn.Module):
    def __init__(self, bitlen: int, decoder_arch: str):
        """ A watermark decoder trained with transfer learning. """
        super(WatermarkDecoder, self).__init__()
        self.model_type = decoder_arch

        # Register mean/std buffers for on-device normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)
        if decoder_arch == "resnet18":
            base_model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.hidden_size = 512
            self.dense = nn.Linear(self.hidden_size, bitlen)
        elif decoder_arch == "resnet50":
            base_model1 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, bitlen)
            )
        elif decoder_arch == "resnet101":
            base_model1 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, bitlen)
            )
        else:
            raise ValueError(decoder_arch)
        self.base_model1 = torch.nn.Sequential(*list(base_model1.children())[:-1])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor input for ResNet backbones.

        Args:
            x: Tensor in [-1, 1], shape [B, C, H, W]
        Returns:
            Normalized tensor in Imagenet space, resized to 224x224.
        """
        # Ensure finite values and correct dtype
        x = torch.clamp(x, -1.0, 1.0)
        x = x.to(dtype=torch.float32)
        # Resize to 224x224 using bilinear interpolation
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Map from [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize using Imagenet stats (buffers already on correct device)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return x

    def forward(self, image1):
        image1 = self.preprocess(image1)
        f1 = self.base_model1(image1)
        return self.dense(f1.view(-1, self.hidden_size))

