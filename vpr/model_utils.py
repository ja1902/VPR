"""
Shared utilities for loading the DINOv2-SALAD model from local files.
"""
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from . import config

SALAD_REPO = config.BASE_DIR / "salad"
DINOV2_REPO = config.BASE_DIR / "dinov2"
SALAD_WEIGHTS = config.BASE_DIR / "weights" / "dino_salad.ckpt"
DINOV2_WEIGHTS = config.BASE_DIR / "weights" / "dinov2_vitb14_pretrain.pth"


def load_salad_model(device):
    sys.path.insert(0, str(SALAD_REPO))
    sys.path.insert(0, str(DINOV2_REPO))

    from models.backbones.dinov2 import DINOv2, DINOV2_ARCHS
    from models.aggregators.salad import SALAD as SALADAggregator
    from dinov2.models import vision_transformer as vits

    backbone_name = "dinov2_vitb14"
    num_channels = DINOV2_ARCHS[backbone_name]

    dinov2_model = vits.vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        block_chunks=0,
    )
    dinov2_model.load_state_dict(
        torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True)
    )

    backbone = DINOv2.__new__(DINOv2)
    nn.Module.__init__(backbone)
    backbone.model = dinov2_model
    backbone.num_channels = num_channels
    backbone.num_trainable_blocks = 4
    backbone.norm_layer = True
    backbone.return_token = True

    aggregator = SALADAggregator(
        num_channels=num_channels,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
    )

    class SALADModel(nn.Module):
        def __init__(self, backbone, aggregator):
            super().__init__()
            self.backbone = backbone
            self.aggregator = aggregator

        def forward(self, x):
            x = self.backbone(x)
            x = self.aggregator(x)
            return x

    model = SALADModel(backbone, aggregator)
    state_dict = torch.load(SALAD_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(
            (config.SALAD_INPUT_SIZE, config.SALAD_INPUT_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
