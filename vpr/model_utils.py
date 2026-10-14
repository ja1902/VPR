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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_fp16(device):
    return device.type == "cuda" and config.USE_FP16


def load_salad_model(device):
    sys.path.insert(0, str(SALAD_REPO))
    sys.path.insert(0, str(DINOV2_REPO))

    from models.backbones.dinov2 import DINOv2, DINOV2_ARCHS
    from models.aggregators.salad import SALAD as SALADAggregator, get_matching_probs
    from dinov2.models import vision_transformer as vits

    class EfficientSALAD(SALADAggregator):
        """SALAD aggregation without materialising the [B, l, m, n] repeat() tensors.

        The original computes (f.repeat(...) * p.repeat(...)).sum(-1), which allocates
        ~50 MB per image at 322px. The einsum below is mathematically identical.
        """

        def forward(self, x):
            x, t = x

            f = self.cluster_features(x).flatten(2)
            p = self.score(x).flatten(2)
            t = self.token_features(t)

            p = get_matching_probs(p, self.dust_bin, 3)
            p = torch.exp(p)[:, :-1, :]

            agg = torch.einsum("bln,bmn->blm", f, p)

            f = torch.cat([
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize(agg, p=2, dim=1).flatten(1)
            ], dim=-1)

            return nn.functional.normalize(f, p=2, dim=-1)

    backbone_name = "dinov2_vitb14"
    num_channels = DINOV2_ARCHS[backbone_name]

    # Backbone weights come from the SALAD checkpoint (loaded strictly below), so the
    # ViT is only constructed here, not initialised from the DINOv2 pretrain weights.
    dinov2_model = vits.vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        block_chunks=0,
    )

    backbone = DINOv2.__new__(DINOv2)
    nn.Module.__init__(backbone)
    backbone.model = dinov2_model
    backbone.num_channels = num_channels
    backbone.num_trainable_blocks = 4
    backbone.norm_layer = True
    backbone.return_token = True

    aggregator = EfficientSALAD(
        num_channels=num_channels,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
    )

    class SALADModel(nn.Module):
        def __init__(self, backbone, aggregator, backbone_dtype):
            super().__init__()
            self.backbone = backbone
            self.aggregator = aggregator
            self.backbone_dtype = backbone_dtype

        @torch.inference_mode()
        def forward(self, x):
            # The ViT runs in fp16 on CUDA; the Sinkhorn aggregation stays in fp32.
            f, t = self.backbone(x.to(self.backbone_dtype))
            return self.aggregator((f.float(), t.float()))

    backbone_dtype = torch.float16 if use_fp16(device) else torch.float32
    model = SALADModel(backbone, aggregator, backbone_dtype)
    state_dict = torch.load(SALAD_WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.backbone.to(backbone_dtype)
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
