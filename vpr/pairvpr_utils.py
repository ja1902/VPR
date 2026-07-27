"""
Pair-VPR re-ranking utilities.
Loads Pair-VPR from local files (bypasses torch.hub).
"""
import sys
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from . import config

PAIRVPR_REPO = config.BASE_DIR / "Pair-VPR"
PAIRVPR_WEIGHTS = config.BASE_DIR / "weights" / "pairvpr-vitB.pth"
DINOV2_REG_WEIGHTS = config.BASE_DIR / "weights" / "dinov2_vitb14_reg4_pretrain.pth"
DINOV2_REPO = config.BASE_DIR / "dinov2"


def _build_dinov2_reg_local():
    """Build DINOv2 ViT-B/14 with registers from local repo."""
    sys.path.insert(0, str(DINOV2_REPO))
    from dinov2.models import vision_transformer as vits

    model = vits.vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    state_dict = torch.load(DINOV2_REG_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def load_pair_vpr_model(device):
    """Load Pair-VPR model with local DINOv2 backbone."""
    sys.path.insert(0, str(PAIRVPR_REPO))
    os.environ["XFORMERS_DISABLED"] = "1"

    from pairvpr.configs import pairvpr_speed
    from pairvpr.models.pairvpr import PairVPRNet, DINOV2_ARCHS
    from pairvpr.models.tools.pos_embed import interpolate_pos_embed
    import pairvpr.models.dinov2 as pairvpr_dinov2

    cfg = OmegaConf.create(pairvpr_speed)

    original_init = pairvpr_dinov2.DINOv2.__init__

    def patched_init(self, model_name='dinov2_vitb14', num_trainable_blocks=2,
                     return_classtok=True):
        nn.Module.__init__(self)
        assert model_name in DINOV2_ARCHS, f'Unknown model name {model_name}'
        self.model = _build_dinov2_reg_local()
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.return_classtok = return_classtok

    pairvpr_dinov2.DINOv2.__init__ = patched_init

    try:
        model = PairVPRNet(cfg)
        ckpt = torch.load(PAIRVPR_WEIGHTS, map_location="cpu", weights_only=False)
        interpolate_pos_embed(cfg, model, ckpt)
        model.load_state_dict(ckpt, strict=False)
        model = model.to(device).eval()
    finally:
        pairvpr_dinov2.DINOv2.__init__ = original_init

    return model


def extract_dense_features(model, tensor, device):
    """Extract dense feature map and global descriptor from a preprocessed image tensor."""
    with torch.no_grad():
        feat_map, global_desc = model(tensor.to(device), None, mode="global")
    return feat_map, global_desc


def rerank(model, query_features, candidate_features, device):
    """
    Re-rank candidates using Pair-VPR pairwise scoring.

    Args:
        model: Loaded PairVPRNet
        query_features: Dense features for query image [1, N, D]
        candidate_features: Dense features for K candidates [K, N, D]
        device: torch device

    Returns:
        sorted_indices: Indices that sort candidates by similarity (best first)
        sorted_scores: Corresponding scores
    """
    K = candidate_features.shape[0]
    q_feats = query_features.expand(K, -1, -1).to(device)
    db_feats = candidate_features.to(device)

    with torch.no_grad():
        scores_a = model(q_feats, db_feats, "pairvpr")
        scores_b = model(db_feats, q_feats, "pairvpr")
        scores = scores_a + scores_b

    sorted_result = scores.cpu().squeeze(1).sort(descending=True)
    return sorted_result.indices, sorted_result.values
