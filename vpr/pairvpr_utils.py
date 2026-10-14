"""
Pair-VPR re-ranking utilities.
Loads Pair-VPR from local files (bypasses torch.hub).
"""
import sys
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from . import config
from .model_utils import get_transform, use_fp16

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


def _use_fused_attention(blocks):
    """Replace Pair-VPR's naive decoder attention (which materialises the full
    [B, heads, N, N] matrix in fp32) with PyTorch's fused scaled_dot_product_attention."""
    F = nn.functional

    def self_attn_forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

    def cross_attn_forward(self, query, key, value):
        B, Nq, C = query.shape
        heads = self.num_heads
        q = self.projq(query).reshape(B, Nq, heads, C // heads).transpose(1, 2)
        k = self.projk(key).reshape(B, key.shape[1], heads, C // heads).transpose(1, 2)
        v = self.projv(value).reshape(B, value.shape[1], heads, C // heads).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, Nq, C))

    blocks.Attention.forward = self_attn_forward
    blocks.CrossAttention.forward = cross_attn_forward


def load_pair_vpr_model(device):
    """Load Pair-VPR model with local DINOv2 backbone."""
    sys.path.insert(0, str(PAIRVPR_REPO))
    os.environ["XFORMERS_DISABLED"] = "1"

    from pairvpr.configs import pairvpr_speed
    from pairvpr.models.pairvpr import PairVPRNet, DINOV2_ARCHS
    from pairvpr.models.tools.pos_embed import interpolate_pos_embed
    from pairvpr.models.tools import blocks
    import pairvpr.models.dinov2 as pairvpr_dinov2

    _use_fused_attention(blocks)
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
        del ckpt
        if use_fp16(device):
            model = model.half()
        model = model.to(device).eval()
    finally:
        pairvpr_dinov2.DINOv2.__init__ = original_init

    return model


def _model_dtype(model):
    return next(model.parameters()).dtype


def extract_dense_features(model, tensor, device):
    """Extract dense feature map and global descriptor from a preprocessed image tensor."""
    with torch.inference_mode():
        feat_map, global_desc = model(tensor.to(device, _model_dtype(model)), None, mode="global")
    return feat_map, global_desc


def encode_batched(model, tensors, device, batch_size=config.PAIRVPR_BATCH_SIZE):
    """Dense features for a stack of preprocessed images [K, 3, H, W] -> [K, N, D]."""
    feats = []
    for i in range(0, len(tensors), batch_size):
        feat_map, _ = extract_dense_features(model, tensors[i:i + batch_size], device)
        feats.append(feat_map)
    return torch.cat(feats)


def rerank(model, query_features, candidate_features, device,
           batch_size=config.PAIRVPR_BATCH_SIZE):
    """
    Re-rank candidates using Pair-VPR pairwise scoring.

    Args:
        model: Loaded PairVPRNet
        query_features: Dense features for query image [1, N, D]
        candidate_features: Dense features for K candidates [K, N, D]
        device: torch device
        batch_size: Candidate pairs scored per forward pass (bounds peak VRAM)

    Returns:
        sorted_indices: Indices that sort candidates by similarity (best first)
        sorted_scores: Corresponding scores
    """
    query_features = query_features.to(device)
    scores = []
    with torch.inference_mode():
        for i in range(0, candidate_features.shape[0], batch_size):
            db_feats = candidate_features[i:i + batch_size].to(device)
            q_feats = query_features.expand(db_feats.shape[0], -1, -1)
            scores_a = model(q_feats, db_feats, "pairvpr")
            scores_b = model(db_feats, q_feats, "pairvpr")
            scores.append((scores_a + scores_b).float().cpu())

    sorted_result = torch.cat(scores).squeeze(1).sort(descending=True)
    return sorted_result.indices, sorted_result.values


def rerank_results(query_img, faiss_results, device):
    """Load Pair-VPR and re-rank FAISS results against a query PIL image.

    Returns a list of {"score", "meta", "original_rank"} dicts, best first.
    """
    model = load_pair_vpr_model(device)
    transform = get_transform()

    query_feats, _ = extract_dense_features(model, transform(query_img).unsqueeze(0), device)
    cand_tensors = torch.stack([
        transform(Image.open(r["meta"]["image_path"]).convert("RGB")) for r in faiss_results
    ])
    candidate_feats = encode_batched(model, cand_tensors, device)
    sorted_indices, sorted_scores = rerank(model, query_feats, candidate_feats, device)

    del model, candidate_feats
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return [
        {
            "score": score.item(),
            "meta": faiss_results[idx.item()]["meta"],
            "original_rank": idx.item() + 1,
        }
        for idx, score in zip(sorted_indices, sorted_scores)
    ]
