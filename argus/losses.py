import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def mil_ranking_loss(normal_scores, abnormal_scores, margin=1.0):
    if len(normal_scores) == 0 or len(abnormal_scores) == 0:
        return torch.tensor(0.0, device=normal_scores.device if len(normal_scores) > 0 else abnormal_scores.device)
    max_normal = normal_scores.max(dim=1).values
    max_abnormal = abnormal_scores.max(dim=1).values
    loss = F.relu(margin - max_abnormal.mean() + max_normal.mean())
    return loss


def rtfm_magnitude_loss(normal_features: torch.Tensor, abnormal_features: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Feature magnitude margin loss for RTFM (Tian et al., ICCV 2021).
    Enforces that maximum feature magnitude of abnormal snippets exceeds
    maximum normal snippet magnitude by at least `margin`.
    """
    if len(normal_features) == 0 or len(abnormal_features) == 0:
        device = normal_features.device if len(normal_features) > 0 else abnormal_features.device
        return torch.tensor(0.0, device=device)

    # Compute L2 norm of temporal feature vectors along channel dimension
    norm_mag = torch.norm(normal_features, p=2, dim=-1)   # (B_norm, T)
    abn_mag = torch.norm(abnormal_features, p=2, dim=-1) # (B_abn, T)

    max_norm = norm_mag.max(dim=1).values
    max_abn = abn_mag.max(dim=1).values
    loss = F.relu(margin - max_abn.mean() + max_norm.mean())
    return loss


def anomaly_loss(
    pred_scores,
    targets,
    smoothness_weight=0.1,
    sparsity_weight=0.01,
    mil_weight=0.5,
    valid_mask=None,
):
    if valid_mask is None:
        valid_mask = torch.ones_like(pred_scores, dtype=torch.bool)
    if valid_mask.shape != pred_scores.shape or not valid_mask.any(dim=1).all():
        raise ValueError("valid_mask must match pred_scores and contain at least one valid item per sample")
    # Clamp to avoid numerical precision errors (e.g. > 1.0 or log(0) in BCE)
    masked_scores = pred_scores.masked_fill(~valid_mask, float('-inf'))
    video_scores = masked_scores.max(dim=1).values.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(video_scores, targets.float())

    if pred_scores.shape[1] > 1:
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        diff = pred_scores[:, 1:] - pred_scores[:, :-1]
        smooth = (diff[pair_mask] ** 2).mean() if pair_mask.any() else pred_scores.new_tensor(0.0)
    else:
        smooth = torch.tensor(0.0, device=pred_scores.device)

    # Sparsity is a normal-video prior; applying it to abnormal clips suppresses
    # the very threat scores the model is learning to detect.
    normal_valid = valid_mask & (targets == 0.0).unsqueeze(1)
    sparse = pred_scores[normal_valid].mean() if normal_valid.any() else pred_scores.new_tensor(0.0)

    # MIL Ranking loss if batch contains both normal (0) and abnormal (1) clips
    abnormal_mask = (targets == 1.0)
    normal_mask = (targets == 0.0)
    if abnormal_mask.any() and normal_mask.any() and mil_weight > 0.0:
        normal_scores = pred_scores[normal_mask].masked_fill(~valid_mask[normal_mask], float('-inf'))
        abnormal_scores = pred_scores[abnormal_mask].masked_fill(~valid_mask[abnormal_mask], float('-inf'))
        mil_loss = mil_ranking_loss(normal_scores, abnormal_scores)
    else:
        mil_loss = torch.tensor(0.0, device=pred_scores.device)

    total = bce + smoothness_weight * smooth + sparsity_weight * sparse + mil_weight * mil_loss
    return total, {
        'bce': bce.item(),
        'smooth': smooth.item(),
        'sparse': sparse.item(),
        'mil': mil_loss.item(),
        'total': total.item()
    }


def compute_model_loss(
    model_type: str,
    out,
    targets: torch.Tensor,
    model: Optional[nn.Module] = None,
    smoothness_weight: float = 0.1,
    sparsity_weight: float = 0.01,
    mil_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Unified loss dispatcher tailored to each architecture's supervision paradigm:
      - argus & convnext_mil: BCE + Smoothness + Sparsity + Top-k MIL Ranking
      - causal_tsm: BCE + Temporal Smoothness
      - rtfm_lite: BCE + Smoothness + Feature Magnitude Ranking Loss
    """
    mtype = model_type.lower().strip()

    if mtype in {"causal_tsm", "tsm"}:
        return anomaly_loss(
            out.scores.float(),
            targets,
            smoothness_weight=smoothness_weight,
            sparsity_weight=0.0,
            mil_weight=0.0,
            valid_mask=out.valid_mask,
        )
    elif mtype in {"rtfm_lite", "rtfm"}:
        base_loss, ldict = anomaly_loss(
            out.scores.float(),
            targets,
            smoothness_weight=smoothness_weight,
            sparsity_weight=0.0,
            mil_weight=0.0,
            valid_mask=out.valid_mask,
        )
        temp_feats = getattr(model, "last_temporal_features", None)
        if temp_feats is not None:
            abn_mask = (targets == 1.0)
            norm_mask = (targets == 0.0)
            if abn_mask.any() and norm_mask.any() and mil_weight > 0.0:
                norm_f = temp_feats[norm_mask]
                abn_f = temp_feats[abn_mask]
                mag_loss = rtfm_magnitude_loss(norm_f, abn_f, margin=1.0)
            else:
                mag_loss = torch.tensor(0.0, device=targets.device)
            total = base_loss + mil_weight * mag_loss
            ldict["mag_loss"] = mag_loss.item()
            ldict["total"] = total.item()
            return total, ldict
        return base_loss, ldict
    else:
        return anomaly_loss(
            out.scores.float(),
            targets,
            smoothness_weight=smoothness_weight,
            sparsity_weight=sparsity_weight,
            mil_weight=mil_weight,
            valid_mask=out.valid_mask,
        )
