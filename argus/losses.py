import torch
import torch.nn as nn
import torch.nn.functional as F

def mil_ranking_loss(normal_scores, abnormal_scores, margin=1.0):
    if len(normal_scores) == 0 or len(abnormal_scores) == 0:
        return torch.tensor(0.0, device=normal_scores.device if len(normal_scores) > 0 else abnormal_scores.device)
    max_normal = normal_scores.max(dim=1).values
    max_abnormal = abnormal_scores.max(dim=1).values
    loss = F.relu(margin - max_abnormal.mean() + max_normal.mean())
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

