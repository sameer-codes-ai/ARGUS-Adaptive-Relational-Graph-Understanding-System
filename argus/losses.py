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


def anomaly_loss(pred_scores, targets, smoothness_weight=0.1, sparsity_weight=0.01, mil_weight=0.5):
    # pred_scores: (B, T), targets: (B,)
    video_scores = pred_scores.max(dim=1).values
    bce = F.binary_cross_entropy(video_scores, targets.float())

    if pred_scores.shape[1] > 1:
        diff = pred_scores[:, 1:] - pred_scores[:, :-1]
        smooth = (diff ** 2).mean()
    else:
        smooth = torch.tensor(0.0, device=pred_scores.device)

    sparse = pred_scores.mean()

    # MIL Ranking loss if batch contains both normal (0) and abnormal (1) clips
    abnormal_mask = (targets == 1.0)
    normal_mask = (targets == 0.0)
    if abnormal_mask.any() and normal_mask.any() and mil_weight > 0.0:
        mil_loss = mil_ranking_loss(pred_scores[normal_mask], pred_scores[abnormal_mask])
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

