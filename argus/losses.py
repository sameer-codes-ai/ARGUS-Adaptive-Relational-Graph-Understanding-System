import torch
import torch.nn as nn
import torch.nn.functional as F

def anomaly_loss(pred_scores, targets, smoothness_weight=0.1, sparsity_weight=0.01):
    video_scores = pred_scores.max(dim=1).values
    bce = F.binary_cross_entropy(video_scores, targets.float())

    if pred_scores.shape[1] > 1:
        diff = pred_scores[:, 1:] - pred_scores[:, :-1]
        smooth = (diff ** 2).mean()
    else:
        smooth = torch.tensor(0.0, device=pred_scores.device)

    sparse = pred_scores.mean()
    total = bce + smoothness_weight * smooth + sparsity_weight * sparse
    return total, {'bce': bce.item(), 'smooth': smooth.item(), 'sparse': sparse.item(), 'total': total.item()}


def mil_ranking_loss(normal_scores, abnormal_scores, margin=1.0):
    max_normal = normal_scores.max(dim=1).values
    max_abnormal = abnormal_scores.max(dim=1).values
    loss = F.relu(margin - max_abnormal.mean() + max_normal.mean())
    return loss
