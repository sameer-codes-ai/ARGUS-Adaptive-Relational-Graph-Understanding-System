import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSampler(nn.Module):
    def __init__(self, threshold=0.12, max_gap=4):
        super().__init__()
        self.threshold = threshold
        self.max_gap = max_gap

    def importance_scores(self, frames):
        b, t = frames.shape[:2]
        gray = frames.mean(dim=2)
        if t <= 1:
            return torch.ones((b, t), device=frames.device)

        diff = (gray[:, 1:] - gray[:, :-1]).abs().mean(dim=(-1, -2))
        grad_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(-1, -2))
        grad_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=(-1, -2))
        spatial_energy = (grad_x + grad_y)[:, 1:]

        flat = gray[:, 1:].flatten(2)
        p = F.softmax(flat, dim=-1)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
        entropy = (entropy - entropy.mean(dim=-1, keepdim=True)) / (entropy.std(dim=-1, keepdim=True) + 1e-6)

        hist_diff = (gray[:, 1:] - gray[:, :-1]).pow(2).mean(dim=(-1, -2))

        step_score = diff + 0.3 * spatial_energy + 0.1 * F.relu(entropy) + 0.2 * hist_diff
        first = step_score[:, :1]
        scores = torch.cat([first, step_score], dim=1)
        return scores

    def select(self, frames):
        b, t = frames.shape[:2]
        scores = self.importance_scores(frames)
        selected_batches = []
        for i in range(b):
            chosen = [0]
            last = 0
            for k in range(1, t):
                gap = k - last
                is_above = bool((scores[i, k] >= self.threshold).item())
                if is_above or gap >= self.max_gap or k == t - 1:
                    chosen.append(k)
                    last = k
            selected_batches.append(torch.tensor(sorted(list(set(chosen))), device=frames.device, dtype=torch.long))
        return selected_batches, scores
