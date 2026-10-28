"""
Explainability Engine — Multi-modal anomaly attribution.

Implements Module 9 of the ARGUS research design:
  - Temporal attention map extraction from the causal transformer.
  - GNN gradient-based node importance attribution.
  - Spatial heatmap generation from bounding-box importances.

Usage::

    from argus.explain import ExplainabilityEngine

    engine = ExplainabilityEngine(model)
    engine.register_hooks()
    result = engine.compute_node_attribution(clip_batch)
    heatmap = engine.generate_spatial_heatmap(frame, boxes, importances)
    engine.cleanup()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class ExplainabilityEngine:
    """Multi-modal explainability for ARGUS anomaly predictions."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: list = []
        self._attention_maps: Dict[str, torch.Tensor] = {}

    # ── Hook management ───────────────────────────────────────────────
    def register_hooks(self) -> None:
        """Register forward hooks on the temporal encoder to capture attention weights."""
        self._clear_hooks()
        encoder = self.model.temporal_encoder.encoder
        for idx, layer in enumerate(encoder.layers):
            pre_hook = layer.self_attn.register_forward_pre_hook(
                self._make_pre_hook(), with_kwargs=True
            )
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(f"temporal_layer_{idx}")
            )
            self._hooks.append(pre_hook)
            self._hooks.append(hook)

    @staticmethod
    def _make_pre_hook():
        def _pre_hook(_module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs
        return _pre_hook

    def _make_hook(self, name: str):
        def _hook(_module, _input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_maps[name] = output[1].detach()
        return _hook

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attention_maps.clear()

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        self._clear_hooks()

    # ── Node attribution ──────────────────────────────────────────────
    @torch.enable_grad()
    def compute_node_attribution(
        self, frames: torch.Tensor, target_frame_idx: int = -1
    ) -> Dict:
        """
        Gradient-based importance attribution for graph nodes.

        Args:
            frames: ``(1, T, 3, H, W)`` single-clip input.
            target_frame_idx: index into the predicted score sequence to
                differentiate (``-1`` uses the frame with the highest score).
        Returns:
            dict with ``scores``, ``node_importances``, ``attention_maps``,
            ``sampled_indices``, ``boxes``.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model(frames)
            valid_length = int(out.sequence_lengths[0].item()) if out.sequence_lengths is not None else out.scores.shape[1]
            scores = out.scores[0, :valid_length]
            if target_frame_idx == -1:
                target = scores.max()
            else:
                if not -valid_length <= target_frame_idx < valid_length:
                    raise IndexError("target_frame_idx is outside the valid predicted sequence")
                target = scores[target_frame_idx]

            clip_node_imp = out.node_importance[0] if out.node_importance else []
            tensors = [emb for emb in clip_node_imp if isinstance(emb, torch.Tensor) and emb.requires_grad]
            gradients = torch.autograd.grad(target, tensors, allow_unused=True) if tensors else []
            gradient_by_id = {id(tensor): grad for tensor, grad in zip(tensors, gradients)}
            node_importances: List[Optional[torch.Tensor]] = []
            for emb in clip_node_imp:
                grad = gradient_by_id.get(id(emb)) if isinstance(emb, torch.Tensor) else None
                node_importances.append(grad.norm(dim=-1).detach().cpu() if grad is not None else None)
        finally:
            self.model.train(was_training)

        sampled = (
            out.sampled_indices[0].cpu().tolist()
            if isinstance(out.sampled_indices[0], torch.Tensor)
            else list(out.sampled_indices[0])
        )

        boxes_list = []
        if out.boxes and len(out.boxes) > 0:
            clip_boxes = out.boxes[0]
            if isinstance(clip_boxes, list):
                boxes_list = [
                    b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
                    for b in clip_boxes
                ]

        return {
            "scores": scores.detach().cpu().numpy(),
            "node_importances": node_importances,
            "attention_maps": {
                k: v.cpu().numpy() for k, v in self._attention_maps.items()
            },
            "sampled_indices": sampled,
            "boxes": boxes_list,
        }

    # ── Temporal attention ────────────────────────────────────────────
    def get_temporal_attention_map(self) -> Optional[np.ndarray]:
        """
        Return the ``(T, T)`` average temporal attention matrix from the
        last forward pass, or ``None`` if no hooks were active.
        """
        if not self._attention_maps:
            return None

        layers = []
        for attn in self._attention_maps.values():
            if attn.ndim == 4:          # (B, heads, T, T)
                layers.append(attn[0].mean(dim=0).cpu().numpy())
            elif attn.ndim == 3:        # (B*heads, T, T)
                layers.append(attn.mean(dim=0).cpu().numpy())
        return np.mean(layers, axis=0) if layers else None

    # ── Spatial heatmap ───────────────────────────────────────────────
    @staticmethod
    def generate_spatial_heatmap(
        frame: torch.Tensor,
        boxes: torch.Tensor,
        importances: torch.Tensor,
    ) -> np.ndarray:
        """
        Paint a ``(H, W)`` saliency heatmap from bounding-box importances.

        Args:
            frame: ``(3, H, W)`` reference frame (used only for its size).
            boxes: ``(N, 4)`` normalised ``xyxy`` boxes.
            importances: ``(N,)`` importance scores.
        Returns:
            ``(H, W)`` heatmap normalised to ``[0, 1]``.
        """
        h, w = frame.shape[-2:]
        heatmap = np.zeros((h, w), dtype=np.float32)
        if len(boxes) == 0 or len(importances) == 0:
            return heatmap

        imp = importances.detach().cpu().numpy() if isinstance(importances, torch.Tensor) else np.asarray(importances)
        if imp.max() > 0:
            imp = imp / imp.max()

        bx = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
        for box, score in zip(bx, imp):
            x1 = max(0, int(box[0] * w))
            y1 = max(0, int(box[1] * h))
            x2 = min(w, int(box[2] * w))
            y2 = min(h, int(box[3] * h))
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], score)
        return heatmap
