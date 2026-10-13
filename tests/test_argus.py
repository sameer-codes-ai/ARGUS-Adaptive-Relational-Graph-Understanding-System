from pathlib import Path
import torch
from argus import (
    ARGUS,
    ARGUSConfig,
    AdaptiveSampler,
    SpatialBackbone,
    SemanticDetector,
    RoIFeatureExtractor,
    DynamicSceneGraphBuilder,
    ExplainabilityEngine,
    AlertDispatcher,
    AlertSeverity,
    generate_dashboard_html,
    anomaly_loss,
    mil_ranking_loss,
    binary_metrics,
    VideoManifest,
    SyntheticSurveillance
)


def test_sampler_keeps_temporal_coverage():
    clip = torch.zeros(1, 12, 3, 32, 32)
    picked, scores = AdaptiveSampler(threshold=99, max_gap=3).select(clip)
    assert picked[0].tolist() == [0, 3, 6, 9, 11]
    assert scores.shape == (1, 12)


def test_end_to_end_forward_and_loss():
    model = ARGUS()
    out = model(torch.rand(2, 8, 3, 96, 96))
    assert out.scores.shape[0] == 2 and out.scores.ndim == 2
    assert len(out.sampled_indices) == 2
    assert len(out.boxes) == 2

    loss, loss_dict = anomaly_loss(out.scores, torch.tensor([0., 1.]))
    assert torch.isfinite(loss)
    assert 'bce' in loss_dict and 'smooth' in loss_dict


def test_mil_ranking_loss():
    normal_scores = torch.tensor([[0.1, 0.2], [0.05, 0.15]])
    abnormal_scores = torch.tensor([[0.8, 0.9], [0.7, 0.85]])
    loss = mil_ranking_loss(normal_scores, abnormal_scores, margin=1.0)
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_streaming_memory_is_bounded():
    model = ARGUS()
    model(torch.rand(1, 8, 3, 96, 96), stream=True)
    model(torch.rand(1, 8, 3, 96, 96), stream=True)
    assert model.memory.bank is not None
    assert model.memory.bank.shape[1] <= model.cfg.memory_size


def test_synthetic_dataset_loader():
    dataset = SyntheticSurveillance(count=8, frames=16, size=96)
    assert len(dataset) == 8
    clip, label = dataset[0]
    assert clip.shape == (16, 3, 96, 96)
    assert label in (0.0, 1.0)


def test_binary_metrics():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    metrics = binary_metrics(scores, labels)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['auc_roc'] == 1.0


def test_config_serialization():
    cfg = ARGUSConfig(image_size=112, dim=256, temporal_layers=4)
    d = cfg.to_dict()
    assert d['image_size'] == 112
    assert d['dim'] == 256

    reconstructed = ARGUSConfig.from_dict(d)
    assert reconstructed.image_size == 112
    assert reconstructed.dim == 256


# ── Outer Layer Unit Tests ───────────────────────────────────────────────

def test_spatial_backbone():
    backbone = SpatialBackbone(out_dim=128, frozen=True)
    dummy = torch.randn(2, 3, 64, 64)
    tokens, feat_maps = backbone(dummy)
    assert tokens.ndim == 3 and tokens.shape[0] == 2
    assert tokens.shape[-1] == 128
    assert len(feat_maps) == 4


def test_semantic_detector_and_roi():
    detector = SemanticDetector(max_objects=8)
    frame = torch.rand(3, 96, 96)
    res = detector(frame)
    assert res.boxes.ndim == 2 and res.boxes.shape[1] == 4
    assert res.semantic_types.shape[0] == res.boxes.shape[0]

    roi_extractor = RoIFeatureExtractor(in_channels=64, out_dim=32)
    feat_map = torch.randn(64, 12, 12)
    roi_feats = roi_extractor(feat_map, res.boxes)
    assert roi_feats.shape == (res.boxes.shape[0], 32)


def test_semantic_scene_graph_relations():
    builder = DynamicSceneGraphBuilder(max_objects=6, use_semantic=True)
    frame = torch.rand(3, 64, 64)
    node_feats, (src, dst), edge_attr, boxes = builder.build(frame)
    # Semantic node features have at least 10 dims
    assert node_feats.shape[-1] >= 10
    # Semantic edge features have 7 dims (4 kinematic + 3 relational: P2P, P2O, O2O)
    assert edge_attr.shape[-1] == 7


def test_dynamic_attention_modulation():
    sampler = AdaptiveSampler()
    scores = torch.rand(2, 8)
    chosen = [torch.tensor([0, 2, 4, 7]), torch.tensor([0, 1, 3, 5, 7])]
    mod = sampler.compute_attention_modulation(scores, chosen, max_t=5)
    assert mod.shape == (2, 5)
    assert (mod >= 0.0).all() and (mod <= 1.0).all()


def test_outer_layer_end_to_end_forward():
    cfg = ARGUSConfig(
        image_size=64,
        dim=128,
        graph_dim=64,
        use_pretrained_backbone=True,
        use_semantic_detector=True,
        node_visual_dim=32,
        temporal_layers=2,
        heads=4
    )
    model = ARGUS(config=cfg)
    clip = torch.rand(1, 6, 3, 64, 64)
    out = model(clip)
    assert out.scores.shape[0] == 1
    assert torch.isfinite(out.scores).all()


def test_explainability_and_alerts(tmp_path):
    model = ARGUS()
    engine = ExplainabilityEngine(model)
    clip = torch.rand(1, 4, 3, 64, 64)
    res = engine.compute_node_attribution(clip)
    assert 'scores' in res and 'sampled_indices' in res
    engine.cleanup()

    dispatcher = AlertDispatcher(threshold=0.5)
    event = dispatcher.dispatch(
        video_source="test_cam_01.mp4",
        timeline_scores=[0.2, 0.4, 0.85, 0.3],
        keyframe_indices=[0, 1, 2, 3]
    )
    assert event is not None
    assert event.is_anomaly is True
    assert event.severity == AlertSeverity.HIGH

    dash_path = tmp_path / "dashboard.html"
    generate_dashboard_html(event, dash_path)
    assert dash_path.exists()
    assert dash_path.stat().st_size > 500
