from pathlib import Path
import torch
from argus import ARGUS
from argus.config import ARGUSConfig
from argus.sampling import AdaptiveSampler
from argus.losses import anomaly_loss, mil_ranking_loss
from argus.metrics import binary_metrics
from argus.data import VideoManifest, SyntheticSurveillance


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


def test_manifest_dataset_loader():
    manifest_path = Path('data/train.csv')
    if manifest_path.exists():
        dataset = VideoManifest(manifest_path, frames=8, image_size=64)
        assert len(dataset) > 0
        clip, label = dataset[0]
        assert clip.shape == (8, 3, 64, 64)
        assert label in (0.0, 1.0)


def test_binary_metrics():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    metrics = binary_metrics(scores, labels)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['auc_roc'] == 1.0


def test_config_serialization(tmp_path):
    cfg = ARGUSConfig(image_size=112, dim=256, temporal_layers=4)
    d = cfg.to_dict()
    assert d['image_size'] == 112
    assert d['dim'] == 256

    reconstructed = ARGUSConfig.from_dict(d)
    assert reconstructed.image_size == 112
    assert reconstructed.dim == 256
