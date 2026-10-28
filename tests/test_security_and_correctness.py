from pathlib import Path

import numpy as np
import pytest
import torch

import web_demo
from argus import ARGUS, ARGUSConfig, AlertDispatcher, anomaly_loss, binary_metrics
from argus.data import VideoManifest


def test_static_path_traversal_is_rejected():
    assert web_demo.resolve_static_path('/demo_samples/../.git/config') is None
    assert web_demo.resolve_static_path('/demo_samples/../../Windows/win.ini') is None


def test_static_path_inside_allowed_root_is_accepted():
    sample = Path('demo_samples/sample_normal_NV344.mp4')
    if sample.exists():
        assert web_demo.resolve_static_path('/demo_samples/sample_normal_NV344.mp4') == sample.resolve()


def test_unknown_model_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown model configuration"):
        ARGUSConfig.from_dict({'dim': 64, 'typo_dim': 64})


def test_invalid_model_dimensions_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ARGUSConfig(dim=63, graph_dim=32, heads=4).validate()


def test_masked_loss_ignores_padding_and_does_not_suppress_abnormal_scores():
    scores = torch.tensor([[0.8, 0.0], [0.1, 0.2]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])
    labels = torch.tensor([1.0, 0.0])
    loss, parts = anomaly_loss(scores, labels, valid_mask=mask, mil_weight=0.0)
    assert torch.isfinite(loss)
    assert parts['sparse'] == pytest.approx(0.15)


def test_non_streaming_call_does_not_consume_stream_memory():
    cfg = ARGUSConfig(image_size=32, patch_size=16, dim=32, graph_dim=16, heads=4, temporal_layers=1)
    model = ARGUS(cfg).eval()
    clip = torch.rand(1, 3, 3, 32, 32)
    with torch.no_grad():
        baseline = model(clip).scores.clone()
        model(clip, stream=True, stream_id='camera-a')
        after_stream = model(clip).scores
    assert torch.allclose(baseline, after_stream)


def test_stream_memory_is_isolated_by_source():
    cfg = ARGUSConfig(image_size=32, patch_size=16, dim=32, graph_dim=16, heads=4, temporal_layers=1)
    model = ARGUS(cfg).eval()
    clip = torch.rand(1, 3, 3, 32, 32)
    with torch.no_grad():
        model(clip, stream=True, stream_id='camera-a')
        model(clip, stream=True, stream_id='camera-b')
    assert set(model.memory._stream_banks) == {'camera-a', 'camera-b'}


def test_tied_metrics_are_order_invariant():
    predictions = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    first = binary_metrics(predictions, labels)
    second = binary_metrics(predictions[::-1], labels[::-1])
    assert first['auc_roc'] == second['auc_roc'] == 0.5
    assert first['average_precision'] == second['average_precision'] == 0.5


def test_relative_manifest_paths_resolve_from_manifest_directory(tmp_path):
    media = tmp_path / 'clip.pt'
    torch.save(torch.rand(2, 3, 8, 8), media)
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('path,label\nclip.pt,1\n', encoding='utf-8')
    dataset = VideoManifest(manifest, frames=2, image_size=8)
    assert Path(dataset.records[0]['path']) == media.resolve()


def test_missing_checkpoint_is_not_silently_randomized(tmp_path):
    with pytest.raises(FileNotFoundError):
        web_demo.load_model(tmp_path / 'missing.pt')


def test_normal_event_does_not_claim_anomaly_or_notify():
    received = []
    event = AlertDispatcher(threshold=0.5, callbacks=[received.append]).dispatch(
        'camera-1', [0.1, 0.2], [0, 1]
    )
    assert event is not None
    assert 'No anomaly detected' in event.summary
    assert event.notification_sent is False
    assert received == []
