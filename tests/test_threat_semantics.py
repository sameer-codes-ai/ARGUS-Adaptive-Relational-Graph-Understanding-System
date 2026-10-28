"""
Comprehensive test suite for ARGUS Expanded Semantic Graph & Threat Recognition.

Tests all requirements from Phase 1 through Phase 7:
  - COCO class 43 maps to weapon (3)
  - COCO bag classes (24, 26, 28) map to bag (2)
  - Unknown classes map to other object (4)
  - Semantic graph nodes have 12 dimensions (6 kinematic + 5 one-hot + 1 confidence)
  - Weapon/person edges receive the correct relation flags (P2W, P2B, W2O)
  - Heuristic mode never claims a weapon detection
  - Old legacy3 checkpoints are rejected or handled only through explicit partial loading
  - Full model forward pass works with empty detections, weapon detections, and mixed entity types
  - Explainability and alert events include threat-aware fields and evidence descriptions
"""

import pytest
import torch
import torch.nn as nn

from argus.detector import (
    SemanticDetector,
    DetectionResult,
    _semantic_type,
    SEMANTIC_TYPE_PERSON,
    SEMANTIC_TYPE_VEHICLE,
    SEMANTIC_TYPE_BAG,
    SEMANTIC_TYPE_WEAPON,
    SEMANTIC_TYPE_OTHER,
    NUM_SEMANTIC_CLASSES,
    COCO_TO_ARGUS,
)
from argus.graph import (
    DynamicSceneGraphBuilder,
    EdgeGAT,
    KINEMATIC_NODE_DIM,
    SEMANTIC_NODE_DIM,
    THREAT_RELATIONAL_EDGE_DIM,
    RELATIONAL_EDGE_DIM,
)
from argus.config import ARGUSConfig
from argus.model import ARGUS, ARGUSOutput
from argus.explain import ExplainabilityEngine
from argus.alerts import AlertDispatcher, AlertEvent
from argus.checkpoint import (
    validate_component_metadata,
    load_partial_checkpoint,
    IncompatibleCheckpointError,
)


# ── Phase 1 Tests: Detector Mapping ─────────────────────────────────────
def test_coco_mapping_weapons():
    """COCO class 43 (knife) must map to weapon (3)."""
    assert _semantic_type(43) == SEMANTIC_TYPE_WEAPON
    assert SEMANTIC_TYPE_WEAPON == 3


def test_coco_mapping_bags():
    """COCO bag classes (24=backpack, 26=handbag, 28=suitcase) must map to bag (2)."""
    assert _semantic_type(24) == SEMANTIC_TYPE_BAG
    assert _semantic_type(26) == SEMANTIC_TYPE_BAG
    assert _semantic_type(28) == SEMANTIC_TYPE_BAG
    assert SEMANTIC_TYPE_BAG == 2


def test_coco_mapping_person_and_vehicles():
    """Person (0) and vehicles (1..8) map to person (0) and vehicle (1)."""
    assert _semantic_type(0) == SEMANTIC_TYPE_PERSON
    assert _semantic_type(2) == SEMANTIC_TYPE_VEHICLE  # car
    assert _semantic_type(3) == SEMANTIC_TYPE_VEHICLE  # motorcycle
    assert _semantic_type(5) == SEMANTIC_TYPE_VEHICLE  # bus
    assert _semantic_type(7) == SEMANTIC_TYPE_VEHICLE  # truck


def test_coco_mapping_unknown_classes():
    """Unknown or unmapped COCO classes must map to other object (4)."""
    assert _semantic_type(999) == SEMANTIC_TYPE_OTHER
    assert _semantic_type(15) == SEMANTIC_TYPE_OTHER  # bench
    assert _semantic_type(56) == SEMANTIC_TYPE_OTHER  # chair
    assert _semantic_type(-1) == SEMANTIC_TYPE_OTHER
    assert SEMANTIC_TYPE_OTHER == 4


def test_detection_result_stores_cids_and_scores():
    """DetectionResult keeps original COCO class IDs, confidences, and detector_impl."""
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    scores = torch.tensor([0.88])
    cids = torch.tensor([43])
    sem = torch.tensor([SEMANTIC_TYPE_WEAPON])

    res = DetectionResult(
        boxes=boxes,
        scores=scores,
        class_ids=cids,
        semantic_types=sem,
        detector_impl="yolo"
    )
    assert res.class_ids[0].item() == 43
    assert abs(res.scores[0].item() - 0.88) < 1e-4
    assert res.semantic_types[0].item() == 3
    assert res.detector_impl == "yolo"


# ── Phase 4 Tests: Fallback Mode Honesty ────────────────────────────────
def test_heuristic_fallback_never_claims_weapon_or_bag():
    """In heuristic fallback mode, the detector must NEVER emit fake weapon (3) or bag (2) labels."""
    det = SemanticDetector(max_objects=8, model_name="heuristic")
    det.yolo = None  # Force fallback mode

    # Test on multiple varied frame shapes/energies
    for seed in range(5):
        torch.manual_seed(seed)
        frame = torch.rand(3, 96, 96)
        res = det(frame)

        assert res.detector_impl in {"heuristic", "grid"}
        # Must only emit Person (0), Vehicle (1), or Other Object (4)
        assert not ((res.semantic_types == SEMANTIC_TYPE_WEAPON) | (res.semantic_types == SEMANTIC_TYPE_BAG)).any()
        assert ((res.semantic_types == SEMANTIC_TYPE_PERSON) |
                (res.semantic_types == SEMANTIC_TYPE_VEHICLE) |
                (res.semantic_types == SEMANTIC_TYPE_OTHER)).all()


# ── Phase 2 Tests: Graph Node Dimensions ────────────────────────────────
def test_semantic_graph_nodes_12_dimensions():
    """Each semantic graph node must contain exactly 12 dimensions: 6 kinematics + 5 one-hot + 1 confidence."""
    assert SEMANTIC_NODE_DIM == 12

    builder = DynamicSceneGraphBuilder(max_objects=6, use_semantic=True, semantic_schema_version="threat5")
    frame = torch.rand(3, 64, 64)

    # Synthetic detections: Person (0) and Weapon (3)
    det = DetectionResult(
        boxes=torch.tensor([[0.1, 0.1, 0.3, 0.5], [0.15, 0.15, 0.25, 0.3]]),
        scores=torch.tensor([0.92, 0.84]),
        class_ids=torch.tensor([0, 43]),
        semantic_types=torch.tensor([SEMANTIC_TYPE_PERSON, SEMANTIC_TYPE_WEAPON]),
        detector_impl="yolo"
    )
    nodes, edges, edge_attrs, boxes = builder.build(frame, detection_result=det)
    assert nodes.shape == (2, 12)

    # Check semantic one-hot values
    # Node 0 (Person): index 6:11 is [1, 0, 0, 0, 0], index 11 is conf 0.92
    assert torch.allclose(nodes[0, 6:11], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert abs(nodes[0, 11].item() - 0.92) < 1e-4

    # Node 1 (Weapon): index 6:11 is [0, 0, 0, 1, 0], index 11 is conf 0.84
    assert torch.allclose(nodes[1, 6:11], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
    assert abs(nodes[1, 11].item() - 0.84) < 1e-4


def test_empty_graph_node_and_edge_dimensions():
    """Empty graphs must produce correct 12-dimensional node tensors and 10-dimensional edge tensors."""
    builder = DynamicSceneGraphBuilder(max_objects=6, use_semantic=True, semantic_schema_version="threat5")
    frame = torch.rand(3, 64, 64)

    empty_det = DetectionResult(
        boxes=torch.empty((0, 4)),
        scores=torch.empty((0,)),
        class_ids=torch.empty((0,), dtype=torch.long),
        semantic_types=torch.empty((0,), dtype=torch.long),
        detector_impl="yolo"
    )
    nodes, (src, dst), edge_attrs, boxes = builder.build(frame, detection_result=empty_det)
    assert nodes.shape == (0, 12)
    assert edge_attrs.shape == (0, 10)
    assert src.shape == (0,)
    assert dst.shape == (0,)


def test_non_semantic_grid_path_keeps_6_dimensions():
    """Non-semantic / kinematic graphs must keep their 6-dimensional node and 4-dimensional edge shapes."""
    assert KINEMATIC_NODE_DIM == 6
    builder = DynamicSceneGraphBuilder(max_objects=4, use_semantic=False)
    frame = torch.rand(3, 64, 64)

    det = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
    nodes, (src, dst), edge_attrs, boxes = builder.build(frame, detection_result=det)
    assert nodes.shape == (1, 6)
    assert edge_attrs.shape[-1] == 4


# ── Phase 3 Tests: Threat-Aware Relational Edges ─────────────────────────
def test_threat_relation_edge_encodings():
    """
    Graph edge features (10 dims) must correctly encode:
      [dist, iou, rel_vel, persistence, is_p2p, is_p2o, is_o2o, is_p2w, is_p2b, is_w2o]
    """
    assert THREAT_RELATIONAL_EDGE_DIM == 10

    builder = DynamicSceneGraphBuilder(max_objects=6, dist_threshold=0.8, use_semantic=True, semantic_schema_version="threat5")
    frame = torch.rand(3, 64, 64)

    # 3 entities placed close to each other:
    # 0: Person (0)
    # 1: Weapon (3)
    # 2: Bag (2)
    det = DetectionResult(
        boxes=torch.tensor([
            [0.20, 0.20, 0.40, 0.60],  # Person
            [0.25, 0.25, 0.35, 0.45],  # Weapon close to person
            [0.22, 0.22, 0.38, 0.48],  # Bag close to person & weapon
        ]),
        scores=torch.tensor([0.95, 0.85, 0.75]),
        class_ids=torch.tensor([0, 43, 24]),
        semantic_types=torch.tensor([SEMANTIC_TYPE_PERSON, SEMANTIC_TYPE_WEAPON, SEMANTIC_TYPE_BAG]),
        detector_impl="yolo"
    )
    nodes, (src, dst), edge_attrs, boxes = builder.build(frame, detection_result=det)

    assert edge_attrs.shape[-1] == 10

    # Look up edge from Person (0) to Weapon (1)
    p2w_found = False
    for idx, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        attrs = edge_attrs[idx].tolist()
        # [d, iou, rel_vel, persist, is_p2p, is_p2o, is_o2o, is_p2w, is_p2b, is_w2o]
        if (u, v) == (0, 1) or (u, v) == (1, 0):
            p2w_found = True
            assert attrs[4] == 0.0  # is_p2p
            assert attrs[5] == 1.0  # is_p2o
            assert attrs[6] == 0.0  # is_o2o
            assert attrs[7] == 1.0  # is_p2w (Person-to-Weapon!)
            assert attrs[8] == 0.0  # is_p2b
            assert attrs[9] == 0.0  # is_w2o (one endpoint is person)

        elif (u, v) == (0, 2) or (u, v) == (2, 0):
            assert attrs[4] == 0.0  # is_p2p
            assert attrs[5] == 1.0  # is_p2o
            assert attrs[6] == 0.0  # is_o2o
            assert attrs[7] == 0.0  # is_p2w
            assert attrs[8] == 1.0  # is_p2b (Person-to-Bag!)
            assert attrs[9] == 0.0  # is_w2o

        elif (u, v) == (1, 2) or (u, v) == (2, 1):
            assert attrs[4] == 0.0  # is_p2p
            assert attrs[5] == 0.0  # is_p2o
            assert attrs[6] == 1.0  # is_o2o
            assert attrs[7] == 0.0  # is_p2w
            assert attrs[8] == 0.0  # is_p2b
            assert attrs[9] == 1.0  # is_w2o (Weapon-to-Object!)

    assert p2w_found, "Person-to-Weapon edge must exist and be encoded!"


# ── Phase 6 Tests: Checkpoint & Schema Compatibility ────────────────────
def test_legacy3_checkpoint_rejection():
    """An old legacy3 checkpoint must be rejected clearly when loaded into a threat5 model."""
    cfg = ARGUSConfig(dim=64, graph_dim=32, temporal_layers=1, heads=4, use_semantic_detector=True, semantic_schema_version="threat5")
    model = ARGUS(config=cfg)

    # Construct dummy legacy3 checkpoint dictionary
    legacy_gat = EdgeGAT(in_dim=10, edge_dim=7, out_dim=32, heads=4)
    meta = model.component_metadata()
    dummy_legacy_ckpt = {
        "model": {
            "edge_gat.proj.weight": legacy_gat.proj.weight,
            "edge_gat.edge_proj.weight": legacy_gat.edge_proj.weight,
        },
        "semantic_schema_version": "legacy3",
        "components": {
            "backbone_implementation": meta["backbone_implementation"],
            "backbone_name": meta["backbone_name"],
            "detector_implementation": meta["detector_implementation"],
            "detector_model": meta["detector_model"],
            "semantic_schema_version": "legacy3",
        }
    }

    with pytest.raises(IncompatibleCheckpointError) as exc_info:
        validate_component_metadata(dummy_legacy_ckpt, model)
    assert "Incompatible checkpoint" in str(exc_info.value)
    assert "legacy3" in str(exc_info.value)


def test_partial_checkpoint_loading_skips_edge_gat():
    """Partial initialization from an old checkpoint loads unchanged modules and initializes EdgeGAT fresh."""
    cfg = ARGUSConfig(dim=64, graph_dim=32, temporal_layers=1, heads=4, use_semantic_detector=True, semantic_schema_version="threat5")
    model = ARGUS(config=cfg)

    # Save initial EdgeGAT weights
    initial_gat_weight = model.edge_gat.proj.weight.clone()

    # Create dummy checkpoint with head and temporal weights matching, but legacy edge_gat
    old_state = model.state_dict()
    # Replace edge_gat with 10-dim legacy tensor
    old_state["edge_gat.proj.weight"] = torch.randn(32, 10)

    ckpt = {
        "model": old_state,
        "semantic_schema_version": "legacy3"
    }

    result = load_partial_checkpoint(model, ckpt)
    assert "edge_gat.proj.weight" in result["skipped"]
    # Model's EdgeGAT input dim remains 12
    assert model.edge_gat.proj.in_features == 12


# ── Phase 5 Tests: Explanations & Alerts ─────────────────────────────────
def test_explainability_contains_semantic_and_confidence():
    """ExplainabilityEngine output must include detected semantic types and confidences."""
    cfg = ARGUSConfig(dim=64, graph_dim=32, temporal_layers=1, heads=4, use_semantic_detector=True, detector_name="heuristic")
    model = ARGUS(config=cfg)
    engine = ExplainabilityEngine(model)
    clip = torch.rand(1, 4, 3, 64, 64)

    res = engine.compute_node_attribution(clip)
    assert "semantic_types" in res
    assert "detection_scores" in res
    assert "detector_implementation" in res
    engine.cleanup()


def test_alert_dispatcher_threat_fields():
    """AlertDispatcher must compute num_person_interactions, weapon_count, person_weapon_edges honestly."""
    dispatcher = AlertDispatcher(threshold=0.5)

    # Simulate 2 keyframes:
    # Frame 0: 2 persons interacting
    # Frame 1: Person and Weapon interacting
    boxes = [
        [[0.1, 0.1, 0.3, 0.5], [0.15, 0.15, 0.35, 0.55]],
        [[0.2, 0.2, 0.4, 0.6], [0.22, 0.22, 0.30, 0.35]],
    ]
    sem_types = [
        [SEMANTIC_TYPE_PERSON, SEMANTIC_TYPE_PERSON],
        [SEMANTIC_TYPE_PERSON, SEMANTIC_TYPE_WEAPON],
    ]
    det_scores = [
        [0.90, 0.85],
        [0.92, 0.88],
    ]
    edge_indices = [
        (torch.tensor([0, 1]), torch.tensor([1, 0])),  # P2P
        (torch.tensor([0, 1]), torch.tensor([1, 0])),  # P2W
    ]

    event = dispatcher.dispatch(
        video_source="cam_entrance.mp4",
        timeline_scores=[0.3, 0.82],
        keyframe_indices=[0, 8],
        boxes_per_frame=boxes,
        semantic_types_per_frame=sem_types,
        detection_scores_per_frame=det_scores,
        edge_indices_per_frame=edge_indices,
        detector_impl="yolo",
    )

    assert event is not None
    assert event.is_anomaly is True
    assert event.person_count == 2
    assert event.weapon_count == 1
    assert event.object_counts == {
        "person": 2,
        "vehicle": 0,
        "bag": 0,
        "weapon": 1,
        "other_object": 0,
    }
    assert event.person_weapon_edges == 1
    assert event.num_person_interactions == 1
    assert "Detected knife" in event.summary or "knife" in event.summary
    assert "0.88" in event.summary


def test_alert_dispatcher_heuristic_never_claims_weapon():
    """When detector_impl is heuristic/grid, AlertDispatcher must never claim weapon detections or interactions."""
    dispatcher = AlertDispatcher(threshold=0.5)

    event = dispatcher.dispatch(
        video_source="cam_test.mp4",
        timeline_scores=[0.8],
        keyframe_indices=[0],
        boxes_per_frame=[[[0.1, 0.1, 0.3, 0.4]]],
        semantic_types_per_frame=[[SEMANTIC_TYPE_WEAPON]],  # even if someone passes type 3
        detector_impl="heuristic",
    )
    assert event is not None
    assert event.weapon_count == 0
    assert event.person_weapon_edges == 0
    assert "knife" not in event.summary.lower()


# ── Full End-to-End Model Forward Tests ─────────────────────────────────
def test_full_model_forward_empty_detections():
    """A full model forward pass works cleanly when all detector proposals are empty."""
    cfg = ARGUSConfig(
        image_size=64,
        dim=64,
        graph_dim=32,
        temporal_layers=1,
        heads=4,
        use_semantic_detector=True,
        detector_name="heuristic",
    )
    model = ARGUS(config=cfg)

    # Mock detector to return empty
    class EmptyDetector(nn.Module):
        def forward(self, frame):
            return DetectionResult(
                boxes=torch.empty((0, 4)),
                scores=torch.empty((0,)),
                class_ids=torch.empty((0,), dtype=torch.long),
                semantic_types=torch.empty((0,), dtype=torch.long),
                detector_impl="yolo"
            )

    model.graph_builder.detector = EmptyDetector()
    clip = torch.rand(1, 4, 3, 64, 64)
    out = model(clip)
    assert out.scores.shape[0] == 1
    assert torch.isfinite(out.scores).all()


def test_full_model_forward_mixed_entity_types():
    """A full model forward pass works with mixed entity types (Person, Vehicle, Bag, Weapon, Other)."""
    cfg = ARGUSConfig(
        image_size=64,
        dim=64,
        graph_dim=32,
        temporal_layers=2,
        heads=4,
        use_semantic_detector=True,
        detector_name="heuristic",
    )
    model = ARGUS(config=cfg)

    # Mock detector returning all 5 entity types
    class MixedDetector(nn.Module):
        def forward(self, frame):
            return DetectionResult(
                boxes=torch.tensor([
                    [0.10, 0.10, 0.30, 0.50],  # Person (0)
                    [0.50, 0.50, 0.80, 0.80],  # Vehicle (1)
                    [0.15, 0.15, 0.25, 0.30],  # Bag (2)
                    [0.12, 0.12, 0.20, 0.22],  # Weapon (3)
                    [0.70, 0.10, 0.90, 0.30],  # Other object (4)
                ]),
                scores=torch.tensor([0.95, 0.88, 0.77, 0.85, 0.60]),
                class_ids=torch.tensor([0, 2, 24, 43, 56]),
                semantic_types=torch.tensor([0, 1, 2, 3, 4]),
                detector_impl="yolo"
            )

    model.graph_builder.detector = MixedDetector()
    clip = torch.rand(1, 4, 3, 64, 64)
    out = model(clip)
    assert out.scores.shape[0] == 1
    assert torch.isfinite(out.scores).all()
    assert len(out.boxes[0]) == 4
    assert getattr(out, 'semantic_types', None) is not None


def test_raw_boxes_input_defaults_to_other_object_and_12_dims():
    """When raw boxes are passed without a DetectionResult container, semantic graph remains 12-dim."""
    builder = DynamicSceneGraphBuilder(max_objects=4, use_semantic=True, semantic_schema_version="threat5")
    frame = torch.rand(3, 64, 64)
    raw_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.4], [0.5, 0.5, 0.7, 0.8]])

    nodes, (src, dst), edge_attr, boxes = builder.build(frame, detection_result=raw_boxes)
    assert nodes.shape == (2, 12)
    assert edge_attr.shape[-1] == 10


def test_alert_dispatcher_symmetric_edge_orientation():
    """AlertDispatcher correctly counts undirected edges regardless of order (e.g. (1, 0) or (0, 1))."""
    dispatcher = AlertDispatcher(threshold=0.5)
    event = dispatcher.dispatch(
        video_source="test.mp4",
        timeline_scores=[0.9],
        keyframe_indices=[0],
        boxes_per_frame=[[[0.1, 0.1, 0.3, 0.4], [0.15, 0.15, 0.35, 0.45]]],
        semantic_types_per_frame=[[SEMANTIC_TYPE_PERSON, SEMANTIC_TYPE_WEAPON]],
        detection_scores_per_frame=[[0.9, 0.85]],
        edge_indices_per_frame=[(torch.tensor([1]), torch.tensor([0]))],  # (1, 0) reversed order
        detector_impl="yolo",
    )
    assert event.person_weapon_edges == 1


def test_package_exports():
    """Top-level argus package exports new semantic and checkpoint classes/constants."""
    import argus
    assert hasattr(argus, "IncompatibleCheckpointError")
    assert hasattr(argus, "load_partial_checkpoint")
    assert hasattr(argus, "NUM_SEMANTIC_CLASSES")
    assert hasattr(argus, "SEMANTIC_NODE_DIM")
    assert hasattr(argus, "THREAT_RELATIONAL_EDGE_DIM")
    assert hasattr(argus, "SEMANTIC_TYPE_WEAPON")

