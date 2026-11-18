import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import web_demo


def test_inference_pipeline():
    saved_globals = {
        name: getattr(web_demo, name)
        for name in (
            "GLOBAL_MODEL",
            "GLOBAL_CONFIG",
            "MODEL",
            "CONFIG",
            "DEVICE",
            "EXPLAINER",
            "ALERT_DISPATCHER",
            "MODEL_IS_TRAINED",
        )
        if hasattr(web_demo, name)
    }
    try:
        checkpoint = PROJECT_ROOT / "outputs" / "checkpoints_joint" / "best.pt"
        has_checkpoint = checkpoint.exists()
        web_demo.load_model(checkpoint, allow_random_model=not has_checkpoint)

        # Test A: Street Violence
        res1 = web_demo.process_video_inference(
            PROJECT_ROOT / "demo_samples" / "sample_violence_V691.mp4",
            threshold=0.35,
            frames=24,
        )
        assert res1["status"] == "success"
        assert res1["video_url"].endswith("/demo_samples/sample_violence_V691.mp4")
        assert res1["duration_sec"] > 0
        assert res1["native_fps"] > 0
        assert len(res1["frames"]) > 0
        assert len(res1["timeline"]) > 0
        if web_demo.MODEL_IS_TRAINED:
            assert res1["peak_score"] > 0.50
            assert res1["severity"] in ("CRITICAL", "HIGH")

        # Test B: Normal Pedestrian Walkway
        res2 = web_demo.process_video_inference(
            PROJECT_ROOT / "demo_samples" / "sample_normal_NV344.mp4",
            threshold=0.35,
            frames=24,
        )
        assert res2["status"] == "success"
        assert res2["video_url"].endswith("/demo_samples/sample_normal_NV344.mp4")
        assert res2["duration_sec"] > 0
        assert res2["native_fps"] > 0
        assert len(res2["frames"]) > 0
        assert len(res2["timeline"]) > 0
        if web_demo.MODEL_IS_TRAINED:
            assert res2["peak_score"] < 0.20
            assert res2["severity"] == "NORMAL"
            assert res2["is_anomaly"] == False

        # Test C: Robbery
        res3 = web_demo.process_video_inference(
            PROJECT_ROOT / "demo_samples" / "sample_robbery_087.mp4",
            threshold=0.35,
            frames=24,
        )
        assert res3["status"] == "success"
        assert res3["duration_sec"] > 0
        assert res3["native_fps"] > 0
        assert len(res3["frames"]) > 0
        assert len(res3["timeline"]) > 0
        if web_demo.MODEL_IS_TRAINED:
            assert res3["peak_score"] > 0.50
            assert res3["is_anomaly"] == True

        print("\n[ALL 3 TESTS PASSED SUCCESSFULLY]")
    finally:
        for name, value in saved_globals.items():
            setattr(web_demo, name, value)


if __name__ == "__main__":
    test_inference_pipeline()
