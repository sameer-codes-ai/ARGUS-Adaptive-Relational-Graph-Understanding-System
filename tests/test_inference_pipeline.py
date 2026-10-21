import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import web_demo

def test_inference_pipeline():
    web_demo.load_model(Path('outputs/checkpoints_joint/best.pt'))

    # Test A: Street Violence
    res1 = web_demo.process_video_inference(Path('demo_samples/sample_violence_V691.mp4'), threshold=0.35, frames=24)
    assert res1['status'] == 'success'
    assert res1['video_url'] == '/demo_samples/sample_violence_V691.mp4'
    assert res1['duration_sec'] > 0
    assert res1['native_fps'] > 0
    assert res1['peak_score'] > 0.50
    assert res1['severity'] in ('CRITICAL', 'HIGH')
    assert len(res1['frames']) > 0
    assert len(res1['timeline']) > 0

    # Test B: Normal Pedestrian Walkway
    res2 = web_demo.process_video_inference(Path('demo_samples/sample_normal_NV344.mp4'), threshold=0.35, frames=24)
    assert res2['status'] == 'success'
    assert res2['video_url'] == '/demo_samples/sample_normal_NV344.mp4'
    assert res2['peak_score'] < 0.20
    assert res2['severity'] == 'NORMAL'
    assert res2['is_anomaly'] == False

    # Test C: Robbery
    res3 = web_demo.process_video_inference(Path('demo_samples/sample_robbery_087.mp4'), threshold=0.35, frames=24)
    assert res3['status'] == 'success'
    assert res3['peak_score'] > 0.50
    assert res3['is_anomaly'] == True

    print("\n[ALL 3 TESTS PASSED SUCCESSFULLY]")

if __name__ == '__main__':
    test_inference_pipeline()
