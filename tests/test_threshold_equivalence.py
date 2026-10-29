"""One detector run at the confidence floor gives every threshold: re-running YOLOv8x at tau yields
exactly the cached floor-run detections filtered at tau (greedy NMS only lets higher-scoring boxes
suppress lower ones, so the threshold commutes with it)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and YOLOv8x weights")


@pytest.mark.parametrize("tau", [0.05, 0.5])
def test_rerun_at_threshold_equals_cached_floor_run_filtered(tau):
    from wad import thresh_yolo as ty
    from wad.config import REPO
    from wad.detect import load_model
    from wad.detection_rescore import iou_rows
    from wad.imageio import load_bgr

    det_path = REPO / "cache" / "bdd" / "det" / "v8x_1280" / "detections.parquet"
    if not det_path.exists():
        pytest.skip("BDD100K detection cache not built")
    v = ty.detector_cfg()
    model = load_model(v)
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    im = ty.subset("bdd", 5)
    det = pd.read_parquet(det_path)
    for r in im.itertuples():
        res = model.predict(source=load_bgr(r.path, "apply"), imgsz=1280, conf=tau, iou=0.5, max_det=300,
                            classes=list(names), device="cuda:0", verbose=False)[0]
        a_box, a_score = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
        a_cls = np.array([names[int(c)] for c in res.boxes.cls.cpu().numpy()])
        b = det[(det.image_id == r.image_id) & (det.score >= tau)]
        assert len(a_score) == len(b)
        if len(b) == 0:
            continue
        iou = iou_rows(np.vstack([a_box, b[["x1", "y1", "x2", "y2"]].to_numpy()]))[:len(b), len(b):]
        j = iou.argmax(1)
        assert (iou.max(1) > 0.999).all()
        assert np.allclose(a_score, b.score.to_numpy()[j], atol=1e-5) and (a_cls == b.cls.to_numpy()[j]).all()
