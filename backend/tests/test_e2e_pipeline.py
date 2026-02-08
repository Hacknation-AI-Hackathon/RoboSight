"""
End-to-End Pipeline Integration Tests

Tests the full chain: model outputs → inference → compile → output schemas.
Uses mock data (no GPU needed) and real job results from test runs.
"""

import json
import sys
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.pipeline import infer, compile, eval
from app.pipeline.semantics import _normalize_vl_output, _parse_vl_json
from app.pipeline.infer import _normalize_label, extract_vl_signals
from app.models import (
    DetectionFrame,
    SegmentationFrame,
    SemanticFrame,
    SemanticObject,
    WorldGT,
)

MOCK_DIR = Path(__file__).parent / "mock_data"
JOBS_DIR = Path(__file__).parent.parent / "jobs"


# =========================================================================
# Test 1: VL output normalization
# =========================================================================

def test_vl_normalization_adds_confidence():
    """VL model returns {label, state, action} — normalization should add confidence."""
    raw = {
        "objects": [
            {"label": "dresser", "state": "open", "action": "being opened"},
            {"label": "person", "state": "standing", "action": "reaching"},
        ]
    }
    result = _normalize_vl_output(raw)

    for obj in result["objects"]:
        assert "confidence" in obj, f"Missing confidence in {obj}"
        assert obj["confidence"] == 0.8
        assert "action" not in obj, f"Per-object action should be stripped: {obj}"

    assert result["action"] == "person_opening_dresser"
    print("  PASS: VL normalization adds confidence and infers action")


def test_vl_normalization_preserves_existing_confidence():
    """If VL already provides confidence, don't overwrite it."""
    raw = {
        "objects": [
            {"label": "drawer", "state": "closed", "confidence": 0.95},
        ],
        "action": "no_action",
    }
    result = _normalize_vl_output(raw)
    assert result["objects"][0]["confidence"] == 0.95
    print("  PASS: VL normalization preserves existing confidence")


def test_vl_parse_from_markdown():
    """VL model often wraps JSON in markdown code blocks."""
    raw = '```json\n{"objects": [{"label": "drawer", "state": "open"}]}\n```'
    parsed = _normalize_vl_output(_parse_vl_json(raw))
    assert len(parsed["objects"]) == 1
    assert parsed["objects"][0]["confidence"] == 0.8
    print("  PASS: VL parse handles markdown-wrapped JSON")


# =========================================================================
# Test 2: Label normalization
# =========================================================================

def test_label_normalization():
    """VL labels like 'dresser' should map to SAM3 labels like 'drawer'."""
    assert _normalize_label("dresser") == "drawer"
    assert _normalize_label("cabinet") == "drawer"
    assert _normalize_label("knob") == "handle"
    assert _normalize_label("door_handle") == "handle"
    assert _normalize_label("drawer") == "drawer"  # identity
    assert _normalize_label("unknown_thing") == "unknown_thing"  # passthrough
    print("  PASS: Label normalization maps VL synonyms correctly")


def test_extract_vl_signals_with_alias():
    """extract_vl_signals should find 'dresser' when looking for 'drawer'."""
    sem_frame = {
        "objects": [
            {"label": "dresser", "state": "open", "confidence": 0.85},
        ]
    }
    state, conf = extract_vl_signals(sem_frame, "drawer")
    assert state == "open", f"Expected 'open', got '{state}'"
    assert conf == 0.85, f"Expected 0.85, got {conf}"
    print("  PASS: extract_vl_signals resolves aliases (dresser→drawer)")


# =========================================================================
# Test 3: Pydantic model validation
# =========================================================================

def test_semantic_object_default_confidence():
    """SemanticObject should accept missing confidence (defaults to 0.0)."""
    obj = SemanticObject(label="drawer", state="open")
    assert obj.confidence == 0.0
    print("  PASS: SemanticObject defaults confidence to 0.0")


def test_pydantic_models_validate_mock_data():
    """Mock data should pass Pydantic validation."""
    with open(MOCK_DIR / "mock_detections.json") as f:
        detections = json.load(f)
    with open(MOCK_DIR / "mock_segmentations.json") as f:
        segmentations = json.load(f)
    with open(MOCK_DIR / "mock_semantics.json") as f:
        semantics = json.load(f)

    for d in detections:
        DetectionFrame.model_validate(d)
    for s in segmentations:
        SegmentationFrame.model_validate(s)
    for s in semantics:
        SemanticFrame.model_validate(s)

    print(f"  PASS: All mock data validates ({len(detections)} det, "
          f"{len(segmentations)} seg, {len(semantics)} sem)")


# =========================================================================
# Test 4: Full inference pipeline with mock data
# =========================================================================

def test_inference_with_mock_data():
    """Run full inference → compile chain with mock data."""
    with open(MOCK_DIR / "mock_detections.json") as f:
        detections = json.load(f)
    with open(MOCK_DIR / "mock_segmentations.json") as f:
        segmentations = json.load(f)
    with open(MOCK_DIR / "mock_semantics.json") as f:
        semantics = json.load(f)

    settings = Settings(backend="modal", jobs_dir="/tmp/robosight_test_jobs")

    result = infer.run_inference(
        detections=detections,
        segmentations=segmentations,
        semantics=semantics,
        video_duration=10.0,
        settings=settings,
    )

    assert "objects" in result, "Missing 'objects' in inference result"
    assert "agents" in result, "Missing 'agents' in inference result"
    assert "events" in result, "Missing 'events' in inference result"
    assert "state_timeline" in result, "Missing 'state_timeline' in inference result"

    print(f"  PASS: Inference produced {len(result['objects'])} objects, "
          f"{len(result['events'])} events, "
          f"{len(result['agents'])} agents")

    # Compile outputs
    video_info = {
        "source": "test_mock.mp4",
        "duration_seconds": 10.0,
        "fps": 30.0,
        "resolution": [1920, 1080],
    }

    world_gt = compile.compile_world_gt(result, video_info)
    assert world_gt is not None, "world_gt compilation failed"

    timeline = compile.compile_timeline(result["events"], result["objects"])
    assert isinstance(timeline, list), "timeline should be a list"

    confidence_report = compile.compile_confidence_report(
        result["events"], result["objects"], settings
    )
    assert confidence_report is not None, "confidence_report compilation failed"

    # Metrics
    metrics = eval.compute_metrics(
        events=result["events"],
        corrections=None,
        confidence_report=confidence_report,
        video_duration=10.0,
    )
    assert metrics is not None, "metrics computation failed"

    print(f"  PASS: Compile produced world_gt, "
          f"{len(timeline)} timeline entries, confidence report, metrics")


# =========================================================================
# Test 5: Inference with real job data (VL label mismatch scenario)
# =========================================================================

def test_inference_with_real_job_data():
    """Test with actual Modal test results that have VL label mismatches."""
    job_dir = JOBS_DIR / "test_Drawer_Stable"
    if not job_dir.exists():
        print("  SKIP: test_Drawer_Stable job results not found")
        return

    with open(job_dir / "yolo_detections.json") as f:
        det_data = json.load(f)
    with open(job_dir / "sam3_segmentations.json") as f:
        seg_data = json.load(f)
    with open(job_dir / "vl_semantics.json") as f:
        raw_semantics = json.load(f)
    with open(job_dir / "video_info.json") as f:
        video_info = json.load(f)

    # test_models.py wraps outputs in dicts; unwrap to flat lists
    detections = det_data["detections"] if isinstance(det_data, dict) else det_data
    segmentations = seg_data["segmentations"] if isinstance(seg_data, dict) else seg_data
    if isinstance(raw_semantics, dict):
        raw_semantics = raw_semantics.get("semantics", raw_semantics)

    # Normalize VL output (simulating what semantics.py now does)
    semantics = []
    for frame in raw_semantics:
        normalized = _normalize_vl_output({"objects": frame["objects"]})
        semantics.append({
            **frame,
            "objects": normalized["objects"],
            "action": normalized.get("action", frame.get("action", "no_action")),
        })

    # Verify normalization worked
    for frame in semantics:
        for obj in frame["objects"]:
            assert "confidence" in obj, f"Missing confidence after normalization: {obj}"
            assert "action" not in obj, f"Per-object action not stripped: {obj}"

    settings = Settings(backend="modal", jobs_dir="/tmp/robosight_test_jobs")

    result = infer.run_inference(
        detections=detections,
        segmentations=segmentations,
        semantics=semantics,
        video_duration=video_info["duration_seconds"],
        settings=settings,
    )

    print(f"  Objects discovered: {[o.label for o in result['objects']]}")
    print(f"  Events detected: {len(result['events'])}")
    for evt in result["events"]:
        print(f"    - {evt.action} on {evt.object_label} "
              f"[{evt.start_time:.1f}s - {evt.end_time:.1f}s] "
              f"conf={evt.confidence:.2f}")

    # Compile
    world_gt = compile.compile_world_gt(result, video_info)
    timeline = compile.compile_timeline(result["events"], result["objects"])
    confidence_report = compile.compile_confidence_report(
        result["events"], result["objects"], settings
    )

    print(f"  PASS: Real data pipeline — {len(result['objects'])} objects, "
          f"{len(result['events'])} events, {len(timeline)} timeline entries")


# =========================================================================
# Test 6: Orchestrator validation with intentionally bad data
# =========================================================================

def test_orchestrator_validation():
    """Validate that orchestrator catches bad data gracefully."""
    from app.pipeline.orchestrator import _validate_outputs

    # Good data
    good = {
        "detections": [{"frame_index": 0, "timestamp": 0.0, "person_boxes": [], "confidences": []}],
        "segmentations": [{"frame_index": 0, "timestamp": 0.0, "objects": []}],
        "semantics": [{"frame_index": 0, "timestamp": 0.0, "objects": [], "action": "no_action"}],
    }
    _validate_outputs(good)  # Should not raise
    print("  PASS: Valid data passes orchestrator validation")

    # Bad data (missing required fields)
    bad = {
        "detections": [{"wrong_field": 123}],
        "segmentations": [],
        "semantics": [{"frame_index": 0}],  # missing timestamp
    }
    _validate_outputs(bad)  # Should warn but not raise
    print("  PASS: Invalid data logs warnings but doesn't crash")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("\n=== RoboSight E2E Pipeline Tests ===\n")

    tests = [
        ("VL normalization - add confidence", test_vl_normalization_adds_confidence),
        ("VL normalization - preserve confidence", test_vl_normalization_preserves_existing_confidence),
        ("VL parse from markdown", test_vl_parse_from_markdown),
        ("Label normalization", test_label_normalization),
        ("VL signal extraction with aliases", test_extract_vl_signals_with_alias),
        ("SemanticObject default confidence", test_semantic_object_default_confidence),
        ("Pydantic mock data validation", test_pydantic_models_validate_mock_data),
        ("Full inference with mock data", test_inference_with_mock_data),
        ("Inference with real job data", test_inference_with_real_job_data),
        ("Orchestrator validation", test_orchestrator_validation),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    sys.exit(1 if failed else 0)
