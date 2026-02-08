"""
End-to-End Test with Mock Data

Validates the full Dev B pipeline without GPU dependencies:
  1. Load mock model outputs
  2. Run temporal inference
  3. Compile outputs
  4. Simulate corrections
  5. Calibrate and rerun
  6. Compare metrics
"""

import json
import sys
from pathlib import Path

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.config import Settings
from app.models import (
    Event,
    Correction,
    CorrectionAction,
    CorrectionPayload,
)
from app.pipeline import infer, calibrate, compile, eval


MOCK_DIR = Path(__file__).parent / "mock_data"

MOCK_VIDEO_INFO = {
    "source": "demo_drawer.mp4",
    "duration_seconds": 10.0,
    "fps": 30.0,
    "resolution": [1920, 1080],
}


def load_mock_data():
    """Load mock detections, segmentations, and semantics."""
    with open(MOCK_DIR / "mock_detections.json") as f:
        detections = json.load(f)
    with open(MOCK_DIR / "mock_segmentations.json") as f:
        segmentations = json.load(f)
    with open(MOCK_DIR / "mock_semantics.json") as f:
        semantics = json.load(f)
    return detections, segmentations, semantics


def test_inference(detections, segmentations, semantics, settings):
    """Test the temporal inference engine."""
    print("=" * 60)
    print("STEP 1: TEMPORAL INFERENCE")
    print("=" * 60)

    result = infer.run_inference(
        detections=detections,
        segmentations=segmentations,
        semantics=semantics,
        video_duration=MOCK_VIDEO_INFO["duration_seconds"],
        settings=settings,
    )

    objects = result["objects"]
    agents = result["agents"]
    events = result["events"]
    state_timeline = result["state_timeline"]
    signals_log = result["signals_log"]

    print(f"\nDiscovered objects: {len(objects)}")
    for obj in objects:
        print(f"  - {obj.id}: {obj.label} (category={obj.category}, "
              f"initial_state={obj.initial_state}, "
              f"affordances={obj.affordances})")

    print(f"\nTracked agents: {len(agents)}")
    for agent in agents:
        print(f"  - {agent.id}: {len(agent.track)} track points")

    print(f"\nDetected events: {len(events)}")
    for event in events:
        print(f"  - {event.id}: {event.type} "
              f"[{event.start_time:.1f}s - {event.end_time:.1f}s] "
              f"conf={event.confidence:.3f} "
              f"(motion={event.signals.motion_score:.3f}, "
              f"proximity={event.signals.proximity_score:.3f}, "
              f"vl={event.signals.vl_confidence:.3f})")

    print(f"\nState timelines: {len(state_timeline)}")
    for tl in state_timeline:
        print(f"  - {tl.object_id}:")
        for seg in tl.states:
            print(f"      {seg.state}: {seg.start:.1f}s - {seg.end:.1f}s")

    print(f"\nSignals log:")
    for label, signals in signals_log.items():
        print(f"  - {label}: {len(signals)} signal entries")
        for sig in signals:
            print(f"      t={sig.timestamp:.1f}s motion={sig.motion_score:.3f} "
                  f"proximity={sig.proximity_score:.3f} "
                  f"dwell={sig.dwell_time:.3f} "
                  f"vl_state={sig.vl_state} vl_conf={sig.vl_confidence:.3f}")

    assert len(objects) > 0, "No objects discovered"
    assert len(agents) > 0, "No agents tracked"
    print("\n✓ Inference passed")
    return result


def test_compile(inference_result, settings):
    """Test output compilation."""
    print("\n" + "=" * 60)
    print("STEP 2: OUTPUT COMPILATION")
    print("=" * 60)

    world_gt = compile.compile_world_gt(inference_result, MOCK_VIDEO_INFO)
    print(f"\nworld_gt.json:")
    print(f"  version: {world_gt.version}")
    print(f"  objects: {len(world_gt.objects)}")
    print(f"  agents: {len(world_gt.agents)}")
    print(f"  events: {len(world_gt.events)}")
    print(f"  state_timeline: {len(world_gt.state_timeline)}")
    print(f"  affordance_map: {len(world_gt.affordance_map)} entries")
    for obj_id, aff in world_gt.affordance_map.items():
        print(f"    {obj_id}: interactable={aff.interactable}, "
              f"states={aff.states}")

    timeline = compile.compile_timeline(
        inference_result["events"], inference_result["objects"]
    )
    print(f"\ntimeline.json: {len(timeline)} entries")
    for entry in timeline:
        print(f"  - [{entry.time:.1f}s-{entry.end_time:.1f}s] "
              f"{entry.event_type}: {entry.description}")

    confidence_report = compile.compile_confidence_report(
        inference_result["events"], inference_result["objects"], settings
    )
    print(f"\nconfidence_report.json:")
    print(f"  total_events: {confidence_report.total_events}")
    print(f"  high_confidence: {confidence_report.high_confidence_count}")
    print(f"  low_confidence: {confidence_report.low_confidence_count}")
    print(f"  overall_confidence: {confidence_report.overall_confidence:.3f}")
    for seg in confidence_report.review_segments:
        print(f"  review: {seg.event_id} ({seg.event_type}) - {seg.reason}")

    world_gt_dict = world_gt.model_dump()
    assert "version" in world_gt_dict, "Missing version"
    assert "objects" in world_gt_dict, "Missing objects"
    assert "events" in world_gt_dict, "Missing events"
    assert "state_timeline" in world_gt_dict, "Missing state_timeline"
    assert "affordance_map" in world_gt_dict, "Missing affordance_map"
    print("\n✓ Compilation passed")
    return world_gt, timeline, confidence_report


def test_calibration(events, settings):
    """Test calibration from simulated corrections."""
    print("\n" + "=" * 60)
    print("STEP 3: CALIBRATION")
    print("=" * 60)

    corrections = []

    if len(events) >= 1:
        corrections.append(
            Correction(
                event_id=events[0].id,
                action=CorrectionAction.ADJUST_BOUNDARY,
                corrected_start=events[0].start_time + 0.2,
                corrected_end=events[0].end_time - 0.1,
            )
        )

    if len(events) >= 2:
        corrections.append(
            Correction(
                event_id=events[1].id,
                action=CorrectionAction.REJECT,
            )
        )

    corrections.append(
        Correction(
            action=CorrectionAction.ADD_EVENT,
            type="reach_into_drawer",
            start_time=5.0,
            end_time=5.8,
        )
    )

    if not corrections:
        corrections.append(
            Correction(
                action=CorrectionAction.ADD_EVENT,
                type="open_drawer",
                start_time=3.0,
                end_time=3.8,
            )
        )

    print(f"\nSimulated corrections: {len(corrections)}")
    for corr in corrections:
        print(f"  - {corr.action.value}: event_id={corr.event_id}")

    print(f"\nBaseline thresholds:")
    baseline_params = settings.get_calibratable_params()
    for k, v in baseline_params.items():
        print(f"  {k}: {v}")

    calibrated_params = calibrate.calibrate_from_corrections(
        settings=settings,
        events=events,
        corrections=corrections,
    )

    print(f"\nCalibrated thresholds:")
    for k, v in calibrated_params.items():
        delta = v - baseline_params[k]
        marker = " *" if abs(delta) > 0.0001 else ""
        print(f"  {k}: {v}{marker}")

    changed = sum(
        1 for k in calibrated_params
        if abs(calibrated_params[k] - baseline_params[k]) > 0.0001
    )
    print(f"\nParameters changed: {changed}/{len(calibrated_params)}")
    print("\n✓ Calibration passed")
    return corrections, calibrated_params


def test_calibrated_rerun(
    detections, segmentations, semantics, calibrated_params, corrections, settings
):
    """Test rerun with calibrated settings."""
    print("\n" + "=" * 60)
    print("STEP 4: CALIBRATED RERUN")
    print("=" * 60)

    calibrated_settings = settings.apply_calibration(calibrated_params)

    result = infer.run_inference(
        detections=detections,
        segmentations=segmentations,
        semantics=semantics,
        video_duration=MOCK_VIDEO_INFO["duration_seconds"],
        settings=calibrated_settings,
    )

    print(f"\nCalibrated run events: {len(result['events'])}")
    for event in result["events"]:
        print(f"  - {event.id}: {event.type} "
              f"[{event.start_time:.1f}s - {event.end_time:.1f}s] "
              f"conf={event.confidence:.3f}")

    confidence_report = compile.compile_confidence_report(
        result["events"], result["objects"], calibrated_settings
    )

    print("\n✓ Calibrated rerun passed")
    return result, confidence_report


def test_metrics(baseline_events, calibrated_events, corrections, 
                 baseline_report, calibrated_report):
    """Test metrics computation and comparison."""
    print("\n" + "=" * 60)
    print("STEP 5: METRICS & COMPARISON")
    print("=" * 60)

    baseline_metrics = eval.compute_metrics(
        events=baseline_events,
        corrections=None,
        confidence_report=baseline_report,
        video_duration=MOCK_VIDEO_INFO["duration_seconds"],
    )

    print(f"\nBaseline metrics:")
    print(f"  precision:      {baseline_metrics.event_precision:.3f}")
    print(f"  recall:         {baseline_metrics.event_recall:.3f}")
    print(f"  boundary_error: {baseline_metrics.avg_boundary_error:.3f}s")
    print(f"  review_pct:     {baseline_metrics.review_percentage:.1f}%")
    print(f"  time_saved:     {baseline_metrics.labeling_time_saved_estimate:.1f}%")
    print(f"  events:         {baseline_metrics.total_events_detected}")

    calibrated_metrics = eval.compute_metrics(
        events=calibrated_events,
        corrections=corrections,
        confidence_report=calibrated_report,
        video_duration=MOCK_VIDEO_INFO["duration_seconds"],
    )

    print(f"\nCalibrated metrics:")
    print(f"  precision:      {calibrated_metrics.event_precision:.3f}")
    print(f"  recall:         {calibrated_metrics.event_recall:.3f}")
    print(f"  boundary_error: {calibrated_metrics.avg_boundary_error:.3f}s")
    print(f"  review_pct:     {calibrated_metrics.review_percentage:.1f}%")
    print(f"  time_saved:     {calibrated_metrics.labeling_time_saved_estimate:.1f}%")
    print(f"  events:         {calibrated_metrics.total_events_detected}")

    comparison = eval.compare_runs(baseline_metrics, calibrated_metrics)
    print(f"\nComparison:")
    print(f"  precision improvement:  {comparison['precision_improvement']:+.3f}")
    print(f"  recall improvement:     {comparison['recall_improvement']:+.3f}")
    print(f"  boundary error reduced: {comparison['boundary_error_reduction']:+.3f}s")
    print(f"  review % reduced:       {comparison['review_percentage_reduction']:+.1f}%")
    print(f"  summary: {comparison['summary']}")

    print("\n✓ Metrics comparison passed")


def main():
    print("RoboSight Dev B — End-to-End Mock Data Test")
    print("=" * 60)

    settings = Settings()
    detections, segmentations, semantics = load_mock_data()
    print(f"Loaded mock data: {len(detections)} detections, "
          f"{len(segmentations)} segmentations, {len(semantics)} semantics")

    inference_result = test_inference(
        detections, segmentations, semantics, settings
    )

    world_gt, timeline, baseline_report = test_compile(
        inference_result, settings
    )

    corrections, calibrated_params = test_calibration(
        inference_result["events"], settings
    )

    calibrated_result, calibrated_report = test_calibrated_rerun(
        detections, segmentations, semantics,
        calibrated_params, corrections, settings
    )

    test_metrics(
        baseline_events=inference_result["events"],
        calibrated_events=calibrated_result["events"],
        corrections=corrections,
        baseline_report=baseline_report,
        calibrated_report=calibrated_report,
    )

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
