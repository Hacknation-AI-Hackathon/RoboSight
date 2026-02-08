"""
Evaluation Metrics

Computes metrics for a pipeline run, enabling baseline vs calibrated
comparison. Metrics map directly to the challenge evaluation criteria:
  - Event precision/recall → Ground Truth Accuracy
  - Average boundary error → Temporal Precision
  - Review percentage → Human-in-the-Loop Efficiency
  - Labeling time saved → Scalability / Cost Reduction
"""

from app.models import (
    Event,
    Correction,
    CorrectionAction,
    MetricsResult,
    ConfidenceReport,
)
from app.config import Settings


def compute_metrics(
    events: list[Event],
    corrections: list[Correction] | None = None,
    confidence_report: ConfidenceReport | None = None,
    video_duration: float = 0.0,
) -> MetricsResult:
    """Compute all evaluation metrics for a pipeline run.

    Args:
        events: Predicted events from the inference engine.
        corrections: Human corrections (None for baseline run).
        confidence_report: Confidence report for review stats.
        video_duration: Video duration for time-saved estimate.

    Returns:
        MetricsResult with all computed metrics.
    """
    if not events:
        return MetricsResult(total_events_detected=0)

    corrections = corrections or []

    precision = _compute_precision(events, corrections)
    recall = _compute_recall(events, corrections)
    boundary_error = _compute_avg_boundary_error(events, corrections)
    review_pct = _compute_review_percentage(confidence_report)
    time_saved = _compute_time_saved(events, corrections, video_duration)

    ground_truth_count = len(events)
    for corr in corrections:
        if corr.action == CorrectionAction.ADD_EVENT:
            ground_truth_count += 1
        elif corr.action == CorrectionAction.REJECT:
            ground_truth_count -= 1
    ground_truth_count = max(ground_truth_count, 0)

    return MetricsResult(
        event_precision=precision,
        event_recall=recall,
        avg_boundary_error=boundary_error,
        review_percentage=review_pct,
        labeling_time_saved_estimate=time_saved,
        total_events_detected=len(events),
        total_events_ground_truth=ground_truth_count,
    )


def _compute_precision(
    events: list[Event],
    corrections: list[Correction],
) -> float:
    """Compute event precision: fraction of predicted events that are correct.

    An event is considered incorrect if it was rejected by a human.
    Adjusted boundaries still count as correct (the event was real,
    just imprecisely bounded).

    Returns:
        Precision between 0.0 and 1.0.
    """
    if not events:
        return 0.0

    rejected_ids = {
        corr.event_id
        for corr in corrections
        if corr.action == CorrectionAction.REJECT and corr.event_id
    }

    correct = sum(1 for e in events if e.id not in rejected_ids)
    return round(correct / len(events), 4)


def _compute_recall(
    events: list[Event],
    corrections: list[Correction],
) -> float:
    """Compute event recall: fraction of true events that were detected.

    True events = detected events - rejected + added.
    Detected true positives = detected events - rejected.

    Returns:
        Recall between 0.0 and 1.0.
    """
    rejected_count = sum(
        1 for corr in corrections if corr.action == CorrectionAction.REJECT
    )
    added_count = sum(
        1 for corr in corrections if corr.action == CorrectionAction.ADD_EVENT
    )

    true_positives = len(events) - rejected_count
    total_true_events = true_positives + added_count

    if total_true_events <= 0:
        return 1.0 if not events else 0.0

    return round(max(0.0, true_positives / total_true_events), 4)


def _compute_avg_boundary_error(
    events: list[Event],
    corrections: list[Correction],
) -> float:
    """Compute average boundary error in seconds.

    Only uses adjust_boundary corrections where the human provided
    corrected start/end times.

    Returns:
        Average absolute boundary error in seconds. 0.0 if no corrections.
    """
    events_by_id = {e.id: e for e in events}
    errors = []

    for corr in corrections:
        if corr.action != CorrectionAction.ADJUST_BOUNDARY or not corr.event_id:
            continue

        event = events_by_id.get(corr.event_id)
        if not event:
            continue

        if corr.corrected_start is not None:
            errors.append(abs(corr.corrected_start - event.start_time))
        if corr.corrected_end is not None:
            errors.append(abs(corr.corrected_end - event.end_time))

    if not errors:
        return 0.0

    return round(sum(errors) / len(errors), 4)


def _compute_review_percentage(
    confidence_report: ConfidenceReport | None,
) -> float:
    """Compute percentage of events that need human review.

    Returns:
        Percentage between 0.0 and 100.0.
    """
    if not confidence_report or confidence_report.total_events == 0:
        return 0.0

    return round(
        (confidence_report.low_confidence_count / confidence_report.total_events) * 100.0,
        2,
    )


def _compute_time_saved(
    events: list[Event],
    corrections: list[Correction],
    video_duration: float,
) -> float:
    """Estimate labeling time saved compared to fully manual annotation.

    Assumptions:
      - Manual labeling: ~2 minutes per event from scratch
      - AI-assisted review: ~15 seconds per correction
      - Approved events (no correction needed): ~5 seconds to verify

    Returns:
        Estimated percentage of time saved (0.0 to 100.0).
    """
    if not events:
        return 0.0

    manual_time_per_event = 120.0
    correction_time = 15.0
    verify_time = 5.0

    total_manual = len(events) * manual_time_per_event
    total_assisted = (
        len(corrections) * correction_time
        + (len(events) - len(corrections)) * verify_time
    )

    if total_manual <= 0:
        return 0.0

    saved = max(0.0, (total_manual - total_assisted) / total_manual) * 100.0
    return round(saved, 2)


def compare_runs(
    baseline_metrics: MetricsResult,
    calibrated_metrics: MetricsResult,
) -> dict:
    """Compare baseline and calibrated run metrics.

    Args:
        baseline_metrics: Metrics from the zero-shot baseline run.
        calibrated_metrics: Metrics from the calibrated rerun.

    Returns:
        Dict with improvement deltas and summary.
    """
    precision_delta = calibrated_metrics.event_precision - baseline_metrics.event_precision
    recall_delta = calibrated_metrics.event_recall - baseline_metrics.event_recall
    boundary_delta = baseline_metrics.avg_boundary_error - calibrated_metrics.avg_boundary_error
    review_delta = baseline_metrics.review_percentage - calibrated_metrics.review_percentage

    return {
        "precision_improvement": round(precision_delta, 4),
        "recall_improvement": round(recall_delta, 4),
        "boundary_error_reduction": round(boundary_delta, 4),
        "review_percentage_reduction": round(review_delta, 2),
        "baseline": baseline_metrics.model_dump(),
        "calibrated": calibrated_metrics.model_dump(),
        "summary": _generate_comparison_summary(
            precision_delta, recall_delta, boundary_delta, review_delta
        ),
    }


def _generate_comparison_summary(
    precision_delta: float,
    recall_delta: float,
    boundary_delta: float,
    review_delta: float,
) -> str:
    """Generate a human-readable summary of improvements."""
    improvements = []

    if precision_delta > 0:
        improvements.append(f"precision improved by {precision_delta * 100:.1f}%")
    if recall_delta > 0:
        improvements.append(f"recall improved by {recall_delta * 100:.1f}%")
    if boundary_delta > 0:
        improvements.append(f"boundary error reduced by {boundary_delta:.2f}s")
    if review_delta > 0:
        improvements.append(f"review segments reduced by {review_delta:.1f}%")

    if improvements:
        return "Calibration improved results: " + ", ".join(improvements)
    elif precision_delta == 0 and recall_delta == 0:
        return "No significant change after calibration"
    else:
        return "Calibration had mixed results; manual review recommended"
