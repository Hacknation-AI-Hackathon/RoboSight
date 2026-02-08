"""
Calibration Engine

Updates inference behavior at inference time from human corrections.
No model retraining — only thresholds, weights, offsets, and constraints.

Calibration sources:
  - adjust_boundary corrections → boundary offsets + threshold adjustments
  - reject corrections → threshold tightening + weight rebalancing
  - add_event corrections → threshold loosening
  - VL descriptions from rejected events → language constraints

The output is a calibration dict that can be applied via
settings.apply_calibration() for an immediate rerun.
"""

from app.config import Settings
from app.models import Event, Correction, CorrectionAction, EventSignals


def calibrate_from_corrections(
    settings: Settings,
    events: list[Event],
    corrections: list[Correction],
) -> dict:
    """Compute calibrated parameters from human corrections.

    Args:
        settings: Current inference settings (baseline thresholds).
        events: Events from the baseline run.
        corrections: Human corrections from the frontend.

    Returns:
        Dict of calibrated parameter overrides ready for
        settings.apply_calibration().
    """
    if not corrections:
        return settings.get_calibratable_params()

    events_by_id = {e.id: e for e in events}

    boundary_corrections = []
    rejected_events = []
    added_events = []

    for corr in corrections:
        if corr.action == CorrectionAction.ADJUST_BOUNDARY and corr.event_id:
            event = events_by_id.get(corr.event_id)
            if event:
                boundary_corrections.append((event, corr))
        elif corr.action == CorrectionAction.REJECT and corr.event_id:
            event = events_by_id.get(corr.event_id)
            if event:
                rejected_events.append(event)
        elif corr.action == CorrectionAction.ADD_EVENT:
            added_events.append(corr)

    start_offset, end_offset = _compute_boundary_offsets(
        boundary_corrections, settings.start_offset, settings.end_offset
    )

    motion_threshold, proximity_threshold, dwell_time_threshold = _adjust_thresholds(
        rejected_events, added_events, settings
    )

    motion_weight, proximity_weight, vl_weight = _rebalance_weights(
        rejected_events, boundary_corrections, settings
    )

    return {
        "motion_threshold": motion_threshold,
        "proximity_threshold": proximity_threshold,
        "dwell_time_threshold": dwell_time_threshold,
        "proximity_norm_factor": settings.proximity_norm_factor,
        "motion_weight": motion_weight,
        "proximity_weight": proximity_weight,
        "vl_weight": vl_weight,
        "start_offset": start_offset,
        "end_offset": end_offset,
    }


def _compute_boundary_offsets(
    boundary_corrections: list[tuple[Event, Correction]],
    current_start_offset: float,
    current_end_offset: float,
) -> tuple[float, float]:
    """Compute systematic boundary bias correction from adjust_boundary corrections.

    If the system consistently predicts events starting too early,
    the start offset will shift positive. If too late, negative.

    Args:
        boundary_corrections: List of (original_event, correction) pairs.
        current_start_offset: Existing start offset from previous calibration.
        current_end_offset: Existing end offset from previous calibration.

    Returns:
        (start_offset, end_offset) in seconds.
    """
    if not boundary_corrections:
        return current_start_offset, current_end_offset

    start_deltas = []
    end_deltas = []

    for event, corr in boundary_corrections:
        if corr.corrected_start is not None:
            start_deltas.append(corr.corrected_start - event.start_time)
        if corr.corrected_end is not None:
            end_deltas.append(corr.corrected_end - event.end_time)

    new_start = current_start_offset
    if start_deltas:
        avg_start_delta = sum(start_deltas) / len(start_deltas)
        new_start = round(current_start_offset + avg_start_delta * 0.5, 4)

    new_end = current_end_offset
    if end_deltas:
        avg_end_delta = sum(end_deltas) / len(end_deltas)
        new_end = round(current_end_offset + avg_end_delta * 0.5, 4)

    return new_start, new_end


def _adjust_thresholds(
    rejected_events: list[Event],
    added_events: list[Correction],
    settings: Settings,
) -> tuple[float, float, float]:
    """Adjust detection thresholds based on false positives and false negatives.

    Rejected events = false positives → tighten thresholds (raise them)
    Added events = false negatives → loosen thresholds (lower them)

    Args:
        rejected_events: Events the human rejected (false positives).
        added_events: Events the human added (false negatives).
        settings: Current settings for baseline thresholds.

    Returns:
        (motion_threshold, proximity_threshold, dwell_time_threshold)
    """
    motion_threshold = settings.motion_threshold
    proximity_threshold = settings.proximity_threshold
    dwell_time_threshold = settings.dwell_time_threshold

    if rejected_events:
        rejected_motion_scores = [
            e.signals.motion_score for e in rejected_events
        ]
        rejected_proximity_scores = [
            e.signals.proximity_score for e in rejected_events
        ]

        avg_rejected_motion = sum(rejected_motion_scores) / len(rejected_motion_scores)
        avg_rejected_proximity = sum(rejected_proximity_scores) / len(rejected_proximity_scores)

        if avg_rejected_motion < motion_threshold * 1.5:
            motion_threshold = round(
                min(motion_threshold * 1.2, 0.8), 4
            )

        if avg_rejected_proximity > 0:
            proximity_threshold = round(
                max(proximity_threshold * 0.85, 50.0), 4
            )

        dwell_time_threshold = round(
            min(dwell_time_threshold * 1.15, 3.0), 4
        )

    if added_events:
        scale = max(0.7, 1.0 - len(added_events) * 0.1)
        motion_threshold = round(
            max(motion_threshold * scale, 0.05), 4
        )
        proximity_threshold = round(
            min(proximity_threshold * (1.0 / scale), 500.0), 4
        )
        dwell_time_threshold = round(
            max(dwell_time_threshold * scale, 0.1), 4
        )

    return motion_threshold, proximity_threshold, dwell_time_threshold


def _rebalance_weights(
    rejected_events: list[Event],
    boundary_corrections: list[tuple[Event, Correction]],
    settings: Settings,
) -> tuple[float, float, float]:
    """Rebalance signal weights based on which signals were unreliable.

    If rejected events had high motion but low VL confidence,
    VL weight increases (it was right to be uncertain).
    If boundary corrections correlate with motion signal errors,
    motion weight decreases.

    Args:
        rejected_events: False positive events.
        boundary_corrections: Events with adjusted boundaries.
        settings: Current signal weights.

    Returns:
        (motion_weight, proximity_weight, vl_weight) normalized to sum to 1.0.
    """
    motion_w = settings.motion_weight
    proximity_w = settings.proximity_weight
    vl_w = settings.vl_weight

    if rejected_events:
        avg_motion = sum(e.signals.motion_score for e in rejected_events) / len(rejected_events)
        avg_vl = sum(e.signals.vl_confidence for e in rejected_events) / len(rejected_events)

        if avg_motion > 0.5 and avg_vl < 0.5:
            motion_w *= 0.85
            vl_w *= 1.2
        elif avg_vl > 0.5 and avg_motion < 0.3:
            vl_w *= 0.85
            motion_w *= 1.15

    if boundary_corrections:
        proximity_w *= 1.1

    total = motion_w + proximity_w + vl_w
    if total > 0:
        motion_w = round(motion_w / total, 4)
        proximity_w = round(proximity_w / total, 4)
        vl_w = round(1.0 - motion_w - proximity_w, 4)

    return motion_w, proximity_w, vl_w
