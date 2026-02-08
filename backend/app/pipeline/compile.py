"""
Output Compiler

Assembles the three final JSON deliverables from inference results:
  1. world_gt.json     — complete ground truth for humanoid navigation
  2. timeline.json     — human-readable event sequence
  3. confidence_report.json — segments flagged for human review
"""

from app.models import (
    WorldGT,
    VideoInfo,
    TrackedObject,
    Agent,
    Event,
    ObjectStateTimeline,
    AffordanceEntry,
    TimelineEntry,
    ConfidenceReport,
    ConfidenceSegment,
)
from app.config import Settings


def compile_world_gt(
    inference_result: dict,
    video_info: dict,
) -> WorldGT:
    """Assemble the complete world_gt.json from inference output.

    Args:
        inference_result: Output from run_inference() containing
            objects, agents, events, state_timeline.
        video_info: Video metadata dict (source, duration, fps, resolution).

    Returns:
        WorldGT model ready for serialization.
    """
    objects: list[TrackedObject] = inference_result.get("objects", [])
    agents: list[Agent] = inference_result.get("agents", [])
    events: list[Event] = inference_result.get("events", [])
    state_timeline: list[ObjectStateTimeline] = inference_result.get("state_timeline", [])

    affordance_map = _build_affordance_map(objects, state_timeline)

    vid = VideoInfo(
        source=video_info.get("source", "unknown"),
        duration_seconds=video_info.get("duration_seconds", 0.0),
        fps=video_info.get("fps", 30.0),
        resolution=video_info.get("resolution", [1920, 1080]),
    )

    return WorldGT(
        version="0.1",
        video=vid,
        objects=objects,
        agents=agents,
        state_timeline=state_timeline,
        events=events,
        affordance_map=affordance_map,
    )


def _build_affordance_map(
    objects: list[TrackedObject],
    state_timeline: list[ObjectStateTimeline],
) -> dict[str, AffordanceEntry]:
    """Build per-object affordance map from tracked objects and state history.

    Args:
        objects: Tracked objects with affordances from ontology.
        state_timeline: State sequences per object.

    Returns:
        Dict mapping object_id to AffordanceEntry.
    """
    timeline_by_id = {tl.object_id: tl for tl in state_timeline}
    affordance_map = {}

    for obj in objects:
        observed_states = []
        tl = timeline_by_id.get(obj.id)
        if tl:
            observed_states = list({seg.state for seg in tl.states})

        affordance_map[obj.id] = AffordanceEntry(
            interactable="openable" in obj.affordances or "closable" in obj.affordances,
            movable="movable" in obj.affordances,
            traversable="traversable" in obj.affordances,
            states=observed_states if observed_states else [obj.initial_state],
        )

    return affordance_map


def compile_timeline(
    events: list[Event],
    objects: list[TrackedObject],
) -> list[TimelineEntry]:
    """Generate a human-readable event timeline.

    Args:
        events: Detected events from inference.
        objects: Tracked objects for label lookup.

    Returns:
        List of TimelineEntry models ordered by time.
    """
    objects_by_id = {obj.id: obj for obj in objects}
    timeline = []

    for event in sorted(events, key=lambda e: e.start_time):
        obj = objects_by_id.get(event.object_id)
        object_label = obj.label if obj else "unknown"

        description = _generate_event_description(event, object_label)

        timeline.append(
            TimelineEntry(
                time=event.start_time,
                end_time=event.end_time,
                event_type=event.type,
                object_label=object_label,
                description=description,
                confidence=event.confidence,
            )
        )

    return timeline


def _generate_event_description(event: Event, object_label: str) -> str:
    """Generate a human-readable description for an event.

    Args:
        event: The event to describe.
        object_label: Label of the object involved.

    Returns:
        Description string.
    """
    action = event.type.replace("_", " ")
    time_range = f"{event.start_time:.1f}s - {event.end_time:.1f}s"
    confidence_pct = f"{event.confidence * 100:.0f}%"

    return f"Agent {event.agent_id} performs {action} on {object_label} ({time_range}, confidence: {confidence_pct})"


def compile_confidence_report(
    events: list[Event],
    objects: list[TrackedObject],
    settings: Settings | None = None,
) -> ConfidenceReport:
    """Generate a confidence report flagging segments for human review.

    Args:
        events: Detected events from inference.
        objects: Tracked objects for label lookup.
        settings: Config for low confidence threshold.

    Returns:
        ConfidenceReport model.
    """
    if settings is None:
        settings = Settings()

    objects_by_id = {obj.id: obj for obj in objects}
    review_segments = []
    high_count = 0
    low_count = 0
    total_confidence = 0.0

    for event in events:
        total_confidence += event.confidence

        if event.confidence < settings.low_confidence_threshold:
            low_count += 1
            obj = objects_by_id.get(event.object_id)
            object_label = obj.label if obj else "unknown"
            reason = _determine_low_confidence_reason(event, settings)

            review_segments.append(
                ConfidenceSegment(
                    event_id=event.id,
                    event_type=event.type,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    confidence=event.confidence,
                    reason=reason,
                )
            )
        else:
            high_count += 1

    overall = total_confidence / len(events) if events else 0.0

    return ConfidenceReport(
        total_events=len(events),
        high_confidence_count=high_count,
        low_confidence_count=low_count,
        review_segments=review_segments,
        overall_confidence=round(overall, 4),
    )


def _determine_low_confidence_reason(event: Event, settings: Settings) -> str:
    """Determine why an event has low confidence.

    Args:
        event: The low-confidence event.
        settings: Config for threshold comparisons.

    Returns:
        Human-readable reason string.
    """
    reasons = []

    if event.signals.motion_score < settings.motion_threshold:
        reasons.append(
            f"low motion ({event.signals.motion_score:.2f} < {settings.motion_threshold})"
        )

    proximity_score_threshold = 1.0 / (
        1.0 + settings.proximity_threshold / settings.proximity_norm_factor
    )
    if event.signals.proximity_score < proximity_score_threshold:
        reasons.append(
            f"low proximity ({event.signals.proximity_score:.2f} < {proximity_score_threshold:.2f})"
        )

    if event.signals.vl_confidence < settings.low_confidence_threshold:
        reasons.append(
            f"low VL confidence ({event.signals.vl_confidence:.2f} < {settings.low_confidence_threshold})"
        )

    if not reasons:
        reasons.append("combined signal score below threshold")

    return "; ".join(reasons)
