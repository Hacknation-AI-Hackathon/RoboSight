"""
Temporal Ground-Truth Inference Engine

The core intelligence of the system. Takes raw model outputs (detections,
segmentations, semantics) and produces events + state timelines through
multi-signal fusion and a constraint-based state machine.

Organized in three layers:
  1. Signal computation — pure math functions
  2. State machine — transition logic with constraints
  3. Main pipeline — orchestrates the full temporal analysis
"""

import math
from app.models import (
    DetectionFrame,
    SegmentationFrame,
    SegmentedObject,
    SemanticFrame,
    Signal,
    StateSegment,
    TrackedObject,
    Agent,
    AgentTrackPoint,
    Event,
    EventSignals,
    ObjectStateTimeline,
)
from app.config import Settings


# ==========================================================================
# 1. SIGNAL COMPUTATION — Pure functions, no side effects
# ==========================================================================


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    """Compute the center point of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (cx, cy) center coordinates.
    """
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def bbox_area(bbox: list[float]) -> float:
    """Compute the area of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Area in pixels squared. Returns 0 if degenerate.
    """
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return w * h


def compute_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        bbox_a: [x1, y1, x2, y2]
        bbox_b: [x1, y1, x2, y2]

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    union = area_a + area_b - intersection

    if union == 0.0:
        return 0.0

    return intersection / union


def compute_motion_score(
    bbox_prev: list[float], bbox_curr: list[float]
) -> float:
    """Compute object motion score based on bounding box displacement.

    Motion = 1 - IoU(prev, curr). High score means the object moved.

    Args:
        bbox_prev: Previous frame bbox [x1, y1, x2, y2].
        bbox_curr: Current frame bbox [x1, y1, x2, y2].

    Returns:
        Motion score between 0.0 (no movement) and 1.0 (completely displaced).
    """
    if not bbox_prev or not bbox_curr:
        return 0.0
    return 1.0 - compute_iou(bbox_prev, bbox_curr)


def compute_proximity_score(
    person_bbox: list[float],
    object_bbox: list[float],
    norm_factor: float = 500.0,
) -> float:
    """Compute proximity score between a person and an object.

    Score = 1 / (1 + distance / norm_factor). High score means close.

    Args:
        person_bbox: Person bbox [x1, y1, x2, y2].
        object_bbox: Object bbox [x1, y1, x2, y2].
        norm_factor: Normalization factor for distance scaling.

    Returns:
        Proximity score between 0.0 (far) and 1.0 (overlapping).
    """
    if not person_bbox or not object_bbox:
        return 0.0

    pcx, pcy = bbox_center(person_bbox)
    ocx, ocy = bbox_center(object_bbox)
    distance = math.sqrt((pcx - ocx) ** 2 + (pcy - ocy) ** 2)

    return 1.0 / (1.0 + distance / norm_factor)


def compute_dwell_time(
    timestamps: list[float],
    proximity_scores: list[float],
    proximity_threshold: float = 0.5,
) -> float:
    """Compute accumulated time a person dwells near an object.

    Sums time intervals where the proximity score exceeds the threshold.

    Args:
        timestamps: Ordered list of timestamps (seconds).
        proximity_scores: Proximity score at each timestamp.
        proximity_threshold: Minimum proximity score to count as dwelling.

    Returns:
        Total dwell time in seconds.
    """
    if len(timestamps) < 2 or len(timestamps) != len(proximity_scores):
        return 0.0

    dwell = 0.0
    for i in range(1, len(timestamps)):
        if proximity_scores[i] >= proximity_threshold:
            dwell += timestamps[i] - timestamps[i - 1]

    return dwell


def extract_vl_signals(
    semantic_frame: dict, object_label: str
) -> tuple[str, float]:
    """Extract VL state and confidence for a specific object from a semantic frame.

    Args:
        semantic_frame: Raw semantic frame dict from VL model.
        object_label: The object label to look for (e.g. "drawer").

    Returns:
        (state, confidence) tuple. Defaults to ("unknown", 0.0) if not found.
    """
    objects = semantic_frame.get("objects", [])
    for obj in objects:
        label = obj.get("label", "") if isinstance(obj, dict) else obj.label
        if label == object_label:
            state = obj.get("state", "unknown") if isinstance(obj, dict) else obj.state
            conf = obj.get("confidence", 0.0) if isinstance(obj, dict) else obj.confidence
            return state, conf

    return "unknown", 0.0


def get_nearest_person_bbox(
    detections: list[dict],
    timestamp: float,
    max_gap: float = 0.5,
) -> list[float] | None:
    """Find the nearest person bounding box to a given timestamp.

    Args:
        detections: List of detection frame dicts.
        timestamp: Target timestamp in seconds.
        max_gap: Maximum allowed time gap to use a detection.

    Returns:
        Person bbox [x1, y1, x2, y2] or None if no detection within max_gap.
    """
    best = None
    best_diff = float("inf")

    for det in detections:
        t = det.get("timestamp", 0.0) if isinstance(det, dict) else det.timestamp
        diff = abs(t - timestamp)
        if diff < best_diff:
            best_diff = diff
            boxes = det.get("person_boxes", []) if isinstance(det, dict) else det.person_boxes
            if boxes:
                best = boxes[0]
            best_diff = diff

    if best is not None and best_diff <= max_gap:
        return best
    return None


def get_object_bbox_at_keyframe(
    segmentations: list[dict],
    frame_index: int,
    object_label: str,
) -> list[float] | None:
    """Get bounding box for a specific object at a specific keyframe.

    Args:
        segmentations: List of segmentation frame dicts.
        frame_index: Target keyframe index.
        object_label: Object label to search for.

    Returns:
        Object bbox [x1, y1, x2, y2] or None if not found.
    """
    for seg in segmentations:
        idx = seg.get("frame_index", -1) if isinstance(seg, dict) else seg.frame_index
        if idx == frame_index:
            objects = seg.get("objects", []) if isinstance(seg, dict) else seg.objects
            for obj in objects:
                label = obj.get("label", "") if isinstance(obj, dict) else obj.label
                if label == object_label:
                    return obj.get("bbox", None) if isinstance(obj, dict) else obj.bbox
    return None


def compute_signals_for_keyframe_pair(
    detections: list[dict],
    seg_prev: dict,
    seg_curr: dict,
    sem_curr: dict,
    object_label: str,
    settings: Settings | None = None,
) -> Signal:
    """Compute all signals for a pair of consecutive keyframes.

    This is the central signal fusion function. It combines:
      - Object motion (bbox displacement)
      - Person-object proximity
      - Dwell time accumulation
      - VL semantic state and confidence

    Args:
        detections: All detection frames for person tracking.
        seg_prev: Previous keyframe segmentation dict.
        seg_curr: Current keyframe segmentation dict.
        sem_curr: Current keyframe semantic dict.
        object_label: Which object to compute signals for.
        settings: Config for thresholds and normalization.

    Returns:
        Signal model with all computed values.
    """
    if settings is None:
        settings = Settings()

    curr_frame_index = (
        seg_curr.get("frame_index", 0)
        if isinstance(seg_curr, dict)
        else seg_curr.frame_index
    )
    curr_timestamp = (
        seg_curr.get("timestamp", 0.0)
        if isinstance(seg_curr, dict)
        else seg_curr.timestamp
    )

    bbox_prev = get_object_bbox_at_keyframe(
        [seg_prev],
        seg_prev.get("frame_index", -1) if isinstance(seg_prev, dict) else seg_prev.frame_index,
        object_label,
    )
    bbox_curr = get_object_bbox_at_keyframe(
        [seg_curr], curr_frame_index, object_label
    )

    motion = compute_motion_score(bbox_prev, bbox_curr) if bbox_prev and bbox_curr else 0.0

    person_bbox = get_nearest_person_bbox(detections, curr_timestamp)
    proximity = (
        compute_proximity_score(person_bbox, bbox_curr, settings.proximity_norm_factor)
        if person_bbox and bbox_curr
        else 0.0
    )

    prev_timestamp = (
        seg_prev.get("timestamp", 0.0)
        if isinstance(seg_prev, dict)
        else seg_prev.timestamp
    )
    timestamps_between = [
        d.get("timestamp", 0.0) if isinstance(d, dict) else d.timestamp
        for d in detections
        if prev_timestamp
        <= (d.get("timestamp", 0.0) if isinstance(d, dict) else d.timestamp)
        <= curr_timestamp
    ]
    proximity_scores_between = []
    for t in timestamps_between:
        pb = get_nearest_person_bbox(detections, t, max_gap=0.3)
        if pb and bbox_curr:
            proximity_scores_between.append(
                compute_proximity_score(pb, bbox_curr, settings.proximity_norm_factor)
            )
        else:
            proximity_scores_between.append(0.0)

    dwell = compute_dwell_time(
        timestamps_between,
        proximity_scores_between,
        1.0 / (1.0 + settings.proximity_threshold / settings.proximity_norm_factor),
    )

    vl_state, vl_confidence = extract_vl_signals(sem_curr, object_label)

    return Signal(
        frame_index=curr_frame_index,
        timestamp=curr_timestamp,
        object_label=object_label,
        motion_score=round(motion, 4),
        proximity_score=round(proximity, 4),
        dwell_time=round(dwell, 4),
        vl_state=vl_state,
        vl_confidence=round(vl_confidence, 4),
    )


# ==========================================================================
# 2. STATE MACHINE — Transition logic with constraints
# ==========================================================================


VALID_TRANSITIONS = {
    "closed": ["open", "partially_open"],
    "open": ["closed", "partially_open"],
    "partially_open": ["open", "closed"],
    "unknown": ["closed", "open", "partially_open"],
}

ACTION_MAP = {
    ("closed", "open"): "open_{label}",
    ("closed", "partially_open"): "open_{label}",
    ("open", "closed"): "close_{label}",
    ("open", "partially_open"): "close_{label}",
    ("partially_open", "open"): "open_{label}",
    ("partially_open", "closed"): "close_{label}",
    ("unknown", "open"): "open_{label}",
    ("unknown", "closed"): "close_{label}",
}


class ObjectStateMachine:
    """Constraint-based state machine for a single tracked object.

    Enforces:
      - Cannot transition to the same state
      - Only valid transitions allowed (closed→open, open→closed, etc.)
      - States persist unless multi-signal evidence supports change
      - Motion alone without proximity does NOT trigger a transition
    """

    def __init__(
        self,
        object_id: str,
        object_label: str,
        initial_state: str = "closed",
        settings: Settings | None = None,
    ):
        self.object_id = object_id
        self.object_label = object_label
        self.current_state = initial_state
        self.settings = settings or Settings()
        self.state_history: list[StateSegment] = []
        self.events: list[Event] = []
        self.signals_log: list[Signal] = []
        self._state_start_time: float = 0.0
        self._event_counter: int = 0

    def should_transition(self, signal: Signal) -> bool:
        """Determine if a state transition should occur based on fused signals.

        Transition requires ALL of:
          1. Motion score above threshold (object physically moved)
          2. Proximity score above threshold (person is near)
          3. VL model confirms a different state

        Args:
            signal: Computed signals for the current keyframe pair.

        Returns:
            True if transition evidence is sufficient.
        """
        motion_ok = signal.motion_score >= self.settings.motion_threshold
        proximity_ok = signal.proximity_score >= (
            1.0 / (1.0 + self.settings.proximity_threshold / self.settings.proximity_norm_factor)
        )
        vl_confirms = (
            signal.vl_state != "unknown"
            and signal.vl_state != self.current_state
            and signal.vl_confidence >= self.settings.low_confidence_threshold
        )

        if not proximity_ok:
            return False

        if motion_ok and vl_confirms:
            return True

        weighted_score = (
            signal.motion_score * self.settings.motion_weight
            + signal.proximity_score * self.settings.proximity_weight
            + signal.vl_confidence * self.settings.vl_weight
        )
        if (
            weighted_score >= 0.6
            and signal.vl_state != self.current_state
            and signal.vl_state != "unknown"
            and signal.dwell_time >= self.settings.dwell_time_threshold
        ):
            return True

        return False

    def determine_target_state(self, signal: Signal) -> str | None:
        """Determine the target state based on signal evidence.

        Uses VL model state as primary indicator, constrained by
        valid transition rules.

        Args:
            signal: Computed signals for the current keyframe pair.

        Returns:
            Target state string, or None if no valid transition.
        """
        vl_state = signal.vl_state

        if vl_state == "unknown" or vl_state == self.current_state:
            return None

        allowed = VALID_TRANSITIONS.get(self.current_state, [])
        if vl_state in allowed:
            return vl_state

        return None

    def compute_event_confidence(self, signal: Signal) -> float:
        """Compute confidence score for a detected event.

        Weighted combination of all signal channels.

        Args:
            signal: Computed signals for the transition.

        Returns:
            Confidence between 0.0 and 1.0.
        """
        confidence = (
            signal.motion_score * self.settings.motion_weight
            + signal.proximity_score * self.settings.proximity_weight
            + signal.vl_confidence * self.settings.vl_weight
        )
        return round(min(max(confidence, 0.0), 1.0), 4)

    def process_signal(self, signal: Signal, agent_id: str = "agent_1") -> Event | None:
        """Process a signal and potentially produce a state transition and event.

        This is the main entry point called for each keyframe pair.

        Args:
            signal: Computed signals for the current keyframe pair.
            agent_id: The agent responsible for the interaction.

        Returns:
            Event if a transition occurred, None otherwise.
        """
        self.signals_log.append(signal)

        if not self.should_transition(signal):
            return None

        target_state = self.determine_target_state(signal)
        if target_state is None:
            return None

        prev_state = self.current_state
        transition_time = signal.timestamp

        self.state_history.append(
            StateSegment(
                state=prev_state,
                start=self._state_start_time,
                end=transition_time,
            )
        )

        self.current_state = target_state
        self._state_start_time = transition_time

        self._event_counter += 1
        event_id = f"evt_{self._event_counter}"

        action_template = ACTION_MAP.get(
            (prev_state, target_state), f"{prev_state}_to_{target_state}"
        )
        action_type = action_template.replace("{label}", self.object_label)

        start_offset = self.settings.start_offset
        end_offset = self.settings.end_offset

        event = Event(
            id=event_id,
            type=action_type,
            agent_id=agent_id,
            object_id=self.object_id,
            start_time=round(max(0.0, transition_time - 0.5 + start_offset), 3),
            end_time=round(transition_time + 0.5 + end_offset, 3),
            confidence=self.compute_event_confidence(signal),
            signals=EventSignals(
                motion_score=signal.motion_score,
                proximity_score=signal.proximity_score,
                vl_confidence=signal.vl_confidence,
            ),
        )

        self.events.append(event)
        return event

    def finalize(self, video_duration: float) -> ObjectStateTimeline:
        """Close the final state segment and return the complete state timeline.

        Args:
            video_duration: Total video duration in seconds.

        Returns:
            ObjectStateTimeline with all state segments.
        """
        self.state_history.append(
            StateSegment(
                state=self.current_state,
                start=self._state_start_time,
                end=video_duration,
            )
        )

        return ObjectStateTimeline(
            object_id=self.object_id,
            states=self.state_history,
        )


# ==========================================================================
# 3. MAIN PIPELINE — Orchestrates the full temporal analysis
# ==========================================================================


def _discover_objects(
    segmentations: list[dict],
) -> dict[str, dict]:
    """Discover unique objects across all keyframes.

    Returns a dict keyed by label with first-seen bbox and best score.
    """
    objects = {}
    for seg in segmentations:
        seg_objects = seg.get("objects", []) if isinstance(seg, dict) else seg.objects
        for obj in seg_objects:
            label = obj.get("label", "") if isinstance(obj, dict) else obj.label
            score = obj.get("score", 0.0) if isinstance(obj, dict) else obj.score
            bbox = obj.get("bbox", []) if isinstance(obj, dict) else obj.bbox
            if label not in objects:
                objects[label] = {
                    "label": label,
                    "bbox_initial": bbox,
                    "best_score": score,
                }
            elif score > objects[label]["best_score"]:
                objects[label]["best_score"] = score
    return objects


def _build_agent_tracks(
    detections: list[dict],
) -> list[Agent]:
    """Build agent trajectories from detection data.

    Currently assumes a single person (agent_1). Each detection frame
    with person boxes contributes a track point.
    """
    track_points = []
    for det in detections:
        t = det.get("timestamp", 0.0) if isinstance(det, dict) else det.timestamp
        boxes = det.get("person_boxes", []) if isinstance(det, dict) else det.person_boxes
        if boxes:
            track_points.append(AgentTrackPoint(time=t, bbox=boxes[0]))

    if not track_points:
        return []

    return [Agent(id="agent_1", label="person", track=track_points)]


def _get_initial_state(
    semantics: list[dict], object_label: str
) -> str:
    """Determine initial state for an object from the first semantic keyframe."""
    if not semantics:
        return "closed"

    first = semantics[0]
    state, confidence = extract_vl_signals(first, object_label)
    if state != "unknown" and confidence > 0.5:
        return state
    return "closed"


def _match_semantics_to_keyframe(
    semantics: list[dict], frame_index: int
) -> dict | None:
    """Find the semantic frame matching a given keyframe index."""
    for sem in semantics:
        idx = sem.get("frame_index", -1) if isinstance(sem, dict) else sem.frame_index
        if idx == frame_index:
            return sem
    return None


def _load_ontology() -> dict:
    """Load the object ontology for affordance mapping."""
    import json
    from pathlib import Path

    ontology_path = Path(__file__).parent.parent / "schemas" / "ontology_v0_1.json"
    if ontology_path.exists():
        with open(ontology_path, "r") as f:
            return json.load(f)
    return {}


def _get_affordances(object_label: str, ontology: dict) -> list[str]:
    """Look up affordances for an object type from the ontology."""
    object_types = ontology.get("object_types", {})
    obj_def = object_types.get(object_label, {})
    return obj_def.get("affordances", [])


def _get_category(object_label: str, ontology: dict) -> str:
    """Look up category for an object type from the ontology."""
    object_types = ontology.get("object_types", {})
    obj_def = object_types.get(object_label, {})
    return obj_def.get("category", "interactable")


def run_inference(
    detections: list[dict],
    segmentations: list[dict],
    semantics: list[dict],
    video_duration: float,
    settings: Settings | None = None,
) -> dict:
    """Run the full temporal inference pipeline.

    Takes raw model outputs and produces events, state timelines,
    tracked objects, and agent trajectories.

    Args:
        detections: YOLO person detections (all sampled frames).
        segmentations: SAM3 object segmentations (keyframes only).
        semantics: VL semantic annotations (keyframes only).
        video_duration: Total video duration in seconds.
        settings: Configuration with calibratable thresholds.

    Returns:
        Dict with keys:
            objects: list[TrackedObject]
            agents: list[Agent]
            events: list[Event]
            state_timeline: list[ObjectStateTimeline]
            signals_log: dict[str, list[Signal]] — per object label
    """
    if settings is None:
        settings = Settings()

    ontology = _load_ontology()

    discovered = _discover_objects(segmentations)
    agents = _build_agent_tracks(detections)
    agent_id = agents[0].id if agents else "agent_1"

    tracked_objects = []
    all_events = []
    all_timelines = []
    all_signals = {}

    for obj_idx, (label, obj_info) in enumerate(discovered.items(), start=1):
        object_id = f"obj_{obj_idx}"
        initial_state = _get_initial_state(semantics, label)
        affordances = _get_affordances(label, ontology)
        category = _get_category(label, ontology)

        tracked_obj = TrackedObject(
            id=object_id,
            label=label,
            category=category,
            affordances=affordances,
            initial_state=initial_state,
            bbox_initial=obj_info["bbox_initial"],
            confidence=obj_info["best_score"],
        )
        tracked_objects.append(tracked_obj)

        state_machine = ObjectStateMachine(
            object_id=object_id,
            object_label=label,
            initial_state=initial_state,
            settings=settings,
        )

        for i in range(1, len(segmentations)):
            seg_prev = segmentations[i - 1]
            seg_curr = segmentations[i]

            curr_frame_index = (
                seg_curr.get("frame_index", 0)
                if isinstance(seg_curr, dict)
                else seg_curr.frame_index
            )
            sem_curr = _match_semantics_to_keyframe(semantics, curr_frame_index)
            if sem_curr is None:
                continue

            signal = compute_signals_for_keyframe_pair(
                detections, seg_prev, seg_curr, sem_curr, label, settings
            )

            state_machine.process_signal(signal, agent_id=agent_id)

        timeline = state_machine.finalize(video_duration)
        all_timelines.append(timeline)
        all_events.extend(state_machine.events)
        all_signals[label] = state_machine.signals_log

    all_events.sort(key=lambda e: e.start_time)
    for i, event in enumerate(all_events, start=1):
        event.id = f"evt_{i}"

    return {
        "objects": tracked_objects,
        "agents": agents,
        "events": all_events,
        "state_timeline": all_timelines,
        "signals_log": all_signals,
    }
