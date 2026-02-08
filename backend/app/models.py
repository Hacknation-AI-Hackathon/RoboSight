"""
RoboSight Data Models

Pydantic models defining every data contract in the system.
Organized into four layers:

  1. Interface models  — match Dev A's model output formats exactly
  2. Internal models   — used by the inference engine (infer.py)
  3. Output models     — the final JSON deliverables
  4. API models        — FastAPI request/response schemas
"""

from pydantic import BaseModel, Field
from enum import Enum


# ==========================================================================
# 1. INTERFACE MODELS — Dev A outputs → Dev B inputs
# ==========================================================================


class DetectionFrame(BaseModel):
    """YOLO output for a single sampled frame (detect.py → infer.py).

    One entry per sampled frame. person_boxes may be empty if no person detected.
    """

    frame_index: int
    timestamp: float
    person_boxes: list[list[float]] = Field(
        default_factory=list,
        description="List of [x1, y1, x2, y2] bounding boxes for detected persons",
    )
    confidences: list[float] = Field(default_factory=list)


class SegmentedObject(BaseModel):
    """A single object detected by SAM3 within a keyframe."""

    label: str
    instance_id: int
    bbox: list[float] = Field(description="[x1, y1, x2, y2]")
    score: float
    mask_bbox_area: float


class SegmentationFrame(BaseModel):
    """SAM3 output for a single keyframe (segment.py → infer.py).

    One entry per keyframe only. Multiple objects per frame.
    """

    frame_index: int
    timestamp: float
    objects: list[SegmentedObject] = Field(default_factory=list)


class SemanticObject(BaseModel):
    """A single object's semantic annotation from the VL model."""

    label: str
    state: str
    confidence: float


class SemanticFrame(BaseModel):
    """VL model output for a single keyframe (semantics.py → infer.py).

    One entry per keyframe only. Matches keyframes from segmentations.
    """

    frame_index: int
    timestamp: float
    objects: list[SemanticObject] = Field(default_factory=list)
    action: str = "no_action"
    description: str = ""
    raw_response: str = ""


# ==========================================================================
# 2. INTERNAL MODELS — Inference engine internals
# ==========================================================================


class ObjectState(str, Enum):
    """Possible states for a trackable object."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    PARTIALLY_OPEN = "partially_open"
    UNKNOWN = "unknown"


class Signal(BaseModel):
    """Computed signals for a keyframe pair used in state transition decisions.

    Each signal captures a different evidence channel:
      - motion_score:    1 - IoU(bbox_t, bbox_t-1), high = object moved
      - proximity_score: inverse normalized distance person↔object
      - dwell_time:      accumulated seconds person within proximity threshold
      - vl_state:        state string from VL model
      - vl_confidence:   VL model's confidence in its prediction
    """

    frame_index: int
    timestamp: float
    object_label: str
    motion_score: float = 0.0
    proximity_score: float = 0.0
    dwell_time: float = 0.0
    vl_state: str = "unknown"
    vl_confidence: float = 0.0


class StateSegment(BaseModel):
    """A time range where an object holds a particular state."""

    state: str
    start: float
    end: float


class AgentTrackPoint(BaseModel):
    """A single point in an agent's trajectory."""

    time: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2]")


class Agent(BaseModel):
    """A tracked person (agent) with their full trajectory."""

    id: str
    label: str = "person"
    track: list[AgentTrackPoint] = Field(default_factory=list)


class TrackedObject(BaseModel):
    """An object tracked across frames with identity, affordances, and state."""

    id: str
    label: str
    category: str = "interactable"
    affordances: list[str] = Field(default_factory=list)
    initial_state: str = "closed"
    bbox_initial: list[float] = Field(default_factory=list)
    confidence: float = 0.0


class EventSignals(BaseModel):
    """Signal breakdown for a detected event."""

    motion_score: float = 0.0
    proximity_score: float = 0.0
    vl_confidence: float = 0.0


class Event(BaseModel):
    """A detected interaction event between an agent and an object."""

    id: str
    type: str
    agent_id: str
    object_id: str
    start_time: float
    end_time: float
    confidence: float
    signals: EventSignals = Field(default_factory=EventSignals)


# ==========================================================================
# 3. OUTPUT MODELS — Final JSON deliverables
# ==========================================================================


class VideoInfo(BaseModel):
    """Source video metadata."""

    source: str
    duration_seconds: float
    fps: float
    resolution: list[int] = Field(description="[width, height]")


class AffordanceEntry(BaseModel):
    """Per-object affordance map entry."""

    interactable: bool = True
    movable: bool = False
    traversable: bool = False
    states: list[str] = Field(default_factory=list)


class ObjectStateTimeline(BaseModel):
    """State sequence for a single object over the full video duration."""

    object_id: str
    states: list[StateSegment] = Field(default_factory=list)


class WorldGT(BaseModel):
    """Complete world_gt.json output — the primary deliverable.

    Contains everything a humanoid needs to understand the environment:
    objects, agents, state changes, interaction events, and affordances.
    """

    version: str = "0.1"
    video: VideoInfo
    objects: list[TrackedObject] = Field(default_factory=list)
    agents: list[Agent] = Field(default_factory=list)
    state_timeline: list[ObjectStateTimeline] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    affordance_map: dict[str, AffordanceEntry] = Field(default_factory=dict)


class TimelineEntry(BaseModel):
    """A single human-readable event entry for timeline.json."""

    time: float
    end_time: float
    event_type: str
    object_label: str
    description: str
    confidence: float


class ConfidenceSegment(BaseModel):
    """A segment flagged for human review due to low confidence."""

    event_id: str
    event_type: str
    start_time: float
    end_time: float
    confidence: float
    reason: str


class ConfidenceReport(BaseModel):
    """Full confidence_report.json — identifies which segments need review."""

    total_events: int
    high_confidence_count: int
    low_confidence_count: int
    review_segments: list[ConfidenceSegment] = Field(default_factory=list)
    overall_confidence: float = 0.0


# ==========================================================================
# 4. API MODELS — FastAPI request/response schemas
# ==========================================================================


class JobStatus(BaseModel):
    """Response model for job status queries."""

    job_id: str
    status: str = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None


class CorrectionAction(str, Enum):
    """Allowed correction actions from the frontend."""

    ADJUST_BOUNDARY = "adjust_boundary"
    REJECT = "reject"
    ADD_EVENT = "add_event"
    APPROVE = "approve"


class Correction(BaseModel):
    """A single correction from a human reviewer."""

    event_id: str | None = None
    action: CorrectionAction
    corrected_start: float | None = None
    corrected_end: float | None = None
    type: str | None = Field(
        default=None, description="Event type for add_event action"
    )
    start_time: float | None = Field(
        default=None, description="Start time for add_event action"
    )
    end_time: float | None = Field(
        default=None, description="End time for add_event action"
    )


class CorrectionPayload(BaseModel):
    """Payload from frontend containing all corrections for a job."""

    corrections: list[Correction]


class MetricsResult(BaseModel):
    """Evaluation metrics for a pipeline run."""

    event_precision: float = 0.0
    event_recall: float = 0.0
    avg_boundary_error: float = 0.0
    review_percentage: float = 0.0
    labeling_time_saved_estimate: float = 0.0
    total_events_detected: int = 0
    total_events_ground_truth: int = 0
