"""
Annotated Video Generator

Overlays bounding boxes, state labels, and event banners on the original
video to create a visual proof of the world → ground truth transformation.

Overlays:
  - Green bboxes for persons (from detections)
  - Colored bboxes for objects with labels (from segmentations)
  - State labels next to objects (from state_timeline)
  - Event banners at the top (from events)
  - Timeline bar at the bottom showing progress
"""

import cv2
import numpy as np
from pathlib import Path


# Color palette for different object types (BGR format for OpenCV)
COLORS = {
    "person": (0, 255, 0),       # Green
    "drawer": (255, 165, 0),     # Orange
    "door": (0, 165, 255),       # Orange-red
    "handle": (255, 0, 255),     # Magenta
    "cabinet": (0, 255, 255),    # Yellow
    "default": (255, 255, 0),    # Cyan
}

# State colors
STATE_COLORS = {
    "open": (0, 255, 0),         # Green
    "closed": (0, 0, 255),       # Red
    "opening": (0, 255, 255),    # Yellow
    "closing": (0, 165, 255),    # Orange
    "partially_open": (0, 200, 200),
    "in_motion": (255, 255, 0),  # Cyan
}

EVENT_COLOR = (50, 50, 255)      # Red-ish banner


def generate_annotated_video(
    video_path: str,
    output_path: str,
    detections: list[dict],
    segmentations: list[dict],
    events: list[dict],
    state_timeline: list[dict],
    video_info: dict,
    sample_fps: float = 5.0,
) -> str:
    """Create annotated video with overlaid ground truth visualizations.

    Args:
        video_path: Path to original input video.
        output_path: Path for annotated output video.
        detections: Person detections (one per sampled frame).
        segmentations: Object segmentations (one per keyframe).
        events: Detected events with start/end times.
        state_timeline: Object state changes over time.
        video_info: Video metadata (fps, resolution, duration).
        sample_fps: Sampling rate used during ingestion.

    Returns:
        Path to the generated annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = video_info["fps"]
    width, height = video_info["resolution"]
    duration = video_info["duration_seconds"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build lookup structures for efficient per-frame annotation
    detection_lookup = _build_detection_lookup(detections, fps, sample_fps)
    segmentation_lookup = _build_segmentation_lookup(segmentations)
    state_lookup = _build_state_lookup(state_timeline)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # --- Draw person bounding boxes ---
        nearest_det = _get_nearest_detection(detection_lookup, timestamp)
        if nearest_det:
            for box, conf in zip(
                nearest_det["person_boxes"], nearest_det["confidences"]
            ):
                _draw_bbox(
                    frame,
                    box,
                    label=f"person {conf:.2f}",
                    color=COLORS["person"],
                    thickness=2,
                )

        # --- Draw object bounding boxes with state labels ---
        nearest_seg = _get_nearest_segmentation(segmentation_lookup, timestamp)
        if nearest_seg:
            for obj in nearest_seg["objects"]:
                label = obj["label"]
                color = COLORS.get(label, COLORS["default"])

                # Get current state for this object
                current_state = _get_current_state(state_lookup, label, timestamp)
                state_str = f" [{current_state}]" if current_state else ""
                display_label = f"{label}{state_str} {obj['score']:.2f}"

                _draw_bbox(frame, obj["bbox"], label=display_label, color=color, thickness=2)

        # --- Draw event banner ---
        active_event = _get_active_event(events, timestamp)
        if active_event:
            _draw_event_banner(frame, active_event, width)

        # --- Draw timeline bar ---
        _draw_timeline_bar(frame, timestamp, duration, events, width, height)

        # --- Draw timestamp ---
        _draw_timestamp(frame, timestamp)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return output_path


# ---------------------------------------------------------------------------
# Lookup builders
# ---------------------------------------------------------------------------

def _build_detection_lookup(
    detections: list[dict], video_fps: float, sample_fps: float
) -> list[dict]:
    """Index detections by timestamp for quick nearest-neighbor lookup."""
    return sorted(detections, key=lambda d: d["timestamp"])


def _build_segmentation_lookup(segmentations: list[dict]) -> list[dict]:
    """Index segmentations by timestamp."""
    return sorted(segmentations, key=lambda s: s["timestamp"])


def _build_state_lookup(state_timeline: list[dict]) -> dict:
    """Build {object_label: [{state, start, end}, ...]} for quick lookup."""
    lookup = {}
    for entry in state_timeline:
        obj_id = entry.get("object_id", "")
        states = entry.get("states", [])
        lookup[obj_id] = states
    return lookup


# ---------------------------------------------------------------------------
# Per-frame lookup helpers
# ---------------------------------------------------------------------------

def _get_nearest_detection(detections: list[dict], timestamp: float) -> dict | None:
    """Find detection nearest to timestamp."""
    if not detections:
        return None

    best = None
    best_diff = float("inf")
    for det in detections:
        diff = abs(det["timestamp"] - timestamp)
        if diff < best_diff:
            best_diff = diff
            best = det
        elif diff > best_diff:
            # Sorted, so once diff increases we can stop
            break

    # Only use if within 0.5 seconds
    return best if best and best_diff < 0.5 else None


def _get_nearest_segmentation(
    segmentations: list[dict], timestamp: float
) -> dict | None:
    """Find segmentation nearest to timestamp (keyframe-based, so persist longer)."""
    if not segmentations:
        return None

    best = None
    best_diff = float("inf")
    for seg in segmentations:
        diff = abs(seg["timestamp"] - timestamp)
        if diff < best_diff:
            best_diff = diff
            best = seg

    # Segmentations are keyframe-based, persist for up to 2 seconds
    return best if best and best_diff < 2.0 else None


def _get_current_state(
    state_lookup: dict, object_label: str, timestamp: float
) -> str | None:
    """Get the current state of an object at a given timestamp."""
    # Try matching by object_id pattern (obj_1, obj_2, etc.) or label
    for obj_id, states in state_lookup.items():
        for state_entry in states:
            if state_entry["start"] <= timestamp <= state_entry["end"]:
                return state_entry["state"]
    return None


def _get_active_event(events: list[dict], timestamp: float) -> dict | None:
    """Check if any event is active at the current timestamp."""
    for event in events:
        if event["start_time"] <= timestamp <= event["end_time"]:
            return event
    return None


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_bbox(
    frame: np.ndarray,
    bbox: list[float],
    label: str = "",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    """Draw a bounding box with an optional label."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size

        # Background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )


def _draw_event_banner(
    frame: np.ndarray, event: dict, frame_width: int
):
    """Draw an event banner at the top of the frame."""
    event_type = event.get("type", "unknown")
    confidence = event.get("confidence", 0.0)
    text = f"EVENT: {event_type} (conf: {confidence:.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    # Semi-transparent banner background
    banner_height = text_h + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), EVENT_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Text
    text_x = (frame_width - text_w) // 2
    cv2.putText(
        frame,
        text,
        (text_x, text_h + 10),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )


def _draw_timeline_bar(
    frame: np.ndarray,
    timestamp: float,
    duration: float,
    events: list[dict],
    frame_width: int,
    frame_height: int,
):
    """Draw a timeline bar at the bottom showing progress and events."""
    bar_height = 20
    bar_y = frame_height - bar_height - 10
    bar_x = 20
    bar_width = frame_width - 40

    # Background bar
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (60, 60, 60),
        -1,
    )

    # Event regions on the bar
    if duration > 0:
        for event in events:
            evt_start = int(bar_x + (event["start_time"] / duration) * bar_width)
            evt_end = int(bar_x + (event["end_time"] / duration) * bar_width)
            cv2.rectangle(
                frame,
                (evt_start, bar_y),
                (evt_end, bar_y + bar_height),
                EVENT_COLOR,
                -1,
            )

        # Progress indicator
        progress_x = int(bar_x + (timestamp / duration) * bar_width)
        cv2.line(
            frame,
            (progress_x, bar_y - 5),
            (progress_x, bar_y + bar_height + 5),
            (255, 255, 255),
            2,
        )


def _draw_timestamp(frame: np.ndarray, timestamp: float):
    """Draw timestamp in the bottom-left corner."""
    text = f"{timestamp:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, frame.shape[0] - 40), font, 0.5, (255, 255, 255), 1)
