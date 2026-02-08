"""
YOLO Detection Wrapper

Detects persons in video frames. Supports two backends:
  - "modal": Calls Modal-hosted YOLOServer (cloud GPU)
  - "local": Uses local NVIDIA GPU via LocalYOLOServer

Runs on ALL sampled frames (not just keyframes).

Output format (detections):
    [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "person_boxes": [[x1, y1, x2, y2], ...],
            "confidences": [0.95, ...]
        },
        ...
    ]
"""


def run_detection(
    frames_bytes: list[bytes],
    frames_info: list[dict],
    confidence: float = 0.5,
    batch_size: int = 16,
    backend: str = "modal",
) -> list[dict]:
    """Run YOLO person detection on all sampled frames.

    Args:
        frames_bytes: JPEG-encoded frames (one per sampled frame).
        frames_info: Metadata for each frame (index, timestamp, is_keyframe).
        confidence: YOLO confidence threshold.
        batch_size: How many frames to send per batch call.
        backend: "modal" for cloud GPU, "local" for local NVIDIA GPU.

    Returns:
        List of detection dicts, one per frame.
    """
    if backend == "local":
        return _run_local(frames_bytes, frames_info, confidence, batch_size)
    else:
        return _run_modal(frames_bytes, frames_info, confidence, batch_size)


def _run_modal(
    frames_bytes: list[bytes],
    frames_info: list[dict],
    confidence: float,
    batch_size: int,
) -> list[dict]:
    """Run detection via Modal cloud GPU."""
    import modal

    YOLOServer = modal.Cls.from_name("robosight-gpu", "YOLOServer")
    server = YOLOServer()

    detections = []
    for i in range(0, len(frames_bytes), batch_size):
        batch_bytes = frames_bytes[i : i + batch_size]
        batch_info = frames_info[i : i + batch_size]
        batch_results = server.detect_batch.remote(batch_bytes, confidence)

        for info, result in zip(batch_info, batch_results):
            detections.append(
                {
                    "frame_index": info["index"],
                    "timestamp": info["timestamp"],
                    "person_boxes": result["boxes"],
                    "confidences": result["confidences"],
                }
            )

    return detections


def _run_local(
    frames_bytes: list[bytes],
    frames_info: list[dict],
    confidence: float,
    batch_size: int,
) -> list[dict]:
    """Run detection on local NVIDIA GPU."""
    from app.local_app.gpu_models import LocalYOLOServer, get_server

    server = get_server(LocalYOLOServer)

    detections = []
    for i in range(0, len(frames_bytes), batch_size):
        batch_bytes = frames_bytes[i : i + batch_size]
        batch_info = frames_info[i : i + batch_size]
        batch_results = server.detect_batch(batch_bytes, confidence)

        for info, result in zip(batch_info, batch_results):
            detections.append(
                {
                    "frame_index": info["index"],
                    "timestamp": info["timestamp"],
                    "person_boxes": result["boxes"],
                    "confidences": result["confidences"],
                }
            )

    return detections
