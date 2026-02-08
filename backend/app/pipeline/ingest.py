"""
Video Ingestion & Frame Sampling

Decodes raw video, samples frames at a target FPS, and detects keyframes
based on pixel-level motion between consecutive frames.

Output format (frames_info):
    [
        {
            "index": 0,              # sequential sample index
            "frame_idx_original": 0,  # original video frame number
            "timestamp": 0.0,         # seconds into video
            "is_keyframe": True       # significant visual change detected
        },
        ...
    ]

Output format (video_info):
    {
        "duration_seconds": 30.0,
        "fps": 30.0,
        "resolution": [1920, 1080],
        "total_sampled_frames": 150,
        "total_keyframes": 25
    }
"""

import cv2
import numpy as np
from pathlib import Path


def sample_frames(
    video_path: str,
    target_fps: float = 5.0,
    keyframe_threshold: float = 30.0,
) -> tuple[list[dict], dict]:
    """Decode video and sample frames at target FPS with keyframe detection.

    Args:
        video_path: Path to input video file.
        target_fps: Desired sampling rate (frames per second).
        keyframe_threshold: Mean absolute pixel difference threshold for
            marking a frame as a keyframe. Higher = fewer keyframes.

    Returns:
        frames_info: List of frame metadata dicts.
        video_info: Video-level metadata dict.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video cannot be opened or has no frames.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or video_fps == 0:
        cap.release()
        raise ValueError(f"Video has no frames or invalid FPS: {video_path}")

    duration = total_frames / video_fps

    # Calculate frame interval for target sampling rate
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frames_info = []
    prev_frame_gray = None
    frame_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps

            # Convert to grayscale for keyframe detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Keyframe detection: mean absolute pixel difference
            is_keyframe = False
            if prev_frame_gray is not None:
                diff = np.mean(
                    np.abs(gray.astype(np.float32) - prev_frame_gray.astype(np.float32))
                )
                if diff > keyframe_threshold:
                    is_keyframe = True
            else:
                # First frame is always a keyframe
                is_keyframe = True

            prev_frame_gray = gray.copy()

            frames_info.append(
                {
                    "index": sample_idx,
                    "frame_idx_original": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "is_keyframe": is_keyframe,
                }
            )
            sample_idx += 1

        frame_idx += 1

    cap.release()

    # Ensure at least the last frame is a keyframe too
    if frames_info and not frames_info[-1]["is_keyframe"]:
        frames_info[-1]["is_keyframe"] = True

    # If very few keyframes detected, add evenly spaced ones
    total_keyframes = sum(1 for f in frames_info if f["is_keyframe"])
    if total_keyframes < 3 and len(frames_info) >= 3:
        # Force keyframes at roughly even intervals
        step = max(1, len(frames_info) // 5)
        for i in range(0, len(frames_info), step):
            frames_info[i]["is_keyframe"] = True
        total_keyframes = sum(1 for f in frames_info if f["is_keyframe"])

    video_info = {
        "source": path.name,
        "duration_seconds": round(duration, 3),
        "fps": video_fps,
        "resolution": [width, height],
        "total_sampled_frames": sample_idx,
        "total_keyframes": total_keyframes,
    }

    return frames_info, video_info


def extract_frame_bytes(video_path: str, frame_idx_original: int) -> bytes:
    """Extract a single frame as JPEG bytes by its original video frame index.

    Args:
        video_path: Path to video file.
        frame_idx_original: The original frame number in the video.

    Returns:
        JPEG-encoded bytes of the frame.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_original)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_idx_original} from {video_path}")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()


def extract_frames_batch(
    video_path: str, frame_indices: list[int]
) -> list[bytes]:
    """Extract multiple frames efficiently with a single sequential video pass.

    Reads through the video once, collecting frames at the requested
    original frame indices. Much faster than seeking to each frame individually.

    Args:
        video_path: Path to video file.
        frame_indices: List of original video frame numbers to extract.

    Returns:
        List of JPEG-encoded bytes, in the same order as frame_indices.
    """
    if not frame_indices:
        return []

    cap = cv2.VideoCapture(video_path)
    target_set = set(frame_indices)
    max_target = max(target_set)
    results = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            results[frame_idx] = buffer.tobytes()

        # Stop early if we've collected all targets
        if len(results) == len(target_set):
            break

        # Stop if we've passed the last target
        if frame_idx > max_target:
            break

        frame_idx += 1

    cap.release()

    # Return in the requested order
    missing = [idx for idx in frame_indices if idx not in results]
    if missing:
        print(f"Warning: Could not extract frames at indices: {missing}")

    return [results.get(idx, b"") for idx in frame_indices]


def get_keyframe_data(
    frames_info: list[dict],
) -> tuple[list[int], list[dict]]:
    """Extract keyframe indices and info from frames_info.

    Args:
        frames_info: Full list of frame metadata from sample_frames().

    Returns:
        keyframe_original_indices: Original video frame numbers for keyframes.
        keyframes_info: Subset of frames_info that are keyframes.
    """
    keyframes_info = [f for f in frames_info if f["is_keyframe"]]
    keyframe_original_indices = [f["frame_idx_original"] for f in keyframes_info]
    return keyframe_original_indices, keyframes_info
