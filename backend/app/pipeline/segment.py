"""
SAM3 Segmentation Wrapper

Segments objects in keyframes using text prompts. Supports two backends:
  - "modal": Calls Modal-hosted SAM3Server (cloud GPU)
  - "local": Uses local NVIDIA GPU via LocalSAM3Server

Runs on KEYFRAMES ONLY.

Output format (segmentations):
    [
        {
            "frame_index": 5,
            "timestamp": 1.0,
            "objects": [
                {
                    "label": "drawer",
                    "instance_id": 0,
                    "bbox": [400.0, 300.0, 600.0, 450.0],
                    "score": 0.92,
                    "mask_bbox_area": 15000.0
                }
            ]
        },
        ...
    ]
"""


def run_segmentation(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    object_prompts: list[str],
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    backend: str = "modal",
) -> list[dict]:
    """Run SAM3 text-prompted segmentation on keyframes.

    Args:
        keyframe_bytes: JPEG-encoded keyframes.
        keyframes_info: Metadata for each keyframe (index, timestamp).
        object_prompts: Text prompts for objects to detect, e.g. ["drawer", "handle"].
        threshold: Detection confidence threshold.
        mask_threshold: Mask binarization threshold.
        backend: "modal" for cloud GPU, "local" for local NVIDIA GPU.

    Returns:
        List of segmentation dicts, one per keyframe.
    """
    if backend == "local":
        return _run_local(
            keyframe_bytes, keyframes_info, object_prompts, threshold, mask_threshold
        )
    else:
        return _run_modal(
            keyframe_bytes, keyframes_info, object_prompts, threshold, mask_threshold
        )


def _run_modal(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    object_prompts: list[str],
    threshold: float,
    mask_threshold: float,
) -> list[dict]:
    """Run segmentation via Modal cloud GPU."""
    import modal

    SAM3Server = modal.Cls.from_name("robosight-gpu", "SAM3Server")
    server = SAM3Server()

    segmentations = []
    for frame_bytes, info in zip(keyframe_bytes, keyframes_info):
        objects = server.segment_objects.remote(
            frame_bytes, object_prompts, threshold, mask_threshold
        )
        segmentations.append(
            {
                "frame_index": info["index"],
                "timestamp": info["timestamp"],
                "objects": objects,
            }
        )

    return segmentations


def _run_local(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    object_prompts: list[str],
    threshold: float,
    mask_threshold: float,
) -> list[dict]:
    """Run segmentation on local NVIDIA GPU."""
    from app.local_app.gpu_models import LocalSAM3Server, get_server

    server = get_server(LocalSAM3Server)

    segmentations = []
    for frame_bytes, info in zip(keyframe_bytes, keyframes_info):
        objects = server.segment_objects(
            frame_bytes, object_prompts, threshold, mask_threshold
        )
        segmentations.append(
            {
                "frame_index": info["index"],
                "timestamp": info["timestamp"],
                "objects": objects,
            }
        )

    return segmentations
