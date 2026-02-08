"""
Pipeline Orchestrator — Parallel Model Execution

Runs YOLO, SAM3, and VL model inference in parallel using concurrent futures.
All three models are independent — they each take raw frame bytes and produce
independent outputs. The temporal compiler (infer.py) is the first stage
that needs all three outputs combined.

Execution diagram:
    ┌─── YOLO (all frames)     ──→ detections     ─┐
    │                                                │
    ├─── SAM3 (keyframes only) ──→ segmentations   ─┼──→ infer.py
    │                                                │
    └─── VL   (keyframes only) ──→ semantics       ─┘

All three branches run concurrently:
  - Modal backend: separate cloud GPU containers
  - Local backend: shared local GPU (sequential on GPU, but I/O overlapped)

Thread-based parallelism is correct because:
  - Modal: local threads just wait on remote I/O
  - Local: GIL is released during CUDA operations
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from app.pipeline import detect, segment, semantics


def run_models_parallel(
    all_frames_bytes: list[bytes],
    all_frames_info: list[dict],
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    object_prompts: list[str],
    backend: str = "modal",
    yolo_confidence: float = 0.5,
    sam3_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
    vl_max_tokens: int = 256,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run all three model servers in parallel.

    Args:
        all_frames_bytes: JPEG bytes for every sampled frame.
        all_frames_info: Metadata for every sampled frame.
        keyframe_bytes: JPEG bytes for keyframes only.
        keyframes_info: Metadata for keyframes only.
        object_prompts: Text prompts for SAM3 (e.g. ["drawer", "handle"]).
        backend: "modal" for cloud GPU, "local" for local GPU,
                 "local_vllm" for local GPU with vLLM VL acceleration.
        yolo_confidence: YOLO detection confidence threshold.
        sam3_threshold: SAM3 detection threshold.
        sam3_mask_threshold: SAM3 mask binarization threshold.
        vl_max_tokens: Max tokens for VL model generation.
        progress_callback: Optional fn(stage: str, progress: float) for updates.

    Returns:
        dict with keys:
            detections: list[dict]    — YOLO output (all frames)
            segmentations: list[dict] — SAM3 output (keyframes)
            semantics: list[dict]     — VL output (keyframes)
            errors: dict              — any errors encountered (empty if none)
    """
    # Determine VL backend variant
    vl_backend = backend if backend != "local_vllm" else "local_vllm"
    base_backend = "local" if backend.startswith("local") else "modal"

    def _run_yolo():
        if progress_callback:
            progress_callback("yolo_start", 0.2)
        result = detect.run_detection(
            all_frames_bytes, all_frames_info, yolo_confidence, backend=base_backend
        )
        if progress_callback:
            progress_callback("yolo_done", 0.4)
        return result

    def _run_sam3():
        if progress_callback:
            progress_callback("sam3_start", 0.2)
        result = segment.run_segmentation(
            keyframe_bytes, keyframes_info, object_prompts,
            sam3_threshold, sam3_mask_threshold, backend=base_backend
        )
        if progress_callback:
            progress_callback("sam3_done", 0.5)
        return result

    def _run_vl():
        if progress_callback:
            progress_callback("vl_start", 0.2)
        result = semantics.run_semantics(
            keyframe_bytes, keyframes_info, vl_max_tokens, backend=vl_backend
        )
        if progress_callback:
            progress_callback("vl_done", 0.6)
        return result

    # For local backend, we use 3 threads but the GPU will serialize
    # For Modal backend, each thread calls a different remote container
    # Either way, ThreadPoolExecutor is correct (I/O bound, not CPU bound)
    results = {}
    errors = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(_run_yolo): "detections",
            executor.submit(_run_sam3): "segmentations",
            executor.submit(_run_vl): "semantics",
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"[Orchestrator] {name} completed: {len(results[name])} entries")
            except Exception as e:
                errors[name] = str(e)
                results[name] = []  # Empty fallback so pipeline can continue
                print(f"[Orchestrator] ERROR in {name}: {e}")

    results["errors"] = errors
    if errors:
        print(f"[Orchestrator] Completed with errors: {list(errors.keys())}")
    else:
        print("[Orchestrator] All models completed successfully")

    return results


def run_models_sequential(
    all_frames_bytes: list[bytes],
    all_frames_info: list[dict],
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    object_prompts: list[str],
    backend: str = "local",
    yolo_confidence: float = 0.5,
    sam3_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
    vl_max_tokens: int = 256,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run all three models sequentially (useful for debugging or low-VRAM GPUs).

    Same interface and output as run_models_parallel.
    """
    vl_backend = backend if backend != "local_vllm" else "local_vllm"
    base_backend = "local" if backend.startswith("local") else "modal"
    errors = {}

    # 1. YOLO
    if progress_callback:
        progress_callback("yolo_start", 0.1)
    try:
        detections = detect.run_detection(
            all_frames_bytes, all_frames_info, yolo_confidence, backend=base_backend
        )
        print(f"[Orchestrator] detections completed: {len(detections)} entries")
    except Exception as e:
        errors["detections"] = str(e)
        detections = []
        print(f"[Orchestrator] ERROR in detections: {e}")
    if progress_callback:
        progress_callback("yolo_done", 0.3)

    # 2. SAM3
    if progress_callback:
        progress_callback("sam3_start", 0.3)
    try:
        segmentations = segment.run_segmentation(
            keyframe_bytes, keyframes_info, object_prompts,
            sam3_threshold, sam3_mask_threshold, backend=base_backend
        )
        print(f"[Orchestrator] segmentations completed: {len(segmentations)} entries")
    except Exception as e:
        errors["segmentations"] = str(e)
        segmentations = []
        print(f"[Orchestrator] ERROR in segmentations: {e}")
    if progress_callback:
        progress_callback("sam3_done", 0.5)

    # 3. VL
    if progress_callback:
        progress_callback("vl_start", 0.5)
    try:
        sem = semantics.run_semantics(
            keyframe_bytes, keyframes_info, vl_max_tokens, backend=vl_backend
        )
        print(f"[Orchestrator] semantics completed: {len(sem)} entries")
    except Exception as e:
        errors["semantics"] = str(e)
        sem = []
        print(f"[Orchestrator] ERROR in semantics: {e}")
    if progress_callback:
        progress_callback("vl_done", 0.7)

    return {
        "detections": detections,
        "segmentations": segmentations,
        "semantics": sem,
        "errors": errors,
    }
