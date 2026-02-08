"""
FastAPI Application

Routes + pipeline orchestration for the RoboSight backend.
All endpoints are async. Heavy pipeline work runs in background
via asyncio.create_task + asyncio.to_thread.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.job_manager import JobManager
from app.models import (
    JobStatus,
    CorrectionPayload,
    Correction,
    CorrectionAction,
    ReviewPayload,
    ReviewAction,
    MetricsResult,
)
from app.pipeline import infer, calibrate, compile, eval


settings = Settings()
job_manager = JobManager(settings)

app = FastAPI(
    title="RoboSight",
    description="Backend pipeline converting raw human video into structured ground truth for humanoid navigation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# PIPELINE ORCHESTRATION
# ==========================================================================


def _run_pipeline_sync(job_id: str, object_prompts: list[str]) -> None:
    """Run the full pipeline synchronously (called via asyncio.to_thread).

    Stages:
      1. Ingest video → frame sampling
      2. Run models (YOLO + SAM3 + VL) via orchestrator
      3. Run temporal inference
      4. Compile outputs (world_gt, timeline, confidence_report)
      5. Generate annotated video
      6. Compute baseline metrics
    """
    try:
        job_manager.update_progress(job_id, 0.05, "ingesting")

        video_path = str(job_manager.get_input_video_path(job_id))

        from app.pipeline.ingest import (
            sample_frames,
            extract_frames_batch,
            get_keyframe_data,
        )

        frames_info, video_info = sample_frames(
            video_path,
            target_fps=settings.sample_fps,
            keyframe_threshold=settings.keyframe_threshold,
        )
        job_manager.update_progress(job_id, 0.1, "extracting_frames")

        all_frame_indices = [f["frame_idx_original"] for f in frames_info]
        all_frames_bytes = extract_frames_batch(video_path, all_frame_indices)

        keyframe_indices, keyframes_info = get_keyframe_data(frames_info)
        keyframe_bytes = extract_frames_batch(video_path, keyframe_indices)

        job_manager.update_progress(job_id, 0.15, "running_models")

        prompts = object_prompts if object_prompts else settings.object_prompts

        from app.pipeline.orchestrator import run_models_parallel, run_models_sequential

        run_fn = run_models_parallel if settings.parallel_models else run_models_sequential

        def progress_cb(stage: str, progress: float):
            job_manager.update_progress(job_id, 0.15 + progress * 0.45, stage)

        model_outputs = run_fn(
            all_frames_bytes=all_frames_bytes,
            all_frames_info=frames_info,
            keyframe_bytes=keyframe_bytes,
            keyframes_info=keyframes_info,
            object_prompts=prompts,
            backend=settings.backend,
            yolo_confidence=settings.yolo_confidence,
            sam3_threshold=settings.sam3_threshold,
            sam3_mask_threshold=settings.sam3_mask_threshold,
            vl_max_tokens=settings.vl_max_tokens,
            progress_callback=progress_cb,
        )

        detections = model_outputs["detections"]
        segmentations = model_outputs["segmentations"]
        semantics = model_outputs["semantics"]

        job_manager.save_artifact(job_id, "detections.json", detections)
        job_manager.save_artifact(job_id, "segmentations.json", segmentations)
        job_manager.save_artifact(job_id, "semantics.json", semantics)
        job_manager.save_artifact(job_id, "video_info.json", video_info)

        job_manager.update_progress(job_id, 0.65, "running_inference")

        inference_result = infer.run_inference(
            detections=detections,
            segmentations=segmentations,
            semantics=semantics,
            video_duration=video_info["duration_seconds"],
            settings=settings,
        )

        job_manager.update_progress(job_id, 0.75, "compiling_outputs")

        world_gt = compile.compile_world_gt(inference_result, video_info)
        timeline = compile.compile_timeline(
            inference_result["events"], inference_result["objects"]
        )
        confidence_report = compile.compile_confidence_report(
            inference_result["events"], inference_result["objects"], settings
        )

        job_manager.save_artifact(job_id, "world_gt.json", world_gt)
        job_manager.save_artifact(
            job_id, "timeline.json", [e.model_dump() for e in timeline]
        )
        job_manager.save_artifact(job_id, "confidence_report.json", confidence_report)

        job_manager.update_progress(job_id, 0.85, "annotating_video")

        from app.pipeline.annotate import generate_annotated_video

        output_video_path = str(
            job_manager._job_dir(job_id) / "annotated_video.mp4"
        )
        generate_annotated_video(
            video_path=video_path,
            output_path=output_video_path,
            detections=detections,
            segmentations=segmentations,
            events=[e.model_dump() for e in inference_result["events"]],
            state_timeline=[t.model_dump() for t in inference_result["state_timeline"]],
            video_info=video_info,
            sample_fps=settings.sample_fps,
        )

        job_manager.update_progress(job_id, 0.95, "computing_metrics")

        baseline_metrics = eval.compute_metrics(
            events=inference_result["events"],
            corrections=None,
            confidence_report=confidence_report,
            video_duration=video_info["duration_seconds"],
        )
        job_manager.save_artifact(job_id, "metrics_baseline.json", baseline_metrics)

        job_manager.mark_complete(job_id)

    except Exception as e:
        job_manager.mark_failed(job_id, str(e))
        raise


def _run_calibrated_rerun_sync(job_id: str) -> None:
    """Rerun inference with calibrated settings using cached model outputs."""
    try:
        job_manager.mark_rerunning(job_id)
        job_manager.update_progress(job_id, 0.1, "loading_cached_outputs")

        detections = job_manager.load_artifact(job_id, "detections.json")
        segmentations = job_manager.load_artifact(job_id, "segmentations.json")
        semantics = job_manager.load_artifact(job_id, "semantics.json")
        video_info = job_manager.load_artifact(job_id, "video_info.json")

        corrections_data = job_manager.load_artifact(job_id, "corrections.json")
        corrections = CorrectionPayload(**corrections_data).corrections

        events_data = job_manager.load_artifact(job_id, "world_gt.json")
        from app.models import Event, EventSignals
        prev_events = [
            Event(**evt) for evt in events_data.get("events", [])
        ]

        job_manager.update_progress(job_id, 0.2, "calibrating")

        calibrated_params = calibrate.calibrate_from_corrections(
            settings=settings,
            events=prev_events,
            corrections=corrections,
        )

        calibrated_settings = settings.apply_calibration(calibrated_params)
        job_manager.save_artifact(job_id, "calibration.json", calibrated_params)

        job_manager.update_progress(job_id, 0.4, "rerunning_inference")

        inference_result = infer.run_inference(
            detections=detections,
            segmentations=segmentations,
            semantics=semantics,
            video_duration=video_info["duration_seconds"],
            settings=calibrated_settings,
        )

        job_manager.update_progress(job_id, 0.6, "compiling_outputs")

        world_gt = compile.compile_world_gt(inference_result, video_info)
        timeline = compile.compile_timeline(
            inference_result["events"], inference_result["objects"]
        )
        confidence_report = compile.compile_confidence_report(
            inference_result["events"], inference_result["objects"], calibrated_settings
        )

        job_manager.save_artifact(job_id, "world_gt.json", world_gt)
        job_manager.save_artifact(
            job_id, "timeline.json", [e.model_dump() for e in timeline]
        )
        job_manager.save_artifact(job_id, "confidence_report.json", confidence_report)

        job_manager.update_progress(job_id, 0.8, "annotating_video")

        video_path = str(job_manager.get_input_video_path(job_id))
        from app.pipeline.annotate import generate_annotated_video

        output_video_path = str(
            job_manager._job_dir(job_id) / "annotated_video.mp4"
        )
        generate_annotated_video(
            video_path=video_path,
            output_path=output_video_path,
            detections=detections,
            segmentations=segmentations,
            events=[e.model_dump() for e in inference_result["events"]],
            state_timeline=[t.model_dump() for t in inference_result["state_timeline"]],
            video_info=video_info,
            sample_fps=calibrated_settings.sample_fps,
        )

        job_manager.update_progress(job_id, 0.9, "computing_metrics")

        calibrated_metrics = eval.compute_metrics(
            events=inference_result["events"],
            corrections=corrections,
            confidence_report=confidence_report,
            video_duration=video_info["duration_seconds"],
        )
        job_manager.save_artifact(job_id, "metrics_calibrated.json", calibrated_metrics)

        job_manager.mark_complete(job_id)

    except Exception as e:
        job_manager.mark_failed(job_id, str(e))
        raise


# ==========================================================================
# API ROUTES
# ==========================================================================


@app.get("/health")
async def health_check():
    """Server health check with configuration summary."""
    job_count = len(job_manager.list_jobs())
    return {
        "status": "healthy",
        "version": app.version,
        "backend": settings.backend,
        "parallel_models": settings.parallel_models,
        "sample_fps": settings.sample_fps,
        "object_prompts": settings.object_prompts,
        "total_jobs": job_count,
    }


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    object_prompts: str = Form(default=""),
):
    """Upload a video and start the ground truth generation pipeline.

    Returns the job_id and a URL to the converted mp4 immediately
    so the frontend can play the video while processing runs.
    """
    job_id = job_manager.create_job()

    video_bytes = await file.read()
    job_manager.save_input_video(job_id, video_bytes, file.filename or "input.mp4")

    from app.pipeline.convert import ensure_mp4

    raw_path = str(job_manager.get_input_video_path(job_id))
    video_path = await asyncio.to_thread(ensure_mp4, raw_path)

    prompts = []
    if object_prompts:
        prompts = [p.strip() for p in object_prompts.split(",") if p.strip()]

    asyncio.create_task(
        asyncio.to_thread(_run_pipeline_sync, job_id, prompts)
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "input_video_url": f"/jobs/{job_id}/input-video",
    }


@app.get("/jobs/{job_id}/input-video")
async def get_input_video(job_id: str):
    """Serve the converted mp4 input video for frontend playback."""
    try:
        path = job_manager.get_input_video_path(job_id)
        return FileResponse(
            path=str(path),
            media_type="video/mp4",
            filename="input.mp4",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Input video not found for job {job_id}"
        )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get current job status and progress."""
    try:
        job = job_manager.get_job(job_id)
        return JobStatus(
            job_id=job["job_id"],
            status=job["status"],
            progress=job["progress"],
            error=job.get("error"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@app.get("/jobs/{job_id}/results")
async def get_results(job_id: str):
    """Return the complete world_gt.json."""
    try:
        data = job_manager.load_artifact(job_id, "world_gt.json")
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Results not available yet or job not found"
        )


@app.get("/jobs/{job_id}/timeline")
async def get_timeline(job_id: str):
    """Return the timeline.json event sequence."""
    try:
        data = job_manager.load_artifact(job_id, "timeline.json")
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Timeline not available yet or job not found"
        )


@app.get("/jobs/{job_id}/confidence-report")
async def get_confidence_report(job_id: str):
    """Return the confidence_report.json."""
    try:
        data = job_manager.load_artifact(job_id, "confidence_report.json")
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Confidence report not available yet or job not found",
        )


@app.get("/jobs/{job_id}/annotated-video")
async def get_annotated_video(job_id: str):
    """Stream the annotated video file."""
    try:
        path = job_manager.get_artifact_path(job_id, "annotated_video.mp4")
        return FileResponse(
            path=str(path),
            media_type="video/mp4",
            filename="annotated_video.mp4",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Annotated video not available yet or job not found",
        )


@app.post("/jobs/{job_id}/corrections")
async def submit_corrections(job_id: str, payload: CorrectionPayload):
    """Submit human corrections for a job."""
    try:
        job_manager.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_manager.save_artifact(job_id, "corrections.json", payload)
    return {"status": "accepted", "corrections_count": len(payload.corrections)}


@app.post("/jobs/{job_id}/rerun")
async def rerun_job(job_id: str, mode: str = "calibrated"):
    """Rerun the pipeline with calibrated thresholds."""
    try:
        job = job_manager.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] not in ("completed", "failed"):
        raise HTTPException(
            status_code=400, detail="Job must be completed before rerun"
        )

    if not job_manager.artifact_exists(job_id, "corrections.json"):
        raise HTTPException(
            status_code=400, detail="No corrections submitted yet"
        )

    asyncio.create_task(
        asyncio.to_thread(_run_calibrated_rerun_sync, job_id)
    )

    return {"status": "rerunning"}


@app.get("/jobs/{job_id}/metrics")
async def get_metrics(job_id: str, run: str = "baseline"):
    """Get evaluation metrics for a pipeline run.

    Query params:
        run: "baseline" or "calibrated"
    """
    filename = f"metrics_{run}.json"
    try:
        data = job_manager.load_artifact(job_id, filename)
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics ({run}) not available yet or job not found",
        )


@app.get("/jobs/{job_id}/events")
async def get_events(job_id: str):
    """Return events, objects, and state_timeline from world_gt.json for the review UI and charts."""
    try:
        data = job_manager.load_artifact(job_id, "world_gt.json")
        return JSONResponse(content={
            "events": data.get("events", []),
            "objects": data.get("objects", []),
            "state_timeline": data.get("state_timeline", []),
        })
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Events not available yet or job not found",
        )


@app.get("/jobs/{job_id}/segmentations")
async def get_segmentations(job_id: str):
    """Return segmentations.json for chart 1 (Object Detection Confidence)."""
    try:
        data = job_manager.load_artifact(job_id, "segmentations.json")
        return JSONResponse(content=data if isinstance(data, list) else data.get("segmentations", []))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Segmentations not available yet or job not found",
        )


@app.get("/jobs/{job_id}/detections")
async def get_detections(job_id: str):
    """Return detections.json for chart 2 (Person Tracking)."""
    try:
        data = job_manager.load_artifact(job_id, "detections.json")
        return JSONResponse(content=data if isinstance(data, list) else data.get("detections", []))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Detections not available yet or job not found",
        )


@app.get("/jobs/{job_id}/semantics")
async def get_semantics(job_id: str):
    """Return semantics.json for chart 5 (VL Model Consistency)."""
    try:
        data = job_manager.load_artifact(job_id, "semantics.json")
        return JSONResponse(content=data if isinstance(data, list) else data.get("semantics", []))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Semantics not available yet or job not found",
        )


@app.post("/jobs/{job_id}/review")
async def review_job(job_id: str, payload: ReviewPayload):
    """Simple 3-button review: approve, correct, or reject."""
    try:
        job = job_manager.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] not in ("completed", "approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="Job must be completed before review"
        )

    if payload.action == ReviewAction.APPROVE:
        job_manager.mark_approved(job_id)
        return {"status": "approved"}

    if payload.action == ReviewAction.REJECT:
        job_manager.mark_rejected(job_id)
        return {"status": "rejected"}

    # CORRECT: convert rejected_events → corrections, save, and rerun
    if not payload.rejected_events:
        raise HTTPException(
            status_code=400,
            detail="rejected_events required for 'correct' action",
        )

    corrections = CorrectionPayload(
        corrections=[
            Correction(event_id=eid, action=CorrectionAction.REJECT)
            for eid in payload.rejected_events
        ]
    )
    job_manager.save_artifact(job_id, "corrections.json", corrections)

    asyncio.create_task(
        asyncio.to_thread(_run_calibrated_rerun_sync, job_id)
    )

    return {"status": "rerunning"}
