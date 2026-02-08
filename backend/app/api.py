"""
RoboSight FastAPI application.

Endpoints:
  POST /jobs         — Create job, upload video (multipart/form-data), returns job_id.
  GET  /jobs/{id}   — Job status for polling (job_id, status, progress, error).
  GET  /jobs/{id}/annotated-video — Stream annotated_video.mp4 (or input.mp4 if not ready).
"""

import shutil
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import Settings
from app.job_manager import JobManager
from app.models import JobStatus

app = FastAPI(title="RoboSight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = Settings()
_job_manager = JobManager(_settings)


def _run_demo_job(job_id: str) -> None:
    """Background: mark job complete and copy input to annotated_video for demo playback."""
    import time
    time.sleep(4)  # Simulate processing
    try:
        job_dir = _job_manager._job_dir(job_id)
        input_path = job_dir / "input.mp4"
        out_path = job_dir / "annotated_video.mp4"
        if input_path.exists():
            shutil.copy(input_path, out_path)
        _job_manager.mark_complete(job_id)
    except Exception:
        _job_manager.mark_failed(job_id, "Demo processing failed")


@app.post("/jobs", response_model=dict)
async def create_job(file: UploadFile = File(...)):
    """Create a new job and save the uploaded video. Returns job_id for polling."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")):
        raise HTTPException(status_code=400, detail="Video file required (mp4, mov, webm, mkv, m4v, avi)")
    job_id = _job_manager.create_job()
    try:
        contents = await file.read()
        _job_manager.save_input_video(job_id, contents)
    except Exception as e:
        _job_manager.mark_failed(job_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to save video")
    # Start background "processing" so status eventually becomes completed (demo)
    t = Thread(target=_run_demo_job, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get current job status for polling. Poll every 2s until status is 'completed' or 'failed'."""
    try:
        data = _job_manager.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=data["job_id"],
        status=data["status"],
        progress=data.get("progress", 0.0),
        error=data.get("error"),
    )


@app.get("/jobs/{job_id}/annotated-video")
def get_annotated_video(job_id: str):
    """Stream the annotated video (or input video if annotated not ready)."""
    try:
        if _job_manager.artifact_exists(job_id, "annotated_video.mp4"):
            path = _job_manager.get_artifact_path(job_id, "annotated_video.mp4")
        else:
            path = _job_manager.get_input_video_path(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="annotated_video.mp4",
    )
