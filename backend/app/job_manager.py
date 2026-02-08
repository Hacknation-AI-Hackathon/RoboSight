"""
Job Manager

File-system based job lifecycle management. Each job gets a directory
under the configured jobs_dir with all artifacts stored as JSON files.

Directory layout per job:
    jobs/{job_id}/
        status.json
        input.mp4
        detections.json
        segmentations.json
        semantics.json
        world_gt.json
        timeline.json
        confidence_report.json
        annotated_video.mp4
        calibration.json
        metrics_baseline.json
        metrics_calibrated.json
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings


class JobManager:
    """Manages job lifecycle using the local filesystem."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.jobs_dir = Path(self.settings.jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _read_status(self, job_id: str) -> dict:
        path = self._status_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        with open(path, "r") as f:
            return json.load(f)

    def _write_status(self, job_id: str, status: dict) -> None:
        with open(self._status_path(job_id), "w") as f:
            json.dump(status, f, indent=2)

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------

    def create_job(self) -> str:
        """Create a new job with a unique ID and initialized status."""
        job_id = str(uuid.uuid4())
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "stage": "",
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_status(job_id, status)
        return job_id

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> dict:
        """Get full job status dict."""
        return self._read_status(job_id)

    def get_status(self, job_id: str) -> str:
        """Get current status string for a job."""
        return self._read_status(job_id)["status"]

    def update_progress(
        self, job_id: str, progress: float, stage: str = ""
    ) -> None:
        """Update job progress (0.0 to 1.0) and optionally the current stage."""
        status = self._read_status(job_id)
        status["progress"] = round(min(max(progress, 0.0), 1.0), 3)
        status["status"] = "processing"
        if stage:
            status["stage"] = stage
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_status(job_id, status)

    def mark_complete(self, job_id: str) -> None:
        """Mark a job as successfully completed."""
        status = self._read_status(job_id)
        status["status"] = "completed"
        status["progress"] = 1.0
        status["stage"] = "done"
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_status(job_id, status)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        status = self._read_status(job_id)
        status["status"] = "failed"
        status["error"] = error
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_status(job_id, status)

    def mark_rerunning(self, job_id: str) -> None:
        """Mark a job as rerunning after calibration."""
        status = self._read_status(job_id)
        status["status"] = "rerunning"
        status["progress"] = 0.0
        status["stage"] = "calibrated_rerun"
        status["error"] = None
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_status(job_id, status)

    # ------------------------------------------------------------------
    # Artifact I/O
    # ------------------------------------------------------------------

    def save_artifact(self, job_id: str, name: str, data: Any) -> Path:
        """Save a JSON-serializable artifact to the job directory.

        Args:
            job_id: The job identifier.
            name: Artifact filename (e.g. "world_gt.json", "detections.json").
            data: JSON-serializable data (dict, list, or Pydantic model).

        Returns:
            Path to the saved artifact file.
        """
        path = self._job_dir(job_id) / name
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load_artifact(self, job_id: str, name: str) -> Any:
        """Load a JSON artifact from the job directory.

        Args:
            job_id: The job identifier.
            name: Artifact filename (e.g. "world_gt.json").

        Returns:
            Parsed JSON data.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        path = self._job_dir(job_id) / name
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {name} for job {job_id}")
        with open(path, "r") as f:
            return json.load(f)

    def artifact_exists(self, job_id: str, name: str) -> bool:
        """Check if an artifact file exists for a job."""
        return (self._job_dir(job_id) / name).exists()

    def get_artifact_path(self, job_id: str, name: str) -> Path:
        """Get the filesystem path for an artifact (for file streaming).

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        path = self._job_dir(job_id) / name
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {name} for job {job_id}")
        return path

    # ------------------------------------------------------------------
    # Video file management
    # ------------------------------------------------------------------

    def save_input_video(self, job_id: str, video_bytes: bytes) -> Path:
        """Save the uploaded video to the job directory.

        Returns:
            Path to the saved video file.
        """
        path = self._job_dir(job_id) / "input.mp4"
        with open(path, "wb") as f:
            f.write(video_bytes)
        return path

    def get_input_video_path(self, job_id: str) -> Path:
        """Get path to the input video for a job.

        Raises:
            FileNotFoundError: If no input video exists.
        """
        path = self._job_dir(job_id) / "input.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Input video not found for job {job_id}")
        return path

    # ------------------------------------------------------------------
    # Job listing and cleanup
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        """List all jobs with their current status."""
        jobs = []
        if not self.jobs_dir.exists():
            return jobs
        for job_dir in sorted(self.jobs_dir.iterdir()):
            status_path = job_dir / "status.json"
            if status_path.exists():
                with open(status_path, "r") as f:
                    jobs.append(json.load(f))
        return jobs

    def delete_job(self, job_id: str) -> None:
        """Delete a job and all its artifacts."""
        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)
