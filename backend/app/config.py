"""
RoboSight Configuration

Central configuration using Pydantic Settings. All calibratable thresholds
live here so they can be updated by the calibration engine at inference time.

Environment variables override defaults with ROBOSIGHT_ prefix:
    ROBOSIGHT_BACKEND=local
    ROBOSIGHT_SAMPLE_FPS=3.0
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with calibratable inference thresholds."""

    # --- Backend selection ---
    # "modal" = cloud GPU via Modal
    # "local" = local NVIDIA GPU (all models loaded simultaneously)
    # "local_vllm" = local GPU with vLLM for VL model acceleration
    backend: str = "modal"

    # --- Frame sampling ---
    sample_fps: float = 5.0
    keyframe_threshold: float = 30.0  # pixel diff threshold for keyframe detection

    # --- YOLO ---
    yolo_model: str = "yolov8m.pt"
    yolo_confidence: float = 0.5
    person_class_id: int = 0

    # --- SAM3 ---
    sam3_threshold: float = 0.5
    sam3_mask_threshold: float = 0.5

    # --- Object ontology (configurable per demo scene) ---
    object_prompts: list[str] = ["drawer", "handle", "door", "cabinet"]

    # --- VL model ---
    vl_max_tokens: int = 256

    # --- Inference thresholds (CALIBRATABLE) ---
    motion_threshold: float = 0.15       # IoU change needed to detect object motion
    proximity_threshold: float = 200.0   # pixels — max distance person↔object
    dwell_time_threshold: float = 0.5    # seconds — min time near object
    proximity_norm_factor: float = 500.0 # normalization for proximity score

    # --- Signal weights (CALIBRATABLE) ---
    motion_weight: float = 0.4
    proximity_weight: float = 0.3
    vl_weight: float = 0.3

    # --- Boundary offsets (CALIBRATABLE) ---
    start_offset: float = 0.0  # seconds — systematic start boundary correction
    end_offset: float = 0.0    # seconds — systematic end boundary correction

    # --- Confidence ---
    low_confidence_threshold: float = 0.7  # below this → needs human review

    # --- Paths ---
    jobs_dir: str = "jobs"
    data_dir: str = "data"

    # --- Execution ---
    parallel_models: bool = True  # True = run models in parallel, False = sequential

    model_config = {"env_prefix": "ROBOSIGHT_"}

    def get_calibratable_params(self) -> dict:
        """Return only the parameters that the calibration engine can update."""
        return {
            "motion_threshold": self.motion_threshold,
            "proximity_threshold": self.proximity_threshold,
            "dwell_time_threshold": self.dwell_time_threshold,
            "proximity_norm_factor": self.proximity_norm_factor,
            "motion_weight": self.motion_weight,
            "proximity_weight": self.proximity_weight,
            "vl_weight": self.vl_weight,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    def apply_calibration(self, calibrated: dict) -> "Settings":
        """Return a new Settings with calibrated parameters applied."""
        current = self.model_dump()
        current.update(calibrated)
        return Settings(**current)
