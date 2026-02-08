# RoboSight Backend — Full Implementation Plan

> **One-liner**: A backend pipeline that converts raw human video into structured, temporal ground truth for humanoid navigation by composing YOLO, SAM3, and LFM2.5-VL (on Modal cloud GPU or local NVIDIA GPU), with inference-time calibration from minimal human feedback.

---

## Architecture

Supports **two backends** — same pipeline, same outputs, switchable via config:

### Option A: Modal Cloud GPU
```
Raw Video
  ↓
Frame Sampling (ingest.py)
  ↓
┌────────── Modal Cloud GPU (A100/T4) ──────────┐
│  YOLO (detect.py)      — persons, every frame  │
│  SAM3 (segment.py)     — objects, keyframes     │  ← All 3 run IN PARALLEL
│  LFM2.5-VL (semantics) — actions, keyframes    │     on separate containers
└────────────────────────────────────────────────┘
  ↓
```

### Option B: Local NVIDIA GPU (e.g. RTX 5090 16GB)
```
Raw Video
  ↓
Frame Sampling (ingest.py)
  ↓
┌────────── Local NVIDIA GPU ───────────────────┐
│  YOLO (~300MB)   — ultralytics                 │
│  SAM3 (~2-3GB)   — HF transformers             │  ← All 3 loaded
│  LFM2.5-VL (~3-4GB) — HF or vLLM accelerated  │     simultaneously (~8GB)
└────────────────────────────────────────────────┘
  ↓
```

### Common pipeline (both backends):
```
Model Outputs (detections + segmentations + semantics)
  ↓
orchestrator.py            — parallel execution + output merging
  ↓
Temporal Ground-Truth Compiler (infer.py)      ← Dev B — THE CORE LOGIC
  ↓
Calibration Engine (calibrate.py)              ← Dev B
  ↓
Output Assembly (compile.py)                   ← Dev B
  ↓
Annotated Video (annotate.py)                  ← Dev A
  ↓
Evaluation Metrics (eval.py)                   ← Dev B
```

**FastAPI + Job Manager + API routes** → Dev B

---

## File Structure

```
backend/
├── requirements.txt              # Base dependencies (FastAPI, cv2, etc.)
├── requirements-local-gpu.txt    # Additional deps for local GPU (torch, transformers, ultralytics)
├── IMPLEMENTATION_PLAN.md
├── app/
│   ├── __init__.py
│   ├── api.py                    # FastAPI routes + pipeline orchestrator     [Dev B]
│   ├── config.py                 # Settings + calibratable thresholds         [DONE]
│   ├── models.py                 # Pydantic models (data contracts)           [Dev B]
│   ├── job_manager.py            # In-memory job lifecycle                    [Dev B]
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Parallel model execution + merging         [DONE]
│   │   ├── ingest.py             # Video decode + frame sampling              [DONE]
│   │   ├── detect.py             # YOLO wrapper (modal + local backends)      [DONE]
│   │   ├── segment.py            # SAM3 wrapper (modal + local backends)      [DONE]
│   │   ├── semantics.py          # LFM2.5-VL wrapper (modal + local + vllm)  [DONE]
│   │   ├── infer.py              # State machine + event detection            [Dev B]
│   │   ├── calibrate.py          # Threshold calibration from corrections     [Dev B]
│   │   ├── compile.py            # world_gt.json assembly                     [Dev B]
│   │   ├── annotate.py           # Video overlay (bboxes + labels)            [DONE]
│   │   └── eval.py               # Metrics computation                        [Dev B]
│   ├── modal_app/
│   │   ├── __init__.py
│   │   └── gpu_models.py         # 3 Modal GPU servers (YOLO,SAM3,VL)        [DONE]
│   ├── local_app/
│   │   ├── __init__.py
│   │   └── gpu_models.py         # 3 Local GPU servers (same interface)       [DONE]
│   └── schemas/
│       ├── ontology_v0_1.json    # Object ontology                            [Dev B]
│       └── world_gt.schema.json
├── data/                         # Demo videos go here
├── jobs/                         # Job output directory (auto-created)
└── scripts/
    └── run_demo.py               # End-to-end demo script
```

---

## Task Delegation

### Dev A: MODEL INTEGRATION + VIDEO I/O

| Priority | File | What |
|----------|------|------|
| 1 | `modal_app/gpu_models.py` | Three Modal classes: YOLOServer, SAM3Server, VLServer |
| 2 | `pipeline/ingest.py` | Video decode, frame sampling at 5 FPS, keyframe detection |
| 3 | `pipeline/detect.py` | YOLO wrapper calling Modal, batch processing |
| 4 | `pipeline/segment.py` | SAM3 wrapper calling Modal, text-prompted segmentation |
| 5 | `pipeline/semantics.py` | LFM2.5-VL wrapper calling Modal, forced JSON schema |
| 6 | `pipeline/annotate.py` | Overlay bboxes + state labels + event banners on video |
| 7 | Integration testing | End-to-end with real video |

### Dev B: LOGIC + API LAYER

| Priority | File | What |
|----------|------|------|
| 1 | `config.py` | Pydantic Settings with all thresholds |
| 2 | `models.py` | All Pydantic models (data contracts between modules) |
| 3 | `job_manager.py` | Job creation, status tracking, directory management |
| 4 | `api.py` | FastAPI routes + pipeline orchestrator + CORS |
| 5 | `pipeline/infer.py` | Temporal compiler: object tracking, signals, state machine, events |
| 6 | `pipeline/calibrate.py` | Threshold/weight updates from corrections |
| 7 | `pipeline/compile.py` | world_gt.json + timeline.json + confidence_report.json |
| 8 | `pipeline/eval.py` | Metrics: precision, recall, boundary error |
| 9 | `schemas/ontology_v0_1.json` | Object ontology |

**Why this split works:**
- Dev A needs Modal/GPU — tests require deployed models
- Dev B needs zero GPU — pure Python logic, testable with mock data
- They connect at 3 clear interfaces (see below)

---

## Data Contracts (Interface Between Dev A & Dev B)

Dev B can build `infer.py` with mock data immediately. These are the exact formats:

### Interface 1: `detections` (detect.py → infer.py)
```json
[
  {
    "frame_index": 0,
    "timestamp": 0.0,
    "person_boxes": [[120.5, 200.0, 350.2, 680.0]],
    "confidences": [0.95]
  }
]
```
One entry per sampled frame. `person_boxes` can be empty if no person detected.

### Interface 2: `segmentations` (segment.py → infer.py)
```json
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
  }
]
```
One entry per **keyframe only**. Multiple objects per frame.

### Interface 3: `semantics` (semantics.py → infer.py)
```json
[
  {
    "frame_index": 5,
    "timestamp": 1.0,
    "objects": [
      {"label": "drawer", "state": "open", "confidence": 0.9}
    ],
    "action": "person_opening_drawer",
    "description": "Person reaching toward drawer handle",
    "raw_response": "..."
  }
]
```
One entry per **keyframe only**. Matches keyframes from segmentations.

---

## Output Schemas

### world_gt.json
```json
{
  "version": "0.1",
  "video": {
    "source": "demo.mp4",
    "duration_seconds": 30.0,
    "fps": 30,
    "resolution": [1920, 1080]
  },
  "objects": [
    {
      "id": "obj_1",
      "label": "drawer",
      "category": "interactable",
      "affordances": ["openable", "closable", "graspable"],
      "initial_state": "closed",
      "bbox_initial": [400, 300, 600, 450],
      "confidence": 0.92
    }
  ],
  "agents": [
    {
      "id": "agent_1",
      "label": "person",
      "track": [
        {"time": 0.0, "bbox": [120, 200, 350, 680]},
        {"time": 0.2, "bbox": [125, 202, 355, 682]}
      ]
    }
  ],
  "state_timeline": [
    {
      "object_id": "obj_1",
      "states": [
        {"state": "closed", "start": 0.0, "end": 3.2},
        {"state": "open", "start": 3.2, "end": 8.5},
        {"state": "closed", "start": 8.5, "end": 30.0}
      ]
    }
  ],
  "events": [
    {
      "id": "evt_1",
      "type": "open_drawer",
      "agent_id": "agent_1",
      "object_id": "obj_1",
      "start_time": 2.8,
      "end_time": 3.5,
      "confidence": 0.87,
      "signals": {
        "motion_score": 0.82,
        "proximity_score": 0.95,
        "vl_confidence": 0.84
      }
    }
  ],
  "affordance_map": {
    "obj_1": {
      "interactable": true,
      "movable": false,
      "traversable": false,
      "states": ["open", "closed"]
    }
  }
}
```

### Corrections payload (from frontend → POST /jobs/{id}/corrections)
```json
{
  "corrections": [
    {"event_id": "evt_1", "action": "adjust_boundary", "corrected_start": 2.9, "corrected_end": 3.4},
    {"event_id": "evt_3", "action": "reject"},
    {"event_id": null, "action": "add_event", "type": "reach_into_drawer", "start_time": 4.0, "end_time": 4.8}
  ]
}
```

---

## API Endpoints (Frontend Contract)

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| POST | `/jobs` | `multipart/form-data` with `file` field + optional `object_prompts` | `{job_id, status: "processing"}` |
| GET | `/jobs/{id}` | — | `{job_id, status, progress: 0.0-1.0, error}` |
| GET | `/jobs/{id}/results` | — | Full `world_gt.json` as JSON |
| GET | `/jobs/{id}/timeline` | — | `timeline.json` as JSON |
| GET | `/jobs/{id}/confidence-report` | — | `confidence_report.json` as JSON |
| GET | `/jobs/{id}/annotated-video` | — | `annotated_video.mp4` as file stream |
| POST | `/jobs/{id}/corrections` | Corrections JSON | `{status: "accepted"}` |
| POST | `/jobs/{id}/rerun` | `?mode=calibrated` | `{status: "rerunning"}` (poll for completion) |
| GET | `/jobs/{id}/metrics` | `?run=baseline` or `?run=calibrated` | Metrics JSON |

CORS enabled for frontend.

---

## Technical Details

### Dual Backend System
All pipeline wrappers (`detect.py`, `segment.py`, `semantics.py`) accept a `backend` parameter:
- `"modal"` → calls Modal cloud GPU containers via `.remote()`
- `"local"` → calls local NVIDIA GPU via singleton model servers
- `"local_vllm"` → same as local but uses vLLM for VL model acceleration

Set in `config.py` via `ROBOSIGHT_BACKEND=local` environment variable.

The `orchestrator.py` runs all 3 models **in parallel** using `ThreadPoolExecutor`:
- Modal: each thread calls a different remote container (true parallelism)
- Local: threads share one GPU but overlap I/O with compute

### Modal Setup (backend=modal)
- Single Modal App `robosight-gpu` with 3 classes
- YOLO on T4, SAM3 + VL on A100
- Models loaded once via `@modal.enter()`, persisted across calls
- Local code calls `.remote()` — no web endpoints needed
- Frame transfer: JPEG bytes (~100-200KB per frame)

### Local GPU Setup (backend=local)
- All 3 models loaded simultaneously as singletons (~6-8GB VRAM)
- YOLO: ~300MB (ultralytics)
- SAM3: ~2-3GB (HF transformers)
- LFM2.5-VL: ~3-4GB (bfloat16)
- Fits in 16GB with room to spare
- Optional vLLM acceleration for VL model (PagedAttention, continuous batching)
- Install: `pip install -r requirements-local-gpu.txt`

### Frame Sampling Strategy
- Sample at 5 FPS from source video
- Keyframe detection: mean absolute pixel difference > threshold (30.0)
- First frame is always a keyframe
- For a 30s video at 30fps: ~150 sampled frames, ~15-30 keyframes

### State Machine (infer.py core logic)
```
States: closed → opening → open → closing → closed
Signals per keyframe pair:
  - motion_score: 1 - IoU(object_bbox_t, object_bbox_t-1)
  - proximity_score: 1 / (1 + distance(person_center, object_center) / norm)
  - dwell_time: accumulated seconds person within proximity_threshold
  - vl_state: state string from VL model
  - vl_confidence: confidence from VL model

Transition rules:
  closed → open:   motion > threshold AND proximity > threshold AND vl confirms
  open → closed:   motion > threshold AND proximity > threshold AND vl confirms

Constraints:
  - Cannot transition to same state
  - States persist unless multi-signal evidence
  - Motion alone without proximity does NOT trigger change
```

### Calibration (calibrate.py)
Updates at inference time — NO retraining:
- **Thresholds**: motion, proximity, dwell_time (adjusted from correction boundary signals)
- **Weights**: how much each signal matters (adjusted from which signals were wrong)
- **Boundary offsets**: systematic start/end bias correction
- **Language constraints**: VL-derived rules (e.g., "slight adjustment ≠ open")

Rerun = same video + same cached model outputs + new thresholds → improved results

### Job Storage
File-system based, no database:
```
jobs/{job_id}/
  input.mp4
  detections.json       # cached for rerun
  segmentations.json    # cached for rerun
  semantics.json        # cached for rerun
  world_gt.json
  timeline.json
  confidence_report.json
  annotated_video.mp4
  calibration.json
  metrics_baseline.json
  metrics_calibrated.json
```

---

## Implementation Timeline

### Phase 0: Project Setup (before anything else)

1. Create `.gitignore` at repo root with `.claude/`, `__pycache__/`, `*.pyc`, `.env`, `jobs/`, `venv/`, etc.
2. Save this implementation plan as `backend/IMPLEMENTATION_PLAN.md` so teammate can read it
3. Create all directories in the file structure

### Phase 1: Foundation (parallel, ~2 hours)

**Dev A:**
1. Set up directory structure + `requirements.txt`
2. Write `modal_app/gpu_models.py` (all 3 servers)
3. Write `pipeline/ingest.py` (frame sampling)
4. Deploy Modal: `modal deploy app/modal_app/gpu_models.py`
5. Test YOLO server with a sample image

**Dev B:**
1. Write `config.py`, `models.py`, `job_manager.py`
2. Write `api.py` skeleton (all routes stubbed)
3. Write `schemas/ontology_v0_1.json`
4. Create mock data files matching the 3 interfaces
5. Start `pipeline/infer.py` with mock data

### Phase 2: Core Pipeline (parallel, ~3 hours)

**Dev A:**
1. Write `pipeline/detect.py`, `pipeline/segment.py`, `pipeline/semantics.py`
2. Test each wrapper individually
3. Test: video → frames → all 3 model outputs

**Dev B:**
1. Complete `pipeline/infer.py` (full state machine + events)
2. Write `pipeline/calibrate.py`
3. Write `pipeline/compile.py`
4. Write `pipeline/eval.py`
5. Wire up API endpoints

### Phase 3: Integration (~2 hours)

1. Connect real model outputs to compiler (validate data formats)
2. Wire up full pipeline orchestrator in `api.py`
3. Dev A: write `pipeline/annotate.py`
4. End-to-end test: POST /jobs → poll → get results
5. Test corrections + rerun flow

### Phase 4: Polish (~1 hour)

1. Run on real demo video
2. Verify all outputs
3. Baseline vs calibrated metrics comparison
4. Edge case fixes

---

## Verification Plan

1. `modal deploy backend/app/modal_app/gpu_models.py` succeeds
2. Each Modal class responds to `.remote()` calls
3. `uvicorn app.api:app` starts without errors
4. `POST /jobs` with a test video returns job_id
5. `GET /jobs/{id}` shows progress advancing to `completed`
6. `GET /jobs/{id}/results` returns valid `world_gt.json` with objects, events, state_timeline
7. `GET /jobs/{id}/annotated-video` returns playable video with overlays
8. `POST /jobs/{id}/corrections` followed by `POST /jobs/{id}/rerun?mode=calibrated`
9. `GET /jobs/{id}/metrics?run=calibrated` shows improvement over baseline
10. Frontend can call all endpoints with CORS
