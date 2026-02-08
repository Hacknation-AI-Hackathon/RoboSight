# RoboSight

An AI pipeline that converts raw video into structured, temporal ground truth for humanoid robot navigation. Three vision models (YOLO, SAM3, LFM2.5-VL) run in parallel on cloud or local GPU to detect objects, segment scenes, and understand actions — then a temporal compiler fuses their signals into a calibratable world model.

## How It Works

```
Raw Video
  |
  v
Frame Sampling (5 FPS + keyframe detection)
  |
  v
+-------------------------------------------+
|  YOLO          SAM3          LFM2.5-VL    |  <- 3 models in parallel
|  (persons,     (objects,     (actions,     |     Modal cloud GPU or
|   every frame)  keyframes)   keyframes)   |     local NVIDIA GPU
+-------------------------------------------+
  |
  v
Temporal Compiler (state machine + signal fusion)
  |
  v
+-------------------------------------------+
|  world_gt.json    annotated_video.mp4     |
|  timeline.json    confidence_report.json  |
+-------------------------------------------+
  |
  v
Human Review + Corrections (natural language)
  |
  v
Calibration Engine (threshold + weight tuning)
  |
  v
Improved Results (no GPU reprocessing needed)
```

## Key Features

- **Three-model fusion**: YOLO for person detection, SAM3 for object segmentation, LFM2.5-VL for semantic understanding — fused via weighted signal combination
- **Dual GPU backend**: Switch between Modal cloud GPU (T4/A100) and local NVIDIA GPU via config
- **Temporal state machine**: Tracks object states (open/closed) over time using motion, proximity, and VL signals with configurable thresholds
- **Human-in-the-loop calibration**: Submit corrections in natural language ("event 2 ends at 9.9s"), system recalibrates thresholds and reruns inference using cached GPU outputs — no re-processing
- **Apple Vision Pro UI**: Immersive frontend with drag-and-drop upload, Vision Pro mask playback, annotated video review, and interactive analysis dashboard

## Project Structure

```
RoboSight/
├── backend/
│   ├── app/
│   │   ├── api.py                  # FastAPI routes + CORS
│   │   ├── config.py               # Pydantic Settings (all thresholds)
│   │   ├── models.py               # Pydantic data contracts
│   │   ├── job_manager.py          # File-system job lifecycle
│   │   ├── pipeline/
│   │   │   ├── ingest.py           # Video decode + frame sampling
│   │   │   ├── orchestrator.py     # Parallel model execution
│   │   │   ├── detect.py           # YOLO person detection wrapper
│   │   │   ├── segment.py          # SAM3 object segmentation wrapper
│   │   │   ├── semantics.py        # LFM2.5-VL action understanding
│   │   │   ├── infer.py            # Temporal compiler (state machine)
│   │   │   ├── calibrate.py        # Threshold tuning from corrections
│   │   │   ├── compile.py          # Output JSON assembly
│   │   │   ├── annotate.py         # Video overlay rendering (H.264)
│   │   │   ├── eval.py             # Metrics (precision, recall, boundary error)
│   │   │   └── convert.py          # Video format conversion
│   │   ├── modal_app/
│   │   │   └── gpu_models.py       # 3 Modal GPU servers (YOLO, SAM3, VL)
│   │   └── local_app/
│   │       └── gpu_models.py       # 3 local GPU servers (same interface)
│   ├── scripts/
│   │   ├── run_demo.py             # End-to-end demo
│   │   └── test_models.py          # Model integration tests
│   ├── tests/
│   │   └── test_e2e_pipeline.py    # Pipeline tests with mock data
│   ├── requirements.txt            # Base dependencies
│   └── requirements-local-gpu.txt  # Additional deps for local GPU
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main application component
│   │   └── App.css                 # Full styling (Vision Pro theme)
│   ├── public/                     # Vision Pro assets, transition video
│   ├── package.json
│   └── vite.config.js              # Dev proxy: /api -> localhost:8000
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- FFmpeg (for video processing)
- Modal account (for cloud GPU) **or** NVIDIA GPU with CUDA (for local)

### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For local GPU mode (optional):
pip install -r requirements-local-gpu.txt

# Start the API server
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies /api -> backend)
npm run dev
```

The frontend runs at `http://localhost:5173` and proxies API calls to the backend on port 8000.

### Modal Cloud GPU Setup (Optional)

```bash
# Authenticate with Modal
modal token new

# Deploy GPU model servers
modal deploy backend/app/modal_app/gpu_models.py
```

### Configuration

All settings are controlled via environment variables with the `ROBOSIGHT_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOSIGHT_BACKEND` | `modal` | `modal`, `local`, or `local_vllm` |
| `ROBOSIGHT_SAMPLE_FPS` | `5.0` | Frame sampling rate |
| `ROBOSIGHT_KEYFRAME_THRESHOLD` | `30.0` | Pixel diff threshold for keyframes |
| `ROBOSIGHT_MOTION_THRESHOLD` | `0.15` | IoU change to detect object motion |
| `ROBOSIGHT_PROXIMITY_THRESHOLD` | `200.0` | Max person-object distance (px) |
| `ROBOSIGHT_PARALLEL_MODELS` | `true` | Run 3 models in parallel |
| `ROBOSIGHT_OBJECT_PROMPTS` | `drawer,handle,door,cabinet` | Objects to detect |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs` | Upload video (multipart), starts pipeline |
| `GET` | `/jobs/{id}` | Job status + progress (0.0–1.0) |
| `GET` | `/jobs/{id}/input-video` | Converted input video (MP4) |
| `GET` | `/jobs/{id}/annotated-video` | Annotated video with overlays |
| `GET` | `/jobs/{id}/results` | Full `world_gt.json` |
| `GET` | `/jobs/{id}/timeline` | Event timeline |
| `GET` | `/jobs/{id}/confidence-report` | Per-event confidence scores |
| `GET` | `/jobs/{id}/metrics?run=baseline` | Evaluation metrics |
| `GET` | `/jobs/{id}/events` | Detected events + state timeline |
| `GET` | `/jobs/{id}/detections` | Raw YOLO detections |
| `GET` | `/jobs/{id}/segmentations` | Raw SAM3 segmentations |
| `GET` | `/jobs/{id}/semantics` | Raw VL semantic outputs |
| `GET` | `/jobs/{id}/corrections` | Applied corrections |
| `GET` | `/jobs/{id}/calibration` | Learned calibration parameters |
| `POST` | `/jobs/{id}/corrections` | Submit corrections JSON |
| `POST` | `/jobs/{id}/rerun` | Rerun inference with calibration |

## Pipeline Details

### Signal Fusion

The temporal compiler fuses three signals per keyframe to determine object state transitions:

| Signal | Source | Measures |
|--------|--------|----------|
| **Motion score** | SAM3 bbox IoU between frames | Physical movement of object |
| **Proximity score** | YOLO person + SAM3 object | How close the person is |
| **VL confidence** | LFM2.5-VL semantic output | Language model's state assessment |

These are combined using calibratable weights (default: motion=0.4, proximity=0.3, VL=0.3) to trigger state transitions in the state machine:

```
closed -> open:    motion > threshold AND proximity > threshold AND VL confirms
open   -> closed:  motion > threshold AND proximity > threshold AND VL confirms
```

### Calibration Loop

When a human submits corrections, the system:

1. **Adjusts thresholds** — boundary corrections shift motion/proximity thresholds
2. **Rebalances weights** — if motion was wrong but VL was right, VL weight increases
3. **Learns offsets** — systematic start/end boundary bias correction
4. **Reruns inference** — uses cached GPU outputs (fast, CPU-only)
5. **Computes new metrics** — compares calibrated vs baseline

Supported correction types:
- `reject` — remove a false positive event
- `adjust_boundary` — fix event start/end times
- `add_event` — add a missed event

### Job Storage

Each job is a directory under `jobs/{job_id}/`:

```
jobs/{job_id}/
  input.mp4                  # Uploaded video
  detections.json            # Cached YOLO outputs
  segmentations.json         # Cached SAM3 outputs
  semantics.json             # Cached VL outputs
  world_gt.json              # Compiled ground truth
  timeline.json              # Event timeline
  confidence_report.json     # Per-event confidence
  annotated_video.mp4        # Video with overlays
  corrections.json           # Human corrections
  calibration.json           # Learned parameters
  metrics_baseline.json      # Zero-shot metrics
  metrics_calibrated.json    # Post-calibration metrics
```

## Frontend

The UI is built with React + Vite and styled after Apple Vision Pro:

1. **Landing page** — "ROBO SIGHT" hero text with drag-and-drop video upload
2. **Processing view** — Video plays inside a Vision Pro headset mask with AI analysis wave animation
3. **Result view** — Annotated video as main stage with draggable PiP of original video; actions menu for approve/correct/reject
4. **Analysis dashboard** — Split layout with:
   - **Left column**: Annotated video, detection confidence chart, person tracking, signal fusion per event, state timeline (with hover tooltips), VL consistency table
   - **Right column**: Efficiency summary, zero-shot vs calibrated comparison, baseline vs calibrated bar chart, corrections applied, learned parameters, raw JSON viewer

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Vite 5, Recharts |
| Backend | FastAPI, Uvicorn, Pydantic |
| Models | YOLO (ultralytics), SAM3 (HF transformers), LFM2.5-VL |
| Cloud GPU | Modal (T4/A100 containers) |
| Video | OpenCV, imageio-ffmpeg, H.264 encoding |
| Storage | File-system (no database) |
