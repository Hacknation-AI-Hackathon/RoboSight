# Backend Integration Plan — Dev A + Dev B

## Current State

**Dev A** (you) built the model infrastructure layer:
- `modal_app/gpu_models.py` — YOLO, SAM3, VL model servers on Modal (tested, working)
- `local_app/gpu_models.py` — Local GPU equivalents
- `pipeline/ingest.py` — Frame sampling + keyframe detection
- `pipeline/detect.py`, `segment.py`, `semantics.py` — Model wrappers
- `pipeline/orchestrator.py` — Parallel model execution
- `pipeline/annotate.py` — Video overlay rendering
- `scripts/test_models.py` — Testing script (produced results in `jobs/`)

**Dev B** (teammate) built the logic + API layer:
- `models.py` — Pydantic data contracts
- `config.py` — Calibratable settings
- `api.py` — FastAPI routes + pipeline orchestration
- `job_manager.py` — File-system job lifecycle
- `pipeline/infer.py` — Temporal compiler (state machine, event detection)
- `pipeline/calibrate.py` — Threshold tuning from corrections
- `pipeline/compile.py` — Output JSON assembly
- `pipeline/eval.py` — Metrics computation
- `pipeline/convert.py` — Video format conversion

**Uncommitted changes**: `gpu_models.py` has battle-tested fixes from Modal testing (apt deps, torchvision, T4 GPUs, HF token auth, batch detection fix).

---

## Issues to Fix

### Issue 1: VL model output doesn't match the contract (CRITICAL)

**Problem:** The VL model returns objects with `{label, state, action}` but the Pydantic contract (`SemanticObject`) and `infer.py` expect `{label, state, confidence}`.

- **Actual VL output** (from test results): `{"label": "drawer", "state": "open", "action": "being opened"}`
- **Expected by contract**: `{"label": "drawer", "state": "open", "confidence": 0.9}`
- `infer.py` line 183: `conf = obj.get("confidence", 0.0)` — silently gets 0.0 always
- The VL model also doesn't return the top-level `action` or `description` fields in its JSON — they're only in the system prompt schema but the model ignores them

**Fix:**
- In `semantics.py` `_parse_vl_json()`: after parsing, normalize each object — add default `confidence: 0.8` if missing, and extract the top-level `action` from the response
- Alternatively, improve the VL system prompt in `gpu_models.py` to actually enforce confidence scores
- Update `SemanticObject` model to have `confidence: float = 0.0` as default (defensive)

### Issue 2: VL label mismatch with SAM3 labels (CRITICAL)

**Problem:** SAM3 segments objects using user-provided prompts (e.g., `["drawer", "handle"]`), but VL independently identifies objects using different labels:
- SAM3 sees: `"drawer"`, `"handle"`
- VL sees: `"dresser"`, `"sofa"`, `"couch"`, `"furniture"`, `"person"`

`infer.py` tries to match VL labels to SAM3 labels (line 181: `if label == object_label`). This **never matches** because VL calls the drawer a "dresser".

**Fix:**
- Add a label normalization/alias mapping in `infer.py` or `semantics.py`
- Map common VL synonyms: `dresser → drawer`, `couch/sofa → ignore`, etc.
- OR: pass the SAM3 object prompts to the VL system prompt so it uses consistent labels

### Issue 3: VL doesn't extract top-level `action` properly (MEDIUM)

**Problem:** The VL model's JSON response only contains `objects` array. It doesn't return `action` or `description` at the top level. `semantics.py` falls back to `"no_action"` and `""` for every frame.

**Fix:**
- Post-process in `semantics.py`: infer the top-level action from object-level `action` fields
- e.g., if any object has `action: "being opened"` and `label: "drawer"`, set frame `action: "person_opening_drawer"`
- Or update the VL system prompt to be more explicit about returning all fields

### Issue 4: Commit gpu_models.py fixes (SMALL)

Your Modal deployment fixes need to be committed:
- `apt_install` for system deps
- `torchvision` added to SAM3/VL images
- `container_idle_timeout` → `scaledown_window`
- GPU downgrade A100 → T4
- HuggingFace token auth
- Batch detection refactor (avoid self-calling `.remote()`)

---

## Implementation Steps

### Step 1: Commit current gpu_models.py fixes
- Stage and commit the working Modal deployment fixes
- Stage `scripts/test_models.py`

### Step 2: Fix VL output normalization in `semantics.py`
- After `_parse_vl_json()`, normalize each object:
  - Ensure `confidence` field exists (default 0.8 if model didn't provide it)
  - Strip the extra `action` field from objects (not in the contract)
- Infer top-level `action` from object actions when model doesn't provide it
- Infer `description` from raw response context

### Step 3: Fix label alignment between VL and SAM3
- Add label alias map in `infer.py` → `_normalize_label()`
- Map: `dresser→drawer`, `cabinet→drawer`, `cupboard→drawer`, `knob→handle`, `pull→handle`, `door_handle→handle`
- Apply normalization in `extract_vl_signals()` before label matching
- Also normalize in `_discover_objects()` so SAM3 and VL labels align

### Step 4: Update Pydantic models for defensive defaults
- `SemanticObject.confidence` → `float = 0.0` (already has no default, should be optional)
- Verify all model fields have sensible defaults for partial data

### Step 5: Add validation in orchestrator
- After running models, validate outputs against Pydantic models
- Log warnings for any validation failures instead of crashing
- Ensure graceful degradation (empty detections shouldn't crash inference)

### Step 6: End-to-end API test
- Run the full pipeline through `api.py` (not just test_models.py)
- POST a video to `/jobs`, wait for completion, GET results
- Verify `world_gt.json`, `timeline.json`, `confidence_report.json` are generated
- Verify annotated video is produced

### Step 7: Verify calibration loop
- Test the corrections → calibrate → rerun flow
- POST corrections to `/jobs/{id}/corrections`
- POST rerun to `/jobs/{id}/rerun`
- Verify calibrated results differ from baseline
