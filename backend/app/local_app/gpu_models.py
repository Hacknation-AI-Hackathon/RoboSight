"""
RoboSight Local GPU Model Servers

Same interface as modal_app/gpu_models.py but runs directly on a local
NVIDIA GPU (e.g. RTX 5090 16GB VRAM).

All three models stay loaded in memory simultaneously (~6-8GB total):
  - YOLO:      ~300MB
  - SAM3:      ~2-3GB
  - LFM2.5-VL: ~3-4GB (bfloat16)
  Total:       ~6-8GB → fits in 16GB with room to spare

Usage:
    from app.local_app.gpu_models import LocalYOLOServer, LocalSAM3Server, LocalVLServer

    yolo = LocalYOLOServer()
    result = yolo.detect_persons(frame_bytes)

    sam3 = LocalSAM3Server()
    objects = sam3.segment_objects(frame_bytes, ["drawer", "handle"])

    vl = LocalVLServer()
    response = vl.analyze_scene(frame_bytes)
"""

import io
import torch
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Singleton pattern — models load once and stay in memory
# ---------------------------------------------------------------------------
_instances: dict[str, object] = {}


def get_server(server_class):
    """Get or create a singleton instance of a model server."""
    name = server_class.__name__
    if name not in _instances:
        print(f"[LocalGPU] Initializing {name}...")
        _instances[name] = server_class()
    return _instances[name]


def unload_all():
    """Unload all models and free GPU memory."""
    global _instances
    _instances.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[LocalGPU] All models unloaded, VRAM freed")


# ---------------------------------------------------------------------------
# 1. Local YOLO Server — Person Detection
# ---------------------------------------------------------------------------
class LocalYOLOServer:
    """Detects persons in video frames using YOLOv8m on local GPU.

    Mirrors the interface of modal_app.gpu_models.YOLOServer.
    """

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolov8m.pt")
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[LocalGPU] YOLO model loaded and warmed up")

    def detect_persons(self, frame_bytes: bytes, confidence: float = 0.5) -> dict:
        """Detect persons in a single frame.

        Returns:
            dict with keys:
                boxes: list of [x1, y1, x2, y2] in absolute pixels
                confidences: list of float confidence scores
        """
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        results = self.model(image, conf=confidence, verbose=False)

        person_boxes = []
        person_confs = []
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            mask = boxes.cls == 0  # person class
            for box, conf in zip(
                boxes.xyxy[mask].cpu().numpy(),
                boxes.conf[mask].cpu().numpy(),
            ):
                person_boxes.append(box.tolist())
                person_confs.append(float(conf))

        return {"boxes": person_boxes, "confidences": person_confs}

    def detect_batch(
        self, frames_bytes: list[bytes], confidence: float = 0.5
    ) -> list[dict]:
        """Batch detection — processes sequentially on local GPU."""
        return [self.detect_persons(fb, confidence) for fb in frames_bytes]


# ---------------------------------------------------------------------------
# 2. Local SAM3 Server — Text-Prompted Object Segmentation
# ---------------------------------------------------------------------------
class LocalSAM3Server:
    """Segments objects using SAM3 with text prompts on local GPU.

    Mirrors the interface of modal_app.gpu_models.SAM3Server.
    """

    def __init__(self):
        from transformers import Sam3Processor, Sam3Model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model.eval()
        print(f"[LocalGPU] SAM3 model loaded on {self.device}")

    def segment_objects(
        self,
        frame_bytes: bytes,
        text_prompts: list[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """Segment objects matching text prompts in a single frame.

        Returns:
            list of dicts with: label, instance_id, bbox, score, mask_bbox_area
        """
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        all_objects = []

        for prompt in text_prompts:
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results.get("masks", [])
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])

            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                all_objects.append(
                    {
                        "label": prompt,
                        "instance_id": i,
                        "bbox": box.tolist() if hasattr(box, "tolist") else list(box),
                        "score": float(score),
                        "mask_bbox_area": float(mask.sum())
                        if hasattr(mask, "sum")
                        else 0.0,
                    }
                )

        return all_objects


# ---------------------------------------------------------------------------
# 3. Local VL Server — Vision-Language Semantic Grounding
# ---------------------------------------------------------------------------

VL_SYSTEM_PROMPT = """You are a scene analysis model for robotics ground truth generation.
Analyze the image and return ONLY a valid JSON object with this exact schema, no other text:
{"objects": [{"label": "string", "state": "string", "confidence": 0.0}], "action": "string", "description": "string"}

Possible object states: open, closed, partially_open, in_motion, stationary
Possible actions: person_opening_drawer, person_closing_drawer, person_opening_door, person_closing_door, person_reaching, person_standing, person_approaching, person_departing, person_walking, no_action
Only include objects you can actually see in the image. Be precise about states."""


class LocalVLServer:
    """Analyzes keyframes using LFM2.5-VL-1.6B on local GPU.

    Mirrors the interface of modal_app.gpu_models.VLServer.
    Uses HuggingFace transformers directly (not vLLM) for simplicity.
    For vLLM acceleration, see LocalVLServerVLLM below.
    """

    def __init__(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
        print("[LocalGPU] LFM2.5-VL model loaded (HF transformers)")

    def analyze_scene(
        self,
        frame_bytes: bytes,
        system_prompt: str = VL_SYSTEM_PROMPT,
        user_prompt: str = (
            "Analyze this scene. What objects are visible, what are their states, "
            "and what action is the person performing? Return JSON only."
        ),
        max_tokens: int = 256,
    ) -> str:
        """Analyze a keyframe for objects, states, and actions.

        Returns:
            Raw string response (should be JSON if model follows prompt).
        """
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def analyze_batch(
        self,
        frames_bytes: list[bytes],
        system_prompt: str = VL_SYSTEM_PROMPT,
        user_prompt: str = (
            "Analyze this scene. What objects are visible, what are their states, "
            "and what action is the person performing? Return JSON only."
        ),
        max_tokens: int = 256,
    ) -> list[str]:
        """Batch analyze multiple keyframes sequentially."""
        return [
            self.analyze_scene(fb, system_prompt, user_prompt, max_tokens)
            for fb in frames_bytes
        ]


# ---------------------------------------------------------------------------
# 4. Local VL Server with vLLM — Accelerated Inference (Optional)
# ---------------------------------------------------------------------------
class LocalVLServerVLLM:
    """Accelerated VL inference using vLLM for LFM2.5-VL-1.6B.

    vLLM provides:
      - PagedAttention for efficient KV cache management
      - Continuous batching for multiple keyframes
      - Faster token generation

    Falls back to HF transformers if vLLM is not available.
    """

    def __init__(self):
        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model="LiquidAI/LFM2.5-VL-1.6B",
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.4,  # Leave room for YOLO + SAM3
            )
            self.sampling_params = SamplingParams(
                max_tokens=256,
                temperature=0.1,  # Low temp for structured JSON output
            )
            self._use_vllm = True
            print("[LocalGPU] LFM2.5-VL model loaded (vLLM accelerated)")

        except ImportError:
            print("[LocalGPU] vLLM not available, falling back to HF transformers")
            self._fallback = LocalVLServer()
            self._use_vllm = False

    def analyze_scene(
        self,
        frame_bytes: bytes,
        system_prompt: str = VL_SYSTEM_PROMPT,
        user_prompt: str = (
            "Analyze this scene. What objects are visible, what are their states, "
            "and what action is the person performing? Return JSON only."
        ),
        max_tokens: int = 256,
    ) -> str:
        if not self._use_vllm:
            return self._fallback.analyze_scene(
                frame_bytes, system_prompt, user_prompt, max_tokens
            )

        from vllm import SamplingParams

        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

        prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling_params=sampling,
        )
        return outputs[0].outputs[0].text

    def analyze_batch(
        self,
        frames_bytes: list[bytes],
        system_prompt: str = VL_SYSTEM_PROMPT,
        user_prompt: str = (
            "Analyze this scene. What objects are visible, what are their states, "
            "and what action is the person performing? Return JSON only."
        ),
        max_tokens: int = 256,
    ) -> list[str]:
        """Batch analyze — vLLM can process multiple prompts efficiently."""
        if not self._use_vllm:
            return self._fallback.analyze_batch(
                frames_bytes, system_prompt, user_prompt, max_tokens
            )

        from vllm import SamplingParams

        prompts = []
        for fb in frames_bytes:
            image = Image.open(io.BytesIO(fb)).convert("RGB")
            prompts.append(
                {
                    "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
                    "multi_modal_data": {"image": image},
                }
            )

        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        outputs = self.llm.generate(prompts, sampling_params=sampling)
        return [out.outputs[0].text for out in outputs]
