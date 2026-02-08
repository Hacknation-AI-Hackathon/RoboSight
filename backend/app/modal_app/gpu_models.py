"""
RoboSight GPU Model Servers — Modal Deployment

Three GPU-backed model servers for the RoboSight pipeline:
  1. YOLOServer  — Person detection on every sampled frame
  2. SAM3Server  — Text-prompted object segmentation on keyframes
  3. VLServer    — Vision-language semantic grounding on keyframes

Deploy: modal deploy backend/app/modal_app/gpu_models.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("robosight-gpu")

# ---------------------------------------------------------------------------
# Container Images (one per model to keep layers independent)
# ---------------------------------------------------------------------------
yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.10.0",
        "numpy>=1.26.0",
        "Pillow>=10.0.0",
    )
)

sam3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "git+https://github.com/huggingface/transformers.git@2a5ba8b53d298ed8421e09831bf96bb6d056a24d",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "accelerate>=1.0.0",
    )
)

vl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "git+https://github.com/huggingface/transformers.git@2a5ba8b53d298ed8421e09831bf96bb6d056a24d",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "accelerate>=1.0.0",
    )
)

# ---------------------------------------------------------------------------
# 1. YOLO Server — Person Detection
# ---------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=yolo_image,
    scaledown_window=300,
    timeout=600,
)
class YOLOServer:
    """Detects persons in video frames using YOLOv8m.

    Input:  JPEG-encoded frame bytes
    Output: {boxes: [[x1,y1,x2,y2], ...], confidences: [float, ...]}
    """

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8m.pt")
        # Warm up with a dummy inference
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("YOLO model loaded and warmed up")

    def _detect_single(self, frame_bytes: bytes, confidence: float = 0.5) -> dict:
        """Internal: detect persons in a single frame."""
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        results = self.model(image, conf=confidence, verbose=False)

        person_boxes = []
        person_confs = []
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            mask = boxes.cls == 0
            for box, conf in zip(
                boxes.xyxy[mask].cpu().numpy(),
                boxes.conf[mask].cpu().numpy(),
            ):
                person_boxes.append(box.tolist())
                person_confs.append(float(conf))

        return {"boxes": person_boxes, "confidences": person_confs}

    @modal.method()
    def detect_persons(self, frame_bytes: bytes, confidence: float = 0.5) -> dict:
        """Detect persons in a single frame.

        Returns:
            dict with keys:
                boxes: list of [x1, y1, x2, y2] in absolute pixels
                confidences: list of float confidence scores
        """
        return self._detect_single(frame_bytes, confidence)

    @modal.method()
    def detect_batch(
        self, frames_bytes: list[bytes], confidence: float = 0.5
    ) -> list[dict]:
        """Batch detection for efficiency. Processes frames sequentially on GPU."""
        return [self._detect_single(fb, confidence) for fb in frames_bytes]


# ---------------------------------------------------------------------------
# 2. SAM3 Server — Text-Prompted Object Segmentation
# ---------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=sam3_image,
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
class SAM3Server:
    """Segments objects in keyframes using SAM3 with text prompts.

    Input:  JPEG frame bytes + list of text prompts (e.g. ["drawer", "handle"])
    Output: list of {label, instance_id, bbox, score, mask_bbox_area}
    """

    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import Sam3Processor, Sam3Model

        hf_token = os.environ.get("HF_TOKEN")
        self.device = "cuda"
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        self.model.eval()
        print("SAM3 model loaded")

    @modal.method()
    def segment_objects(
        self,
        frame_bytes: bytes,
        text_prompts: list[str],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """Segment objects matching text prompts in a single frame.

        Args:
            frame_bytes: JPEG-encoded image
            text_prompts: e.g. ["drawer", "handle", "door"]
            threshold: detection confidence threshold
            mask_threshold: mask binarization threshold

        Returns:
            list of dicts, each with:
                label: str (which prompt matched)
                instance_id: int (per-label instance index)
                bbox: [x1, y1, x2, y2] in absolute pixels
                score: float confidence
                mask_bbox_area: float (mask pixel count within bbox)
        """
        import torch
        from PIL import Image
        import io

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
# 3. VL Server — Vision-Language Semantic Grounding
# ---------------------------------------------------------------------------

VL_SYSTEM_PROMPT = """You are a scene analysis model for robotics ground truth generation.
Analyze the image and return ONLY a valid JSON object with this exact schema, no other text:
{"objects": [{"label": "string", "state": "string", "confidence": 0.0}], "action": "string", "description": "string"}

Possible object states: open, closed, partially_open, in_motion, stationary
Possible actions: person_opening_drawer, person_closing_drawer, person_opening_door, person_closing_door, person_reaching, person_standing, person_approaching, person_departing, person_walking, no_action
Only include objects you can actually see in the image. Be precise about states."""


@app.cls(
    gpu="T4",
    image=vl_image,
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
class VLServer:
    """Analyzes keyframes for semantic scene understanding using LFM2.5-VL.

    Input:  JPEG frame bytes
    Output: raw text response (forced JSON via system prompt)
    """

    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        hf_token = os.environ.get("HF_TOKEN")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "LiquidAI/LFM2.5-VL-1.6B",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        self.processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
        print("LFM2.5-VL model loaded")

    @modal.method()
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
        from PIL import Image
        import io

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

    @modal.method()
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
# Local entrypoint for quick testing
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def test():
    """Quick smoke test: send a dummy image to each server."""
    from PIL import Image
    import io

    # Create a simple test image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    test_bytes = buf.getvalue()

    print("Testing YOLO...")
    yolo = YOLOServer()
    yolo_result = yolo.detect_persons.remote(test_bytes)
    print(f"  YOLO result: {yolo_result}")

    print("Testing SAM3...")
    sam3 = SAM3Server()
    sam3_result = sam3.segment_objects.remote(test_bytes, ["person", "object"])
    print(f"  SAM3 result: {sam3_result}")

    print("Testing VL...")
    vl = VLServer()
    vl_result = vl.analyze_scene.remote(test_bytes)
    print(f"  VL result: {vl_result}")

    print("All servers operational!")
