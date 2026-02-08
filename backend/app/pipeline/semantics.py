"""
Vision-Language Semantics Wrapper

Analyzes keyframes for objects, states, and actions. Supports two backends:
  - "modal": Calls Modal-hosted VLServer (cloud GPU)
  - "local": Uses local NVIDIA GPU via LocalVLServer
  - "local_vllm": Uses local NVIDIA GPU with vLLM acceleration

Runs on KEYFRAMES ONLY.

Output format (semantics):
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
        },
        ...
    ]
"""

import json
import re


def run_semantics(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    max_tokens: int = 256,
    backend: str = "modal",
) -> list[dict]:
    """Run VL semantic grounding on keyframes.

    Args:
        keyframe_bytes: JPEG-encoded keyframes.
        keyframes_info: Metadata for each keyframe (index, timestamp).
        max_tokens: Maximum tokens for VL model generation.
        backend: "modal" for cloud GPU, "local" for local HF transformers,
                 "local_vllm" for local vLLM accelerated.

    Returns:
        List of semantic annotation dicts, one per keyframe.
    """
    if backend in ("local", "local_vllm"):
        return _run_local(keyframe_bytes, keyframes_info, max_tokens, backend)
    else:
        return _run_modal(keyframe_bytes, keyframes_info, max_tokens)


def _run_modal(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    max_tokens: int,
) -> list[dict]:
    """Run semantics via Modal cloud GPU."""
    import modal

    VLServer = modal.Cls.from_name("robosight-gpu", "VLServer")
    server = VLServer()

    semantics = []
    for frame_bytes, info in zip(keyframe_bytes, keyframes_info):
        raw_response = server.analyze_scene.remote(
            frame_bytes, max_tokens=max_tokens
        )
        parsed = _normalize_vl_output(_parse_vl_json(raw_response))
        semantics.append(
            {
                "frame_index": info["index"],
                "timestamp": info["timestamp"],
                "objects": parsed.get("objects", []),
                "action": parsed.get("action", "no_action"),
                "description": parsed.get("description", ""),
                "raw_response": raw_response,
            }
        )

    return semantics


def _run_local(
    keyframe_bytes: list[bytes],
    keyframes_info: list[dict],
    max_tokens: int,
    backend: str,
) -> list[dict]:
    """Run semantics on local NVIDIA GPU."""
    from app.local_app.gpu_models import get_server

    if backend == "local_vllm":
        from app.local_app.gpu_models import LocalVLServerVLLM
        server = get_server(LocalVLServerVLLM)
    else:
        from app.local_app.gpu_models import LocalVLServer
        server = get_server(LocalVLServer)

    semantics = []
    for frame_bytes, info in zip(keyframe_bytes, keyframes_info):
        raw_response = server.analyze_scene(
            frame_bytes, max_tokens=max_tokens
        )
        parsed = _normalize_vl_output(_parse_vl_json(raw_response))
        semantics.append(
            {
                "frame_index": info["index"],
                "timestamp": info["timestamp"],
                "objects": parsed.get("objects", []),
                "action": parsed.get("action", "no_action"),
                "description": parsed.get("description", ""),
                "raw_response": raw_response,
            }
        )

    return semantics


def _parse_vl_json(response: str) -> dict:
    """Extract JSON from VL model response, handling non-JSON wrapping."""
    # Strategy 1: Direct parse
    try:
        return json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Find JSON block in response
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: JSON in markdown code blocks
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "objects": [],
        "action": "no_action",
        "description": response[:200] if response else "VL model returned no response",
    }


def _normalize_vl_output(parsed: dict) -> dict:
    """Normalize VL model output to match the SemanticObject contract.

    The VL model often returns {label, state, action} per object instead of
    {label, state, confidence}. This function:
      - Ensures every object has a 'confidence' field (defaults to 0.8).
      - Strips the per-object 'action' field (not in the contract).
      - Infers the top-level 'action' from per-object actions when missing.
    """
    objects = parsed.get("objects", [])
    top_action = parsed.get("action", "")
    inferred_action = "no_action"

    for obj in objects:
        # Ensure confidence exists
        if "confidence" not in obj or not isinstance(obj.get("confidence"), (int, float)):
            obj["confidence"] = 0.8

        # Infer top-level action from per-object action fields
        obj_action = obj.pop("action", "")
        obj_label = obj.get("label", "")
        if obj_action and inferred_action == "no_action":
            inferred_action = _infer_frame_action(obj_label, obj_action)

    # Use model's top-level action if it provided one, otherwise use inferred
    if not top_action or top_action == "no_action":
        parsed["action"] = inferred_action
    else:
        parsed["action"] = top_action

    if not parsed.get("description"):
        parsed["description"] = ""

    parsed["objects"] = objects
    return parsed


# Action keywords mapped to standard pipeline action names
_ACTION_KEYWORDS = {
    "opening": "person_opening_{label}",
    "open": "person_opening_{label}",
    "closing": "person_closing_{label}",
    "close": "person_closing_{label}",
    "reaching": "person_reaching",
    "standing": "person_standing",
    "approaching": "person_approaching",
    "walking": "person_walking",
    "departing": "person_departing",
}


def _infer_frame_action(obj_label: str, obj_action: str) -> str:
    """Map a per-object action string to a standard pipeline action."""
    action_lower = obj_action.lower()
    for keyword, template in _ACTION_KEYWORDS.items():
        if keyword in action_lower:
            return template.replace("{label}", obj_label)
    return "no_action"
