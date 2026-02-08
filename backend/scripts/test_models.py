#!/usr/bin/env python3
"""
RoboSight Model Testing & Analysis Script

Tests YOLO, SAM3, and LFM2.5-VL against demo videos and generates
signal analysis plots for calibration and debugging.

Covers:
  PART 1 — Detection as Signal Generation (bbox stability, motion signal)
  PART 2 — Turn Detection Into Physics (motion score, proximity, velocity)
  PART 3 — Calibration Statistics (threshold estimation from signals)
  PART 4 — Sanity Experiments (motion-only, VL-only, calibration effect)

Usage:
    # Test with local GPU:
    python scripts/test_models.py --backend local --video data/Drawer_Stable.mp4

    # Test with Modal cloud GPU:
    python scripts/test_models.py --backend modal --video data/Drawer_Stable.mp4

    # Test all videos:
    python scripts/test_models.py --backend local --video all

    # Only run YOLO (Part 1 — fastest, start here):
    python scripts/test_models.py --backend local --video data/Drawer_Stable.mp4 --models yolo

    # Run YOLO + SAM3:
    python scripts/test_models.py --backend local --video data/Drawer_Stable.mp4 --models yolo,sam3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.ingest import sample_frames, extract_frames_batch, get_keyframe_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box):
    """Get center of [x1,y1,x2,y2] bbox."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def bbox_area(box):
    """Get area of [x1,y1,x2,y2] bbox."""
    return (box[2] - box[0]) * (box[3] - box[1])


def distance(p1, p2):
    """Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def ema_smooth(values, alpha=0.3):
    """Exponential moving average smoothing."""
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def window_smooth(values, window=5):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        result.append(np.mean(values[start:end]))
    return result


# ---------------------------------------------------------------------------
# PART 1: YOLO Detection Analysis
# ---------------------------------------------------------------------------

def run_yolo_analysis(video_path: str, backend: str, output_dir: str):
    """Run YOLO on all frames and analyze detection stability."""
    print("\n" + "=" * 70)
    print("PART 1: YOLO Detection Analysis")
    print("=" * 70)

    # 1. Sample frames
    print(f"\n[1/4] Sampling frames from {video_path}...")
    frames_info, video_info = sample_frames(video_path, target_fps=5.0)
    all_indices = [f["frame_idx_original"] for f in frames_info]
    all_bytes = extract_frames_batch(video_path, all_indices)
    print(f"  Sampled {len(frames_info)} frames, {video_info['total_keyframes']} keyframes")

    # 2. Run YOLO
    print(f"\n[2/4] Running YOLO person detection ({backend})...")
    from app.pipeline.detect import run_detection
    t0 = time.time()
    detections = run_detection(all_bytes, frames_info, confidence=0.5, backend=backend)
    yolo_time = time.time() - t0
    print(f"  YOLO completed in {yolo_time:.1f}s ({len(detections)} frames processed)")

    # 3. Log detection data
    print(f"\n[3/4] Analyzing detections...")
    frames_with_person = sum(1 for d in detections if len(d["person_boxes"]) > 0)
    print(f"  Person detected in {frames_with_person}/{len(detections)} frames "
          f"({100 * frames_with_person / len(detections):.0f}%)")

    # Per-frame logging
    detection_log = []
    for det in detections:
        entry = {
            "frame_index": det["frame_index"],
            "timestamp": det["timestamp"],
            "num_persons": len(det["person_boxes"]),
        }
        if det["person_boxes"]:
            # Take the highest-confidence person
            best_idx = np.argmax(det["confidences"])
            box = det["person_boxes"][best_idx]
            entry["person_bbox"] = box
            entry["person_center"] = list(bbox_center(box))
            entry["person_area"] = bbox_area(box)
            entry["person_confidence"] = det["confidences"][best_idx]
        detection_log.append(entry)

    # 4. Compute signals
    print(f"\n[4/4] Computing motion signals...")
    timestamps = [d["timestamp"] for d in detection_log]
    person_centers_x = []
    person_centers_y = []
    person_areas = []

    for d in detection_log:
        if "person_center" in d:
            person_centers_x.append(d["person_center"][0])
            person_centers_y.append(d["person_center"][1])
            person_areas.append(d["person_area"])
        else:
            # Interpolate or use last known
            person_centers_x.append(person_centers_x[-1] if person_centers_x else 0)
            person_centers_y.append(person_centers_y[-1] if person_centers_y else 0)
            person_areas.append(person_areas[-1] if person_areas else 0)

    # Smooth
    smooth_cx = ema_smooth(person_centers_x, alpha=0.3)
    smooth_cy = ema_smooth(person_centers_y, alpha=0.3)

    # Save raw data
    raw_data_path = os.path.join(output_dir, "yolo_detections.json")
    with open(raw_data_path, "w") as f:
        json.dump({"detections": detections, "log": detection_log}, f, indent=2)
    print(f"  Raw data saved to {raw_data_path}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"YOLO Detection Analysis: {Path(video_path).stem}", fontsize=14)

        # Plot 1: Person center X over time
        axes[0].plot(timestamps, person_centers_x, "b.", alpha=0.3, markersize=3, label="Raw")
        axes[0].plot(timestamps, smooth_cx, "b-", linewidth=2, label="Smoothed (EMA)")
        axes[0].set_ylabel("Person Center X (px)")
        axes[0].legend()
        axes[0].set_title("Person horizontal position — look for movement patterns")

        # Plot 2: Person center Y over time
        axes[1].plot(timestamps, person_centers_y, "r.", alpha=0.3, markersize=3, label="Raw")
        axes[1].plot(timestamps, smooth_cy, "r-", linewidth=2, label="Smoothed (EMA)")
        axes[1].set_ylabel("Person Center Y (px)")
        axes[1].legend()
        axes[1].set_title("Person vertical position — look for reaching/bending")

        # Plot 3: Person bbox area over time
        smooth_area = ema_smooth(person_areas, alpha=0.3)
        axes[2].plot(timestamps, person_areas, "g.", alpha=0.3, markersize=3, label="Raw")
        axes[2].plot(timestamps, smooth_area, "g-", linewidth=2, label="Smoothed (EMA)")
        axes[2].set_ylabel("Person BBox Area (px²)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].set_title("Person size — approach/depart signal")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "yolo_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved to {plot_path}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    return detections, detection_log, video_info, frames_info, all_bytes


# ---------------------------------------------------------------------------
# PART 1b: SAM3 Segmentation Analysis
# ---------------------------------------------------------------------------

def run_sam3_analysis(video_path: str, backend: str, output_dir: str,
                      frames_info: list, all_bytes: list):
    """Run SAM3 on keyframes and analyze object segmentation."""
    print("\n" + "=" * 70)
    print("PART 1b: SAM3 Segmentation Analysis")
    print("=" * 70)

    # Get keyframes
    kf_orig_indices, kf_info = get_keyframe_data(frames_info)
    kf_indices_in_sampled = [f["index"] for f in kf_info]
    kf_bytes = [all_bytes[i] for i in kf_indices_in_sampled]
    print(f"  {len(kf_info)} keyframes to process")

    # Determine prompts based on video name
    video_stem = Path(video_path).stem.lower()
    if "drawer" in video_stem:
        prompts = ["drawer", "handle"]
    elif "door" in video_stem:
        prompts = ["door", "handle"]
    else:
        prompts = ["drawer", "door", "handle", "cabinet"]
    print(f"  Prompts: {prompts}")

    # Run SAM3
    from app.pipeline.segment import run_segmentation
    t0 = time.time()
    segmentations = run_segmentation(kf_bytes, kf_info, prompts, backend=backend)
    sam3_time = time.time() - t0
    print(f"  SAM3 completed in {sam3_time:.1f}s")

    # Analyze
    for seg in segmentations:
        objs = seg["objects"]
        obj_str = ", ".join([f"{o['label']}({o['score']:.2f})" for o in objs])
        print(f"  t={seg['timestamp']:.1f}s: {len(objs)} objects → {obj_str}")

    # Compute object motion across keyframes (IoU-based tracking)
    print("\n  Object tracking across keyframes (IoU matching):")
    object_tracks = {}  # label -> list of {timestamp, bbox, score}

    for seg in segmentations:
        for obj in seg["objects"]:
            label = obj["label"]
            if label not in object_tracks:
                object_tracks[label] = []
            object_tracks[label].append({
                "timestamp": seg["timestamp"],
                "bbox": obj["bbox"],
                "score": obj["score"],
                "area": obj.get("mask_bbox_area", bbox_area(obj["bbox"])),
            })

    motion_signals = {}
    for label, track in object_tracks.items():
        motions = []
        for i in range(1, len(track)):
            iou = compute_iou(track[i - 1]["bbox"], track[i]["bbox"])
            motion = 1.0 - iou
            motions.append({
                "timestamp": track[i]["timestamp"],
                "motion_score": round(motion, 4),
                "iou": round(iou, 4),
            })
            status = "MOTION" if motion > 0.15 else "stable"
            print(f"    {label} t={track[i]['timestamp']:.1f}s: IoU={iou:.3f} motion={motion:.3f} [{status}]")
        motion_signals[label] = motions

    # Save
    seg_data_path = os.path.join(output_dir, "sam3_segmentations.json")
    with open(seg_data_path, "w") as f:
        json.dump({
            "segmentations": segmentations,
            "object_tracks": {k: v for k, v in object_tracks.items()},
            "motion_signals": motion_signals,
        }, f, indent=2, default=str)
    print(f"\n  Data saved to {seg_data_path}")

    # Plot object motion
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"SAM3 Object Analysis: {Path(video_path).stem}", fontsize=14)

        colors = {"drawer": "orange", "door": "blue", "handle": "magenta", "cabinet": "green"}

        # Plot 1: Object bbox area over time
        for label, track in object_tracks.items():
            ts = [t["timestamp"] for t in track]
            areas = [t["area"] for t in track]
            axes[0].plot(ts, areas, "o-", color=colors.get(label, "gray"),
                        label=label, markersize=6)
        axes[0].set_ylabel("Object Area (px² or mask area)")
        axes[0].legend()
        axes[0].set_title("Object size over time — area change = state change")

        # Plot 2: Motion score over time
        for label, motions in motion_signals.items():
            ts = [m["timestamp"] for m in motions]
            scores = [m["motion_score"] for m in motions]
            axes[1].plot(ts, scores, "o-", color=colors.get(label, "gray"),
                        label=label, markersize=6)
        axes[1].axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="threshold (0.15)")
        axes[1].set_ylabel("Motion Score (1 - IoU)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].set_title("Object motion — spikes = state transitions")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "sam3_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved to {plot_path}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    return segmentations, object_tracks, motion_signals


# ---------------------------------------------------------------------------
# PART 1c: VL Semantics Analysis
# ---------------------------------------------------------------------------

def run_vl_analysis(video_path: str, backend: str, output_dir: str,
                    frames_info: list, all_bytes: list):
    """Run VL on keyframes and analyze semantic understanding."""
    print("\n" + "=" * 70)
    print("PART 1c: VL Semantics Analysis")
    print("=" * 70)

    kf_orig_indices, kf_info = get_keyframe_data(frames_info)
    kf_indices_in_sampled = [f["index"] for f in kf_info]
    kf_bytes = [all_bytes[i] for i in kf_indices_in_sampled]
    print(f"  {len(kf_info)} keyframes to process")

    from app.pipeline.semantics import run_semantics
    t0 = time.time()
    sem = run_semantics(kf_bytes, kf_info, backend=backend)
    vl_time = time.time() - t0
    print(f"  VL completed in {vl_time:.1f}s")

    for s in sem:
        objs = ", ".join([f"{o['label']}:{o['state']}({o.get('confidence', '?')})"
                          for o in s["objects"]])
        print(f"  t={s['timestamp']:.1f}s: action={s['action']} | objects={objs}")
        if s["description"]:
            print(f"            desc: {s['description'][:100]}")

    # Save
    vl_data_path = os.path.join(output_dir, "vl_semantics.json")
    with open(vl_data_path, "w") as f:
        json.dump(sem, f, indent=2)
    print(f"\n  Data saved to {vl_data_path}")

    return sem


# ---------------------------------------------------------------------------
# PART 2: Physics Signals (combining YOLO + SAM3)
# ---------------------------------------------------------------------------

def run_physics_analysis(output_dir: str, detection_log: list,
                         object_tracks: dict, motion_signals: dict,
                         video_path: str):
    """Compute physics signals: proximity, dwell time, combined confidence."""
    print("\n" + "=" * 70)
    print("PART 2: Physics Signal Analysis")
    print("=" * 70)

    # Build person position timeline
    person_timeline = {}
    for d in detection_log:
        if "person_center" in d:
            person_timeline[round(d["timestamp"], 1)] = d["person_center"]

    # For each tracked object, compute proximity to person over time
    proximity_signals = {}
    for label, track in object_tracks.items():
        prox = []
        for entry in track:
            t = round(entry["timestamp"], 1)
            obj_center = bbox_center(entry["bbox"])

            # Find nearest person position
            best_dist = float("inf")
            for pt, pc in person_timeline.items():
                if abs(pt - t) < 0.5:
                    d = distance(pc, obj_center)
                    if d < best_dist:
                        best_dist = d

            proximity = 1.0 / (1.0 + best_dist / 500.0) if best_dist < float("inf") else 0.0
            prox.append({
                "timestamp": entry["timestamp"],
                "distance_px": round(best_dist, 1) if best_dist < float("inf") else -1,
                "proximity_score": round(proximity, 4),
            })
            status = "CLOSE" if proximity > 0.3 else "far"
            print(f"  {label} t={entry['timestamp']:.1f}s: dist={best_dist:.0f}px prox={proximity:.3f} [{status}]")

        proximity_signals[label] = prox

    # Combined signal analysis
    print("\n  Combined signal analysis (motion + proximity):")
    combined = {}
    for label in object_tracks:
        motions = {round(m["timestamp"], 1): m["motion_score"]
                   for m in motion_signals.get(label, [])}
        proximities = {round(p["timestamp"], 1): p["proximity_score"]
                       for p in proximity_signals.get(label, [])}

        entries = []
        for t in sorted(set(list(motions.keys()) + list(proximities.keys()))):
            m = motions.get(t, 0.0)
            p = proximities.get(t, 0.0)
            # Weighted combined score
            score = 0.4 * m + 0.3 * p + 0.3 * 0.5  # VL placeholder at 0.5
            is_event = m > 0.15 and p > 0.3
            entries.append({
                "timestamp": t,
                "motion": round(m, 4),
                "proximity": round(p, 4),
                "combined": round(score, 4),
                "event_candidate": is_event,
            })
            if is_event:
                print(f"    *** EVENT CANDIDATE at t={t:.1f}s: "
                      f"motion={m:.3f} proximity={p:.3f} combined={score:.3f}")

        combined[label] = entries

    # Save
    physics_path = os.path.join(output_dir, "physics_signals.json")
    with open(physics_path, "w") as f:
        json.dump({
            "proximity_signals": proximity_signals,
            "combined_signals": combined,
        }, f, indent=2)
    print(f"\n  Data saved to {physics_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for label, entries in combined.items():
            if not entries:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(f"Physics Signals: {label} — {Path(video_path).stem}", fontsize=14)

            ts = [e["timestamp"] for e in entries]
            motions = [e["motion"] for e in entries]
            proximities = [e["proximity"] for e in entries]
            combined_scores = [e["combined"] for e in entries]
            events = [e["timestamp"] for e in entries if e["event_candidate"]]

            axes[0].plot(ts, motions, "o-", color="orange", markersize=6)
            axes[0].axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
            for et in events:
                axes[0].axvline(x=et, color="green", alpha=0.3, linewidth=3)
            axes[0].set_ylabel("Motion Score")
            axes[0].set_title("Motion (1 - IoU) — spikes = object moved")

            axes[1].plot(ts, proximities, "o-", color="blue", markersize=6)
            axes[1].axhline(y=0.3, color="red", linestyle="--", alpha=0.5)
            for et in events:
                axes[1].axvline(x=et, color="green", alpha=0.3, linewidth=3)
            axes[1].set_ylabel("Proximity Score")
            axes[1].set_title("Proximity — person near object")

            axes[2].plot(ts, combined_scores, "o-", color="purple", markersize=6)
            for et in events:
                axes[2].axvline(x=et, color="green", alpha=0.3, linewidth=3, label="Event" if et == events[0] else "")
            axes[2].set_ylabel("Combined Score")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()
            axes[2].set_title("Combined signal — green bars = event candidates")

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"physics_{label}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot saved to {plot_path}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    return proximity_signals, combined


# ---------------------------------------------------------------------------
# PART 3: Calibration Statistics
# ---------------------------------------------------------------------------

def run_calibration_analysis(output_dir: str, motion_signals: dict,
                             proximity_signals: dict, combined: dict):
    """Compute calibration statistics from signals."""
    print("\n" + "=" * 70)
    print("PART 3: Calibration Statistics")
    print("=" * 70)

    for label in combined:
        entries = combined[label]
        if not entries:
            continue

        motions = [e["motion"] for e in entries if e["motion"] > 0]
        proximities = [e["proximity"] for e in entries if e["proximity"] > 0]
        events = [e for e in entries if e["event_candidate"]]

        print(f"\n  {label}:")
        if motions:
            print(f"    Motion:    mean={np.mean(motions):.3f} std={np.std(motions):.3f} "
                  f"min={np.min(motions):.3f} max={np.max(motions):.3f}")
            suggested_motion_thresh = max(0.05, np.mean(motions) - 0.5 * np.std(motions))
            print(f"    → Suggested motion threshold: {suggested_motion_thresh:.3f}")

        if proximities:
            print(f"    Proximity: mean={np.mean(proximities):.3f} std={np.std(proximities):.3f} "
                  f"min={np.min(proximities):.3f} max={np.max(proximities):.3f}")
            suggested_prox_thresh = max(0.1, np.mean(proximities) - 0.5 * np.std(proximities))
            print(f"    → Suggested proximity threshold: {suggested_prox_thresh:.3f}")

        print(f"    Event candidates: {len(events)}")
        if events:
            event_motions = [e["motion"] for e in events]
            event_prox = [e["proximity"] for e in events]
            print(f"    Event motion:    mean={np.mean(event_motions):.3f}")
            print(f"    Event proximity: mean={np.mean(event_prox):.3f}")

    # Save calibration recommendations
    calibration = {}
    for label in combined:
        entries = combined[label]
        motions = [e["motion"] for e in entries if e["motion"] > 0]
        proximities = [e["proximity"] for e in entries if e["proximity"] > 0]
        if motions and proximities:
            calibration[label] = {
                "suggested_motion_threshold": round(max(0.05, np.mean(motions) - 0.5 * np.std(motions)), 4),
                "suggested_proximity_threshold": round(max(0.1, np.mean(proximities) - 0.5 * np.std(proximities)), 4),
                "motion_stats": {"mean": round(np.mean(motions), 4), "std": round(np.std(motions), 4)},
                "proximity_stats": {"mean": round(np.mean(proximities), 4), "std": round(np.std(proximities), 4)},
            }

    cal_path = os.path.join(output_dir, "calibration_suggestions.json")
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n  Calibration saved to {cal_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_video(video_path: str, backend: str, models: list[str]):
    """Run full analysis on a single video."""
    video_stem = Path(video_path).stem
    output_dir = os.path.join("jobs", f"test_{video_stem}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Processing: {video_stem}")
    print(f"# Backend: {backend}")
    print(f"# Models: {', '.join(models)}")
    print(f"# Output: {output_dir}/")
    print(f"{'#' * 70}")

    # Always run ingestion
    frames_info, video_info = sample_frames(video_path, target_fps=5.0)
    all_indices = [f["frame_idx_original"] for f in frames_info]
    all_bytes = extract_frames_batch(video_path, all_indices)

    # Save video info
    with open(os.path.join(output_dir, "video_info.json"), "w") as f:
        json.dump(video_info, f, indent=2)

    detections = None
    detection_log = None
    object_tracks = None
    motion_signals = None
    segmentations = None

    # PART 1: YOLO
    if "yolo" in models:
        detections, detection_log, _, _, _ = run_yolo_analysis(
            video_path, backend, output_dir
        )

    # PART 1b: SAM3
    if "sam3" in models:
        segmentations, object_tracks, motion_signals = run_sam3_analysis(
            video_path, backend, output_dir, frames_info, all_bytes
        )

    # PART 1c: VL
    if "vl" in models:
        run_vl_analysis(video_path, backend, output_dir, frames_info, all_bytes)

    # PART 2: Physics (needs YOLO + SAM3)
    if detection_log and object_tracks and motion_signals:
        proximity_signals, combined = run_physics_analysis(
            output_dir, detection_log, object_tracks, motion_signals, video_path
        )

        # PART 3: Calibration
        run_calibration_analysis(output_dir, motion_signals, proximity_signals, combined)

    print(f"\n{'=' * 70}")
    print(f"DONE: {video_stem} — results in {output_dir}/")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="RoboSight Model Testing")
    parser.add_argument("--backend", default="local", choices=["modal", "local", "local_vllm"],
                        help="GPU backend to use")
    parser.add_argument("--video", default="data/Drawer_Stable.mp4",
                        help="Video path or 'all' for all videos")
    parser.add_argument("--models", default="yolo,sam3,vl",
                        help="Comma-separated models to test: yolo,sam3,vl")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    if args.video == "all":
        data_dir = "data"
        videos = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
                  if f.endswith(".mp4")]
    else:
        videos = [args.video]

    for video_path in videos:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        process_video(video_path, args.backend, models)


if __name__ == "__main__":
    main()
