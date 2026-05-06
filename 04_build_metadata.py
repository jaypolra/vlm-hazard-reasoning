"""
04_build_metadata.py
--------------------
Walks the entire dataset and builds:
  1. master_metadata.csv   — one row per video
  2. frames_index.csv      — one row per frame (for VLM experiment tracking)

Usage:
    python 04_build_metadata.py
    python 04_build_metadata.py --output my_dataset_index.csv
"""

import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def infer_metadata_from_path(video_path: str) -> dict:
    """Infer phase, category, source from folder structure."""
    parts = Path(video_path).parts

    phase = "unknown"
    category = "unknown"
    source = "unknown"

    for p in parts:
        if "phase1" in p:
            phase = "phase1_opensource"
        elif "phase2" in p:
            phase = "phase2_industrial"
        elif "your_mkv" in p:
            phase = "phase2_industrial"
            source = "your_upload"

        if "PPE" in p or "ppe" in p:
            category = "PPE_violation"
        elif "fall" in p:
            category = "fall_from_height"
        elif "machinery" in p:
            category = "machinery_danger"
        elif "near_miss" in p or "near-miss" in p:
            category = "near_miss"

    if "youtube" in video_path.lower() or source == "unknown":
        source = "youtube"
    if "mendeley" in video_path.lower():
        source = "mendeley"
    if "roboflow" in video_path.lower():
        source = "roboflow"
    if "your_mkv" in video_path.lower():
        source = "your_upload"

    return {"phase": phase, "category": category, "source": source}


def count_frames(frames_dir: str, video_stem: str = None) -> int:
    """Count extracted jpg frames in a frames/ folder, optionally filtered by video stem."""
    if not os.path.exists(frames_dir):
        return 0
    files = os.listdir(frames_dir)
    if video_stem:
        files = [f for f in files if f.startswith(video_stem) and f.endswith(".jpg")]
    else:
        files = [f for f in files if f.endswith(".jpg")]
    return len(files)


def load_manifest(frames_dir: str, video_stem: str) -> dict:
    """Load yt-dlp/extractor manifest if available."""
    manifest_path = os.path.join(frames_dir, f"{video_stem}_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def build_metadata(root: str) -> tuple:
    """Walk dataset and collect all video and frame metadata."""
    video_rows = []
    frame_rows = []
    video_id = 1

    for dirpath, dirnames, filenames in os.walk(root):
        if not (dirpath.endswith("raw") or "/raw" in dirpath or "\\raw" in dirpath):
            continue

        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext not in VIDEO_EXTENSIONS:
                continue

            video_path = os.path.join(dirpath, fname)
            video_stem = Path(fname).stem
            frames_dir = dirpath.replace(os.sep + "raw", os.sep + "frames").replace("/raw", "/frames")

            meta = infer_metadata_from_path(video_path)
            frame_count = count_frames(frames_dir, video_stem)
            manifest = load_manifest(frames_dir, video_stem)

            vid_id = f"VID_{video_id:04d}"
            row = {
                "video_id": vid_id,
                "filename": fname,
                "phase": meta["phase"],
                "category": meta["category"],
                "source": meta["source"],
                "video_path": video_path,
                "frames_dir": frames_dir,
                "frames_extracted": frame_count,
                "duration_sec": manifest.get("duration_sec", ""),
                "resolution": manifest.get("resolution", ""),
                "original_fps": manifest.get("original_fps", ""),
                "extracted_fps": manifest.get("extracted_fps", ""),
                "frames_ready": "YES" if frame_count > 0 else "NO",
                # Experiment tracking columns (fill in later)
                "vlm_tested": "",
                "prompt_type": "",
                "hazard_detected": "",
                "detection_correct": "",
                "notes": ""
            }
            video_rows.append(row)

            # Per-frame rows
            if manifest.get("frames"):
                for frame in manifest["frames"]:
                    frame_rows.append({
                        "video_id": vid_id,
                        "frame_filename": frame["filename"],
                        "timestamp_sec": frame["timestamp_sec"],
                        "frame_index": frame["frame_index"],
                        "phase": meta["phase"],
                        "category": meta["category"],
                        "frame_path": frame["path"],
                        # VLM result columns (fill in during experiment)
                        "vlm_model": "",
                        "prompt_variant": "",
                        "vlm_response": "",
                        "hazard_detected": "",
                        "severity_label": "",
                        "correct": ""
                    })

            video_id += 1

    return video_rows, frame_rows


def write_csv(rows: list, path: str):
    if not rows:
        print(f"  [SKIP] No data for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved : {path} ({len(rows)} rows)")


def print_summary(video_rows: list):
    from collections import Counter
    print("\n Dataset Summary")
    print("=" * 45)

    phase_counts = Counter(r["phase"] for r in video_rows)
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase:<30} {count} videos")

    print()
    cat_counts = Counter(r["category"] for r in video_rows)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<30} {count} videos")

    print()
    ready = sum(1 for r in video_rows if r["frames_ready"] == "YES")
    print(f"  Total videos     : {len(video_rows)}")
    print(f"  Frames extracted : {ready} / {len(video_rows)} videos")
    print(f"  Total frames     : {sum(int(r['frames_extracted']) for r in video_rows)}")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description="Build master metadata CSV for VLM experiment")
    parser.add_argument("--root", default=".", help="Root dataset folder")
    parser.add_argument("--output", default="master_metadata.csv", help="Output CSV filename")
    parser.add_argument("--frames-output", default="frames_index.csv", help="Frames index CSV filename")
    args = parser.parse_args()

    print("=" * 50)
    print(" VLM Hazard Dataset — Metadata Builder")
    print("=" * 50)
    print(f" Scanning : {os.path.abspath(args.root)}")
    print()

    video_rows, frame_rows = build_metadata(args.root)

    output_path = os.path.join(args.root, args.output)
    frames_path = os.path.join(args.root, args.frames_output)

    write_csv(video_rows, output_path)
    write_csv(frame_rows, frames_path)

    print_summary(video_rows)
    print(f"\n Open master_metadata.csv to track your VLM experiment results.")
    print(f" Each video has columns: vlm_tested, prompt_type, hazard_detected, detection_correct")
    print(f"\n Next step: Prompt templates + VLM inference!")


if __name__ == "__main__":
    main()
