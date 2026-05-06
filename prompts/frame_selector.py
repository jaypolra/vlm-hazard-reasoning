"""
ARIA — AI Risk & Incident Analyzer
VLM Reasoning Layer for Industrial Hazard Recognition
CIVS × Purdue University Northwest · SDI Butler Steel Plant

Frame selection module — picks the right frames for each experiment scenario.

Two jobs:
  1. Phase 1 (YouTube videos): select representative frames per hazard
     category from frames_index.csv using even spacing.
  2. Phase 2 (SDI MKVs): select frames that capture each SDI-specific
     hazard scenario based on manually tagged timestamps.

Also exports experiment_queue.csv — one row per (frame, variant, hazard_key)
combination, ready for batch VLM inference via 05_vlm_inference.py.

Part of the research:
  "Bridging the Domain Gap in Industrial Safety with Vision Language Models"
  CIVS × Purdue University Northwest for SDI Butler Steel Plant
"""

import os
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent  # vlm_hazard_dataset/


# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — Frame selection from YouTube videos
# Pick N evenly-spaced frames per video per category
# ─────────────────────────────────────────────────────────────────────

# Which hazard key from hazard_library to use for each video category
PHASE1_CATEGORY_TO_HAZARD = {
    "PPE_violation":    ["ppe_no_helmet", "ppe_no_vest", "ppe_no_harness"],
    "fall_from_height": ["fall_from_height"],
    "machinery_danger": ["machinery_unguarded", "machinery_loto_violation", "forklift_pedestrian"],
    "near_miss":        ["near_miss_falling_object", "near_miss_electrical", "near_miss_vehicle_pedestrian"],
}

# How many frames per video to include in experiments (keep small for cost)
FRAMES_PER_VIDEO_PHASE1 = 5


# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — SDI MKV frame tagging
# Manually tagged frames that best represent each SDI scenario
# Update these after visually reviewing the extracted frames
# ─────────────────────────────────────────────────────────────────────

# Format: {video_id: [(timestamp_approx, scenario_key, note), ...]}
# Fill in after reviewing frames in your_mkv_files/frames/
SDI_FRAME_TAGS = {
    # ── video-1.mkv — Camera 1, North bay view ─────────────────────────
    # Pot hauler comes from north, parks, returns south with glowing molten pot
    "VID_0023": [
        (1,   "people_with_active_equipment", "Front of pot hauler partially visible at north end — vehicle entering bay"),
        (88,  "people_with_active_equipment", "Vehicle begins moving south through bay — active equipment in personnel zone"),
        (240, "people_with_active_equipment", "Vehicle returning north — glowing molten metal pot visible, CRITICAL hazard"),
        (270, "red_light_indicator",          "Vehicle carrying hot metal pot moving through bay — red zone active"),
        (285, "people_with_active_equipment", "Vehicle exiting north end of bay — pot still glowing"),
    ],

    # ── video-2.mkv — Camera 2, South-facing view, sees both ends ──────
    # Best angle for south bay. Also captures north vehicle simultaneously at 3:29.
    "VID_0024": [
        (120, "blind_spot_vehicle",           "North vehicle visible from south angle at 2:00 — shows blind spot from cam1/3"),
        (140, "blind_spot_vehicle",           "North vehicle parked — partially out of camera frame, blind spot confirmed"),
        (180, "people_with_active_equipment", "Second vehicle enters from south end at 3:00 — multi-vehicle scenario begins"),
        (209, "red_light_indicator",          "North vehicle exits blind spot with glowing pot + south vehicle present — CRITICAL multi-vehicle"),
        (230, "people_with_active_equipment", "North vehicle exits bay — south vehicle still stationary at south end"),
        (360, "people_with_active_equipment", "South vehicle from 3:00 still present at south end of bay — extended exposure"),
        (380, "missing_blockers",             "South vehicle still present near end of video — blocker status unclear from this angle"),
    ],

    # ── video-3.mkv — Camera 3, North bay, different angle from cam1 ───
    # Shows blind spot parking most clearly — vehicle disappears into blind spot
    "VID_0025": [
        (100, "blind_spot_vehicle",           "Vehicle entering at 1:40 — partial vehicle visible at blind spot entrance (north bay)"),
        (105, "blind_spot_vehicle",           "Vehicle still entering blind spot — only front/rear visible, body out of frame"),
        (150, "missing_blockers",             "Vehicle fully parked in blind spot at 2:30 — completely out of camera vision, blocker status unconfirmable"),
        (204, "red_light_indicator",          "Vehicle re-emerging from blind spot at 3:24 — glowing pot visible, active hot metal transport"),
        (230, "people_with_active_equipment", "Vehicle exiting blind spot area — molten pot still glowing, personnel must be clear"),
    ],

    # ── video-4.mkv — Camera 4, South bay entrance only ────────────────
    # Different pot hauler than videos 1/2/3. South-only view.
    "VID_0026": [
        (150, "people_with_active_equipment", "Second pot hauler enters from south bay at 2:30 — distinct from north vehicle"),
        (170, "blind_spot_vehicle",           "South vehicle moving out of camera vision at 2:50 — entering south blind spot"),
        (300, "missing_blockers",             "South bay area — vehicle no longer visible, blocker/zone status unconfirmable from this camera"),
        (370, "swing_gate_low_visibility",    "End of video — south bay, low visibility conditions, gate status unclear"),
    ],
}


def load_frames_index() -> list:
    """Load frames_index.csv into a list of dicts."""
    path = ROOT / "frames_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"frames_index.csv not found at {path}. Run 04_build_metadata.py first.")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_master_metadata() -> list:
    """Load master_metadata.csv into a list of dicts."""
    path = ROOT / "master_metadata.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_phase1_frames(n_per_video: int = FRAMES_PER_VIDEO_PHASE1) -> list:
    """
    Select N evenly-spaced frames from each Phase 1 video.
    Returns list of dicts ready for experiment queue.
    """
    all_frames = load_frames_index()
    metadata = {r["video_id"]: r for r in load_master_metadata()}

    selected = []

    # Group frames by video
    by_video = {}
    for frame in all_frames:
        vid = frame["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(frame)

    for vid_id, frames in by_video.items():
        meta = metadata.get(vid_id, {})
        if meta.get("phase") != "phase1_opensource":
            continue

        category = meta.get("category", "unknown")
        hazard_keys = PHASE1_CATEGORY_TO_HAZARD.get(category, [])

        if not hazard_keys:
            continue

        # Evenly space frame selection
        total = len(frames)
        step = max(1, total // n_per_video)
        chosen = frames[::step][:n_per_video]

        for frame in chosen:
            for hazard_key in hazard_keys:
                selected.append({
                    "video_id":     vid_id,
                    "phase":        "phase1",
                    "category":     category,
                    "hazard_key":   hazard_key,
                    "frame_path":   frame["frame_path"],
                    "frame_file":   frame["frame_filename"],
                    "timestamp":    frame["timestamp_sec"],
                    "mode":         "generic",
                    "notes":        meta.get("filename", ""),
                })

    return selected


def select_phase2_frames() -> list:
    """
    Select frames from SDI MKV files based on visual tags.
    Falls back to evenly-spaced selection if no tags defined.
    """
    all_frames = load_frames_index()
    metadata = {r["video_id"]: r for r in load_master_metadata()}

    selected = []

    by_video = {}
    for frame in all_frames:
        vid = frame["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(frame)

    sdi_video_ids = ["VID_0023", "VID_0024", "VID_0025", "VID_0026"]

    for vid_id in sdi_video_ids:
        tags = SDI_FRAME_TAGS.get(vid_id, [])
        meta = metadata.get(vid_id, {})

        # Map VID_00xx to video stem
        vid_stem_map = {
            "VID_0023": "video-1", "VID_0024": "video-2",
            "VID_0025": "video-3", "VID_0026": "video-4"
        }
        vid_stem = vid_stem_map.get(vid_id, "")
        frames_dir = str(ROOT / "your_mkv_files" / "frames")

        if tags:
            # Use manually tagged frames — match to ffmpeg-extracted frames by timestamp
            for ts_sec, hazard_key, note in tags:
                frame_file = f"{vid_stem}_t{float(ts_sec):07.2f}s.jpg"
                frame_path = os.path.join(frames_dir, frame_file)
                if not os.path.exists(frame_path):
                    # Try closest available frame
                    import glob as _glob
                    candidates = _glob.glob(os.path.join(frames_dir, f"{vid_stem}_t*.jpg"))
                    if candidates:
                        frame_path = min(candidates, key=lambda p: abs(
                            float(Path(p).stem.split("_t")[1].replace("s","")) - float(ts_sec)))
                        frame_file = Path(frame_path).name
                    else:
                        continue
                closest = {"timestamp_sec": ts_sec, "frame_path": frame_path, "frame_filename": frame_file}
                selected.append({
                    "video_id":     vid_id,
                    "phase":        "phase2",
                    "category":     hazard_key,
                    "hazard_key":   hazard_key,
                    "frame_path":   closest["frame_path"],
                    "frame_file":   closest["frame_filename"],
                    "timestamp":    closest["timestamp_sec"],
                    "mode":         "sdi",
                    "notes":        note,
                })
        else:
            # Fallback — select 5 evenly-spaced frames per video
            # and test against all Phase 2 SDI hazards
            from hazard_library import list_hazards
            sdi_hazards = list_hazards(phase=2)

            step = max(1, len(frames) // 5)
            chosen = frames[::step][:5]

            for frame in chosen:
                for hazard_key in sdi_hazards:
                    selected.append({
                        "video_id":     vid_id,
                        "phase":        "phase2",
                        "category":     "unknown",
                        "hazard_key":   hazard_key,
                        "frame_path":   frame["frame_path"],
                        "frame_file":   frame["frame_filename"],
                        "timestamp":    frame["timestamp_sec"],
                        "mode":         "sdi",
                        "notes":        f"auto-selected, no manual tags for {vid_id}",
                    })

    return selected


def build_experiment_queue(output_path: str = None) -> list:
    """
    Build the full experiment queue — all frames × all prompt variants.
    Saves to experiment_queue.csv if output_path provided.

    Returns list of dicts, each representing one VLM inference call.
    """
    from prompt_templates import VARIANTS

    phase1_frames = select_phase1_frames()
    phase2_frames = select_phase2_frames()
    all_selected = phase1_frames + phase2_frames

    queue = []
    for entry in all_selected:
        for variant in VARIANTS.keys():
            # Skip domain/cot/multi_q without hazard key
            if variant != "direct" and not entry.get("hazard_key"):
                continue

            queue.append({
                "video_id":     entry["video_id"],
                "phase":        entry["phase"],
                "category":     entry["category"],
                "hazard_key":   entry.get("hazard_key", ""),
                "frame_path":   entry["frame_path"],
                "frame_file":   entry["frame_file"],
                "timestamp":    entry["timestamp"],
                "mode":         entry["mode"],
                "variant":      variant,
                "notes":        entry.get("notes", ""),
                # VLM result columns (filled in by inference script)
                "vlm_model":        "",
                "vlm_response":     "",
                "hazard_detected":  "",
                "severity":         "",
                "safe_or_unsafe":   "",
                "confidence":       "",
                "reason":           "",
                "recommended_action": "",
                "inference_time_sec": "",
                "run_timestamp":    "",
            })

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if queue:
                writer = csv.DictWriter(f, fieldnames=queue[0].keys())
                writer.writeheader()
                writer.writerows(queue)
        print(f"Saved {len(queue)} experiment rows to {output_path}")

    return queue


if __name__ == "__main__":
    output = str(ROOT / "experiment_queue.csv")
    queue = build_experiment_queue(output_path=output)

    # Summary
    from collections import Counter
    phases = Counter(r["phase"] for r in queue)
    variants = Counter(r["variant"] for r in queue)
    modes = Counter(r["mode"] for r in queue)

    print(f"\nExperiment Queue Summary")
    print("=" * 40)
    print(f"Total inference calls : {len(queue)}")
    print(f"\nBy phase:")
    for k, v in sorted(phases.items()):
        print(f"  {k:<15} {v}")
    print(f"\nBy prompt variant:")
    for k, v in sorted(variants.items()):
        print(f"  {k:<15} {v}")
    print(f"\nBy mode:")
    for k, v in sorted(modes.items()):
        print(f"  {k:<15} {v}")
    print(f"\nFile saved: {output}")
