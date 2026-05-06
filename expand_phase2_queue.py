"""
expand_phase2_queue.py
Adds all unqueued MKV frames from your_mkv_files/frames/ to experiment_queue.csv.
Each new unique frame gets 4 rows (one per variant).
Category is assigned based on video source + timestamp heuristics.
"""

import csv
import re
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
QUEUE_CSV    = SCRIPT_DIR / "experiment_queue.csv"
FRAMES_DIR   = SCRIPT_DIR / "your_mkv_files" / "frames"
VARIANTS     = ["direct", "cot", "domain", "multi_q"]

# SDI-specific category heuristics per video source.
# Based on known footage content from AGENT_BRIEFING.md:
#   video-1: main bay, pot hauler activity throughout
#   video-2: hero camera — both vehicles + glowing pot (~t=200s), general bay
#   video-3: Camera 3, covers blind spot — vehicle enters at ~t=100s, re-emerges with pot at ~t=204s
#   video-4: additional plant footage
def assign_category(video: str, timestamp: float) -> str:
    if video == "video-3":
        if 90 <= timestamp <= 130:
            return "blind_spot_vehicle"
        if 190 <= timestamp <= 220:
            return "people_with_active_equipment"
    if video == "video-2":
        if 195 <= timestamp <= 225:
            return "people_with_active_equipment"
    # Default for all other industrial plant footage
    return "industrial_scene"


def parse_timestamp(filename: str) -> float:
    m = re.search(r"_t(\d+\.\d+)s", filename)
    return float(m.group(1)) if m else 0.0


def main():
    # Load existing queue
    with open(QUEUE_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Frames already in the Phase 2 queue
    queued_frames = set(
        r["frame_file"] for r in rows if r["phase"] == "phase2"
    )

    # Next VID number
    p2_ids = [
        int(r["video_id"].replace("VID_", ""))
        for r in rows if r["phase"] == "phase2"
    ]
    next_vid = max(p2_ids) + 1 if p2_ids else 27

    # Collect new frames
    all_frames = sorted(FRAMES_DIR.glob("*.jpg"))
    new_frames = [
        f for f in all_frames
        if f.name not in queued_frames
        and not f.name.startswith("test_")   # skip test frames
    ]

    print(f"Frames already queued : {len(queued_frames)}")
    print(f"New frames to add     : {len(new_frames)}")
    print(f"New rows (×4 variants): {len(new_frames) * 4}")
    print()

    new_rows = []
    for frame_path in new_frames:
        fname    = frame_path.name
        # Derive source video name (video-1, video-2, …)
        video    = re.match(r"(video-\d+)", fname).group(1) if re.match(r"video-\d+", fname) else "video-unknown"
        ts       = parse_timestamp(fname)
        category = assign_category(video, ts)
        vid_str  = f"VID_{next_vid:04d}"
        rel_path = f".\\your_mkv_files\\frames\\{fname}"

        for variant in VARIANTS:
            new_rows.append({
                "video_id"         : vid_str,
                "phase"            : "phase2",
                "category"         : category,
                "hazard_key"       : category,
                "frame_path"       : rel_path,
                "frame_file"       : fname,
                "timestamp"        : str(ts),
                "mode"             : "sdi",
                "variant"          : variant,
                "notes"            : f"Auto-added: {video} t={ts}s",
                "vlm_model"        : "",
                "vlm_response"     : "",
                "hazard_detected"  : "",
                "severity"         : "",
                "safe_or_unsafe"   : "",
                "confidence"       : "",
                "reason"           : "",
                "recommended_action": "",
                "inference_time_sec": "",
                "run_timestamp"    : "",
            })

        next_vid += 1

    # Append to existing rows and save
    all_rows = rows + new_rows
    tmp = QUEUE_CSV.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    tmp.replace(QUEUE_CSV)

    print(f"Done. Added {len(new_rows)} rows ({len(new_frames)} frames × 4 variants).")
    print(f"Total rows in queue: {len(all_rows)}")


if __name__ == "__main__":
    main()
