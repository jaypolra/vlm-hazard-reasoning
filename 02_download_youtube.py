"""
02_download_youtube.py
----------------------
Downloads curated YouTube videos for VLM hazard recognition experiment.
Run from inside the vlm_hazard_dataset/ folder.

Usage:
    python 02_download_youtube.py
    python 02_download_youtube.py --category PPE          # single category
    python 02_download_youtube.py --phase 1               # single phase
    python 02_download_youtube.py --dry-run               # preview only
"""

import subprocess
import os
import argparse
import json
from datetime import datetime

# ─────────────────────────────────────────────
# CURATED VIDEO LIST
# Add/remove URLs here. Each entry has:
#   url      : YouTube URL
#   title    : short descriptive name (used as filename)
#   category : PPE_violation | fall_from_height | machinery_danger | near_miss
#   phase    : 1 (clear/generic) or 2 (industrial/factory)
#   notes    : why this video was chosen
# ─────────────────────────────────────────────
VIDEOS = [

    # ── PHASE 1 — PPE Violation ──────────────────────────────────────────
    {
        "url": "ytsearch1:construction worker no hard hat helmet PPE violation site safety",
        "title": "ppe_violation_no_helmet_construction",
        "category": "PPE_violation",
        "phase": 1,
        "notes": "Worker without helmet on active construction site"
    },
    {
        "url": "ytsearch1:OSHA PPE safety violations workplace examples training video",
        "title": "ppe_osha_violations_compilation",
        "category": "PPE_violation",
        "phase": 1,
        "notes": "OSHA safety violation examples — multiple PPE issues"
    },
    {
        "url": "ytsearch1:no high visibility vest warehouse worker safety violation",
        "title": "ppe_no_vest_warehouse",
        "category": "PPE_violation",
        "phase": 1,
        "notes": "No high-vis vest in active warehouse"
    },
    {
        "url": "ytsearch1:construction site PPE safety violations training video helmet gloves",
        "title": "ppe_construction_safety_violations_training",
        "category": "PPE_violation",
        "phase": 1,
        "notes": "Safety training video — clear PPE violations with narration"
    },
    {
        "url": "ytsearch1:worker no fall harness working at height safety violation",
        "title": "ppe_harness_violation_height",
        "category": "PPE_violation",
        "phase": 1,
        "notes": "No fall harness while working at height"
    },

    # ── PHASE 1 — Fall from Height ──────────────────────────────────────
    {
        "url": "ytsearch1:scaffolding fall near miss construction worker caught on camera",
        "title": "fall_scaffolding_near_miss",
        "category": "fall_from_height",
        "phase": 1,
        "notes": "Scaffolding near-miss fall — clearly visible"
    },
    {
        "url": "ytsearch1:construction worker fall CCTV footage workplace accident",
        "title": "fall_construction_cctv",
        "category": "fall_from_height",
        "phase": 1,
        "notes": "CCTV footage of construction fall incident"
    },
    {
        "url": "ytsearch1:ladder fall workplace accident worker safety",
        "title": "fall_ladder_workplace",
        "category": "fall_from_height",
        "phase": 1,
        "notes": "Ladder fall in workplace setting"
    },
    {
        "url": "ytsearch1:fall from height safety training video hazard prevention annotated",
        "title": "fall_height_safety_training",
        "category": "fall_from_height",
        "phase": 1,
        "notes": "Fall from height safety training — annotated hazards"
    },
    {
        "url": "ytsearch1:rooftop worker fall hazard incident safety video",
        "title": "fall_roof_worker_incident",
        "category": "fall_from_height",
        "phase": 1,
        "notes": "Rooftop worker fall hazard scenario"
    },

    # ── PHASE 1 — Machinery Danger ───────────────────────────────────────
    {
        "url": "ytsearch1:unguarded rotating machinery workplace hazard safety training",
        "title": "machinery_unguarded_equipment",
        "category": "machinery_danger",
        "phase": 1,
        "notes": "Unguarded rotating machinery — classic hazard"
    },
    {
        "url": "ytsearch1:forklift near miss pedestrian warehouse CCTV footage",
        "title": "machinery_forklift_near_miss",
        "category": "machinery_danger",
        "phase": 1,
        "notes": "Forklift near-miss with pedestrian"
    },
    {
        "url": "ytsearch1:worker near conveyor belt entanglement machinery danger safety",
        "title": "machinery_caught_in_conveyor",
        "category": "machinery_danger",
        "phase": 1,
        "notes": "Worker near conveyor belt — machinery entanglement risk"
    },
    {
        "url": "ytsearch1:lockout tagout LOTO violation worker near active machine maintenance safety",
        "title": "machinery_lockout_tagout_violation",
        "category": "machinery_danger",
        "phase": 1,
        "notes": "LOTO violation — worker near active machine during maintenance"
    },
    {
        "url": "ytsearch1:machine guarding safety training guards removed hazard workplace",
        "title": "machinery_safety_training_guards",
        "category": "machinery_danger",
        "phase": 1,
        "notes": "Machine guarding training — shows both safe and unsafe states"
    },

    # ── PHASE 1 — Near-Miss ──────────────────────────────────────────────
    {
        "url": "ytsearch1:warehouse forklift near miss CCTV pedestrian close call",
        "title": "near_miss_warehouse_forklift",
        "category": "near_miss",
        "phase": 1,
        "notes": "Warehouse forklift near-miss CCTV — temporal hazard"
    },
    {
        "url": "ytsearch1:falling object near miss construction site caught on camera",
        "title": "near_miss_falling_object",
        "category": "near_miss",
        "phase": 1,
        "notes": "Falling object near-miss on construction site"
    },
    {
        "url": "ytsearch1:electrical near miss worker live panel arc flash safety incident",
        "title": "near_miss_electrical_hazard",
        "category": "near_miss",
        "phase": 1,
        "notes": "Electrical near-miss — worker approaches live panel"
    },
    {
        "url": "ytsearch1:vehicle pedestrian near miss industrial yard close call safety",
        "title": "near_miss_vehicle_pedestrian",
        "category": "near_miss",
        "phase": 1,
        "notes": "Vehicle-pedestrian near-miss in industrial yard"
    },
    {
        "url": "ytsearch1:workplace near miss compilation safety incidents caught camera",
        "title": "near_miss_compilation_safety",
        "category": "near_miss",
        "phase": 1,
        "notes": "Near-miss compilation — varied hazard types"
    },

    # ── PHASE 2 — Industrial / Factory Setting ───────────────────────────
    # These are harder for VLM — factory context, poor lighting, no narration
    {
        "url": "ytsearch1:factory floor CCTV unsafe behaviour industrial workers no annotation",
        "title": "industrial_factory_cctv_unsafe_behaviour",
        "category": "machinery_danger",
        "phase": 2,
        "notes": "Real factory CCTV — unsafe behaviour, no annotation"
    },
    {
        "url": "ytsearch1:manufacturing plant PPE audit inspection safety violation subtle",
        "title": "industrial_manufacturing_ppe_check",
        "category": "PPE_violation",
        "phase": 2,
        "notes": "Manufacturing plant PPE audit — subtle violations"
    },
    {
        "url": "ytsearch1:forklift factory floor industrial operation proximity hazard real",
        "title": "industrial_forklift_factory_floor",
        "category": "near_miss",
        "phase": 2,
        "notes": "Forklift on factory floor — proximity hazard, real industrial env"
    },
]


def download_video(video: dict, output_dir: str, dry_run: bool = False):
    """Download a single video using yt-dlp."""
    category = video["category"]
    phase = video["phase"]
    title = video["title"]
    url = video["url"]

    out_folder = os.path.join(
        output_dir,
        f"phase{phase}_opensource" if phase == 1 else "phase2_industrial",
        category,
        "raw"
    )
    os.makedirs(out_folder, exist_ok=True)

    out_path = os.path.join(out_folder, f"{title}.%(ext)s")

    print(f"\n{'[DRY RUN] Would download' if dry_run else 'Downloading'}: {title}")
    print(f"  URL     : {url}")
    print(f"  Folder  : {out_folder}")
    print(f"  Notes   : {video['notes']}")

    if dry_run:
        return True

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",  # 720p max — enough for VLM
        "--merge-output-format", "mp4",
        "--output", out_path,
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "3",
        "--js-runtimes", "node",   # required for YouTube n-challenge solving
        "--cookies", "youtube_cookies.txt",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  [OK] Downloaded successfully")
            return True
        else:
            print(f"  [FAIL] yt-dlp error: {result.stderr[-200:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Timeout after 5 minutes")
        return False
    except FileNotFoundError:
        print(f"  [ERROR] yt-dlp not found. Run: pip install yt-dlp")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download VLM hazard experiment videos")
    parser.add_argument("--category", choices=["PPE_violation", "fall_from_height", "machinery_danger", "near_miss"],
                        help="Download only this category")
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="Download only this phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be downloaded without downloading")
    parser.add_argument("--output", default=".", help="Root dataset folder (default: current dir)")
    args = parser.parse_args()

    videos = VIDEOS
    if args.category:
        videos = [v for v in videos if v["category"] == args.category]
    if args.phase:
        videos = [v for v in videos if v["phase"] == args.phase]

    print(f"{'='*50}")
    print(f" VLM Hazard Dataset — YouTube Downloader")
    print(f"{'='*50}")
    print(f" Videos to download : {len(videos)}")
    print(f" Output folder      : {os.path.abspath(args.output)}")
    print(f" Mode               : {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    print(f"{'='*50}")

    results = {"success": [], "failed": []}
    for video in videos:
        ok = download_video(video, args.output, dry_run=args.dry_run)
        (results["success"] if ok else results["failed"]).append(video["title"])

    print(f"\n{'='*50}")
    print(f" Summary")
    print(f"{'='*50}")
    print(f" Succeeded : {len(results['success'])}")
    print(f" Failed    : {len(results['failed'])}")
    if results["failed"]:
        print(f" Failed videos:")
        for t in results["failed"]:
            print(f"   - {t}")
    print(f"\n Next step: Run 03_extract_frames.py")


if __name__ == "__main__":
    main()
