"""
03_extract_frames.py
--------------------
Extracts frames from all videos in the dataset at 1 fps (configurable).
Saves frames as JPG into the corresponding /frames/ folder.
Also generates a per-video frame manifest.

Usage:
    python 03_extract_frames.py                    # extract all
    python 03_extract_frames.py --fps 2            # 2 frames per second
    python 03_extract_frames.py --phase 1          # only phase 1
    python 03_extract_frames.py --max-frames 20    # cap frames per video
    python 03_extract_frames.py --video your_mkv_files/raw/video-1.mkv  # single file
"""

import cv2
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, output_folder: str, fps: float = 1.0,
                   max_frames: int = None, quality: int = 90) -> dict:
    """
    Extract frames from a video file.
    Returns a dict with extraction stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "error": f"Cannot open {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0

    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    os.makedirs(output_folder, exist_ok=True)

    video_name = Path(video_path).stem
    extracted = []
    frame_idx = 0
    saved_count = 0

    print(f"\n  Video  : {Path(video_path).name}")
    print(f"  FPS    : {video_fps:.1f} -> extracting every {frame_interval} frames ({fps} fps)")
    print(f"  Size   : {width}x{height}, duration: {duration_sec:.1f}s")

    pbar = tqdm(total=int(duration_sec * fps), desc="  Frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / video_fps
            frame_filename = f"{video_name}_t{timestamp_sec:07.2f}s_f{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)

            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted.append({
                "filename": frame_filename,
                "frame_index": frame_idx,
                "timestamp_sec": round(timestamp_sec, 2),
                "path": frame_path
            })
            saved_count += 1
            pbar.update(1)

            if max_frames and saved_count >= max_frames:
                print(f"\n  [Capped at {max_frames} frames]")
                break

        frame_idx += 1

    pbar.close()
    cap.release()

    manifest_path = os.path.join(output_folder, f"{video_name}_manifest.json")
    manifest = {
        "video_file": str(video_path),
        "video_name": video_name,
        "original_fps": round(video_fps, 2),
        "extracted_fps": fps,
        "total_video_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "resolution": f"{width}x{height}",
        "frames_extracted": saved_count,
        "frames": extracted
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved  : {saved_count} frames → {output_folder}")
    return {"status": "ok", **manifest}


def find_all_videos(root: str, phase_filter: int = None) -> list:
    """Walk dataset folder and find all video files in /raw/ folders."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    videos = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Only look inside /raw/ folders
        if not dirpath.endswith("raw") and "\\raw" not in dirpath and "/raw" not in dirpath:
            continue

        # Phase filter
        if phase_filter:
            if phase_filter == 1 and "phase2" in dirpath:
                continue
            if phase_filter == 2 and "phase1" in dirpath:
                continue

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in video_extensions:
                video_path = os.path.join(dirpath, fname)
                frames_dir = dirpath.replace(os.sep + "raw", os.sep + "frames").replace("/raw", "/frames")
                videos.append({
                    "video_path": video_path,
                    "frames_dir": frames_dir
                })

    return videos


def main():
    parser = argparse.ArgumentParser(description="Extract frames from hazard videos")
    parser.add_argument("--root", default=".", help="Root dataset folder")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max frames per video (default: 30 — enough for VLM)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100 (default: 90)")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Only process this phase")
    parser.add_argument("--video", help="Process a single video file directly")
    args = parser.parse_args()

    print("=" * 55)
    print(" VLM Hazard Dataset — Frame Extractor")
    print("=" * 55)
    print(f" FPS per video     : {args.fps}")
    print(f" Max frames/video  : {args.max_frames}")
    print(f" JPEG quality      : {args.quality}")
    print("=" * 55)

    if args.video:
        # Single video mode
        video_path = args.video
        frames_dir = str(Path(video_path).parent).replace("raw", "frames")
        if not os.path.exists(frames_dir):
            frames_dir = str(Path(video_path).parent / "frames_output")
        videos = [{"video_path": video_path, "frames_dir": frames_dir}]
    else:
        videos = find_all_videos(args.root, args.phase)

    if not videos:
        print("\n[!] No videos found in /raw/ folders.")
        print("    Make sure videos are placed inside the correct /raw/ directory.")
        return

    print(f"\n Found {len(videos)} video(s) to process\n")

    all_results = []
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {Path(v['video_path']).name}")
        result = extract_frames(
            v["video_path"],
            v["frames_dir"],
            fps=args.fps,
            max_frames=args.max_frames,
            quality=args.quality
        )
        all_results.append(result)

    # Summary
    success = [r for r in all_results if r.get("status") == "ok"]
    failed = [r for r in all_results if r.get("status") != "ok"]
    total_frames = sum(r.get("frames_extracted", 0) for r in success)

    print(f"\n{'='*55}")
    print(f" Done!")
    print(f" Videos processed  : {len(success)} / {len(all_results)}")
    print(f" Total frames saved: {total_frames}")
    if failed:
        print(f" Failed            : {len(failed)}")
        for r in failed:
            print(f"   - {r.get('video_file', '?')}: {r.get('error', '?')}")
    print(f"\n Next step: Run 04_build_metadata.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
