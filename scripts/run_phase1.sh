#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_phase1.sh — Phase 1 End-to-End Pipeline
# Open-source YouTube safety videos baseline experiment
#
# Usage:
#   bash scripts/run_phase1.sh
#   bash scripts/run_phase1.sh --model gemini    # Use Gemini API instead
#   bash scripts/run_phase1.sh --limit 20        # Quick test (20 rows)
#
# Prerequisites:
#   pip install -r requirements.txt
#   yt-dlp must be installed (included in requirements.txt)
#   For Gemini: export GOOGLE_API_KEY=your_key
#   For GPT-4o: export OPENAI_API_KEY=your_key
#   For vLLM:   conda activate gemmaenv && vllm serve google/gemma-3-27b-it --port 8000
#
# Output:
#   phase1_opensource/*/frames/   ← extracted video frames
#   experiment_queue.csv          ← one row per (frame × prompt variant)
#   results/report.html           ← analysis report
#   plots/                        ← charts
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

MODEL="${1:---model vllm}"
EXTRA_ARGS="${@:2}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ARIA — Phase 1 Pipeline (YouTube Safety Baseline)"
echo "═══════════════════════════════════════════════════════"
echo ""

# Step 1 — Download YouTube hazard videos
echo "── Step 1: Download YouTube hazard videos ──────────────"
python 02_download_youtube.py
echo "✓ Done"
echo ""

# Step 2 — Extract frames at 1fps (max 30 per video)
echo "── Step 2: Extract frames from videos ──────────────────"
python 03_extract_frames.py
echo "✓ Done"
echo ""

# Step 3 — Build metadata index
echo "── Step 3: Build metadata index ────────────────────────"
python 04_build_metadata.py
echo "✓ Done"
echo ""

# Step 4 — Build experiment queue (frame × variant combinations)
echo "── Step 4: Build experiment queue ──────────────────────"
python -c "
import sys; sys.path.insert(0, 'prompts')
from frame_selector import build_experiment_queue
q = build_experiment_queue('experiment_queue.csv')
p1 = sum(1 for r in q if r['phase'] == 'phase1')
print(f'  Phase 1 rows: {p1}')
"
echo "✓ Done"
echo ""

# Step 5 — Run VLM inference (Phase 1 only, with resume support)
echo "── Step 5: VLM inference on Phase 1 frames ─────────────"
echo "   Model: $MODEL  |  Extra args: ${EXTRA_ARGS:-none}"
python 05_vlm_inference.py $MODEL --phase 1 --resume $EXTRA_ARGS
echo "✓ Done"
echo ""

# Step 6 — Generate results report
echo "── Step 6: Analyse results ──────────────────────────────"
python 06_results_analysis.py
echo "✓ Done"
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  Phase 1 complete! Open results/report.html to view."
echo "═══════════════════════════════════════════════════════"
