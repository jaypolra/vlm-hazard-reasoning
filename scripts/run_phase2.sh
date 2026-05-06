#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_phase2.sh — Phase 2 End-to-End Pipeline
# SDI Butler industrial footage domain gap experiment
#
# Usage:
#   bash scripts/run_phase2.sh
#   bash scripts/run_phase2.sh --model gemini    # Use Gemini API
#   bash scripts/run_phase2.sh --limit 5         # Quick test (5 rows)
#
# Prerequisites:
#   Phase 1 must be complete first (run_phase1.sh) OR frames_index.csv exists
#   SDI footage placed in your_mkv_files/raw/:
#     your_mkv_files/raw/video-1.mkv
#     your_mkv_files/raw/video-2.mkv
#     your_mkv_files/raw/video-3.mkv
#   For Gemini: export GOOGLE_API_KEY=your_key
#   For GPT-4o: export OPENAI_API_KEY=your_key
#   For vLLM:   conda activate gemmaenv && vllm serve google/gemma-3-27b-it --port 8000
#
#   NOTE: SDI footage is proprietary and not included in this repo.
#         Contact CIVS at Purdue University Northwest for access information.
#
# Output:
#   your_mkv_files/frames/        ← extracted plant footage frames
#   experiment_queue.csv          ← updated with Phase 2 rows
#   results/report.html           ← updated analysis with Phase 2 results
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

MODEL="${1:---model vllm}"
EXTRA_ARGS="${@:2}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ARIA — Phase 2 Pipeline (SDI Industrial Footage)"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check for SDI footage
MKV_DIR="$ROOT/your_mkv_files/raw"
if [ ! -f "$MKV_DIR/video-1.mkv" ] && [ ! -f "$MKV_DIR/video-2.mkv" ]; then
    echo "⚠️  WARNING: SDI footage not found in $MKV_DIR"
    echo "   Place video-1.mkv, video-2.mkv, video-3.mkv there to proceed."
    echo "   (SDI footage is proprietary and not included in this repo.)"
    exit 1
fi
echo "✓ SDI footage found"
echo ""

# Step 1 — Extract frames from MKV files at 1fps
echo "── Step 1: Extract frames from SDI MKV files ───────────"
python 03_extract_frames.py --source mkv
echo "✓ Done"
echo ""

# Step 2 — Rebuild metadata index (adds Phase 2 entries)
echo "── Step 2: Rebuild metadata index ──────────────────────"
python 04_build_metadata.py
echo "✓ Done"
echo ""

# Step 3 — Rebuild experiment queue (adds Phase 2 rows)
echo "── Step 3: Rebuild experiment queue ────────────────────"
python -c "
import sys; sys.path.insert(0, 'prompts')
from frame_selector import build_experiment_queue
q = build_experiment_queue('experiment_queue.csv')
p2 = sum(1 for r in q if r['phase'] == 'phase2')
print(f'  Phase 2 rows: {p2}')
"
echo "✓ Done"
echo ""

# Step 4 — Run VLM inference (Phase 2 only, with resume support)
echo "── Step 4: VLM inference on Phase 2 (industrial) frames ─"
echo "   Model: $MODEL  |  Extra args: ${EXTRA_ARGS:-none}"
echo ""
echo "   This is the domain gap experiment:"
echo "   → 'direct' variant: model sees the plant frame cold (no context)"
echo "   → 'domain' variant: model gets SDI plant context injected"
echo ""
python 05_vlm_inference.py $MODEL --phase 2 --resume $EXTRA_ARGS
echo "✓ Done"
echo ""

# Step 5 — Generate updated results report
echo "── Step 5: Analyse results (Phase 1 + Phase 2 combined) ─"
python 06_results_analysis.py
echo "✓ Done"
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  Phase 2 complete!"
echo "  Open results/report.html to see the domain gap."
echo "  Launch the demo: streamlit run vlm_demo/app.py"
echo "═══════════════════════════════════════════════════════"
