"""
ARIA — AI Risk & Incident Analyzer
VLM Reasoning Layer for Industrial Hazard Recognition
CIVS × Purdue University Northwest · SDI Butler Steel Plant

Results analysis script — aggregates experiment_queue.csv into:
  - Detection rate tables by phase × variant
  - Severity distribution charts
  - Latency boxplots
  - HTML summary report

Usage:
  python 06_results_analysis.py
  # Reads experiment_queue.csv from the same directory
  # Writes charts to plots/ and report to results/report.html

Part of the research:
  "Bridging the Domain Gap in Industrial Safety with Vision Language Models"
  CIVS × Purdue University Northwest for SDI Butler Steel Plant
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Setup
INPUT_CSV = "experiment_queue.csv"
OUTPUT_DIR = Path("results")
PLOTS_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)

    total_rows = len(df)

    # Filter to only processed rows (non-empty vlm_response)
    df_processed = df[df['vlm_response'].notna() & (df['vlm_response'].str.strip() != '')].copy()

    # Separate error rows for reporting, exclude from analysis
    df_errors = df_processed[df_processed['hazard_detected'].str.upper() == 'ERROR'].copy()
    df_runs   = df_processed[df_processed['hazard_detected'].str.upper() != 'ERROR'].copy()

    if len(df_runs) == 0:
        print("No successful results found yet. All rows are errors or unprocessed.")
        if len(df_errors) > 0:
            print(f"  {len(df_errors)} ERROR rows (frame not found / inference crash)")
            print("  Re-run inference with: python -X utf8 05_vlm_inference.py --model gemma4 --resume")
        return

    done = len(df_runs)
    pct = done / total_rows * 100
    print(f"\n{'='*50}")
    print(f"  PROGRESS: {done}/{total_rows} rows successful ({pct:.1f}%)")
    if len(df_errors) > 0:
        print(f"  ERROR rows (excluded from analysis): {len(df_errors)}")
    phase1_done = len(df_runs[df_runs['phase'] == 'phase1'])
    phase2_done = len(df_runs[df_runs['phase'] == 'phase2'])
    print(f"  Phase 1: {phase1_done} rows | Phase 2: {phase2_done} rows")
    print(f"{'='*50}\n")

    # Data Cleaning
    df_runs['hazard_detected'] = df_runs['hazard_detected'].str.upper().str.strip('* ')
    df_runs['is_detected'] = df_runs['hazard_detected'] == 'YES'
    
    # 1. Overall Metrics
    summary = {
        "total_processed": len(df_runs),
        "overall_detection_rate": df_runs['is_detected'].mean(),
        "avg_inference_time": df_runs['inference_time_sec'].mean()
    }
    
    # 2. Performance by Variant (Prompt Engineering Test)
    variant_perf = df_runs.groupby('variant').agg({
        'is_detected': 'mean',
        'inference_time_sec': 'mean',
        'confidence': lambda x: (x == 'HIGH').mean() # Percentage of high confidence
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=variant_perf, x='variant', y='is_detected',
                hue='variant', palette='viridis', legend=False)
    plt.title('Hazard Detection Rate by Prompt Variant')
    plt.ylabel('Detection Rate (Accuracy)')
    plt.savefig(PLOTS_DIR / 'detection_by_variant.png')
    plt.close()

    # 3. Phase 1 vs Phase 2 (Domain Gap Test)
    phase_gap = df_runs.groupby(['phase', 'variant'])['is_detected'].mean().unstack()
    
    plt.figure(figsize=(12, 7))
    phase_gap.plot(kind='bar')
    plt.title('The Domain Gap: Phase 1 (YouTube) vs Phase 2 (SDI Plant)')
    plt.ylabel('Detection Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Prompt Variant')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'domain_gap_analysis.png')
    plt.close()

    # 4. Severity Distribution
    severity_dist = df_runs['severity'].value_counts()
    plt.figure(figsize=(8, 8))
    severity_dist.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Reds_r'))
    plt.title('Distribution of Detected Hazard Severities')
    plt.savefig(PLOTS_DIR / 'severity_pie.png')
    plt.close()

    # 5. Timing Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_runs, x='variant', y='inference_time_sec')
    plt.title('Inference Latency by Variant (Speed vs Quality)')
    plt.savefig(PLOTS_DIR / 'latency_boxplot.png')
    plt.close()

    # Save CSV summaries
    variant_perf.to_csv(OUTPUT_DIR / "variant_performance.csv", index=False)
    
    # Create a HTML Summary Report
    with open(OUTPUT_DIR / "report.html", "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>VLM Safety Analysis Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                img {{ max-width: 800px; margin: 20px 0; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f4f4f4; }}
                .stat-box {{ display: flex; gap: 20px; margin-bottom: 30px; }}
                .stat {{ background: #eef; padding: 20px; border-radius: 8px; flex: 1; text-align: center; }}
                .stat h2 {{ margin: 0; color: #0056b3; }}
            </style>
        </head>
        <body>
            <h1>VLM Hazard Recognition - Experiment Summary</h1>
            <div class="stat-box">
                <div class="stat"><h2>{summary['total_processed']}</h2><p>Rows Processed</p></div>
                <div class="stat"><h2>{summary['overall_detection_rate']:.1%}</h2><p>Overall Detection Rate</p></div>
                <div class="stat"><h2>{summary['avg_inference_time']:.2f}s</h2><p>Avg Latency</p></div>
            </div>
            
            <h2>1. Accuracy by Prompt Variant</h2>
            <p>This chart shows which prompt engineering strategy was most effective.</p>
            <img src="../plots/detection_by_variant.png" />
            
            <h2>2. Domain Gap Analysis</h2>
            <p>Phase 1 contains general YouTube safety videos. Phase 2 contains real SDI Steel Plant footage. 
               The difference in performance indicates the "Domain Gap".</p>
            <img src="../plots/domain_gap_analysis.png" />
            
            <h2>3. Severity Breakdown</h2>
            <img src="../plots/severity_pie.png" />
            
            <h2>4. Latency Analysis</h2>
            <img src="../plots/latency_boxplot.png" />
        </body>
        </html>
        """)

    print(f"\nSuccess! Reports generated in '{OUTPUT_DIR}' and plots in '{PLOTS_DIR}'.")
    print(f"Open '{OUTPUT_DIR}/report.html' to view the full analysis.")

if __name__ == "__main__":
    main()
