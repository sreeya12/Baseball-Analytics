#!/usr/bin/env python3
"""
generate_shap_plots.py — Standalone SHAP plot generator for Diamond IQ

Generates:
  1. Global SHAP summary (beeswarm) plot for all pitchers
  2. Individual SHAP bar plots for each elite-category pitcher

Each plot is saved as a separate PNG, ready for email review or
PowerPoint placement.

Usage:
    python generate_shap_plots.py \
        --model  outputs/strike_pct_model_lgbm.pkl \
        --data   outputs/gmu_roster_session_data.csv \
        --outdir outputs/shap_plots

Requirements:
    pip install lightgbm shap scikit-learn matplotlib pandas numpy
"""

import argparse
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import shap
try:
    import shap
except ImportError:
    print("ERROR: shap is not installed. Run: pip install shap")
    exit(1)


# ─────────────────────────────────────────────────────────────────────
# Elite pitchers (from Settings.py)
# ─────────────────────────────────────────────────────────────────────
ELITE_PITCHERS = [
    "Cassedy, Brandon",
    "Meeks, Gardner",
    "Lavin, Sam",
    "O'Hara, Connor",
    "Elliot, Daniel",
    "Knox, Connor",
    "Rumberg, Logan",
]

# ─────────────────────────────────────────────────────────────────────
# Coach-friendly display names
# ─────────────────────────────────────────────────────────────────────
FRIENDLY_NAMES = {
    "FB_pct":                "Fastball Usage %",
    "DaysSinceLast":         "Days Since Last Outing",
    "RelSide_Avg":           "Release Side Position",
    "PitchCount":            "Pitch Count",
    "SpinRate_FB":           "Fastball Spin Rate (RPM)",
    "HorzApprAngle_Avg":     "Horizontal Approach Angle",
    "Zone_pct_lag1":         "Prev Session Zone %",
    "PitchTypes":            "# Pitch Types Used",
    "RelHeight_Avg":         "Release Height",
    "Zone_pct_roll3_cov":    "Zone % Consistency",
    "SpinRate_Avg_lag1":     "Prev Session Spin Rate",
    "SpinAxis_Avg":          "Spin Axis",
    "HorzBreak_Avg":         "Horizontal Break",
    "Extension_Avg":         "Extension",
    "pitcher_prior_strike_pct": "Career Strike % Prior",
    "SpinRate_Avg":          "Spin Rate",
    "Whiff_pct_roll3_cov":   "Whiff % Consistency",
    "FB_Velo_Max":           "Max Fastball Velocity",
    "Strike_pct_roll3_cov":  "Strike % Consistency",
    "IVB_Avg":               "Induced Vertical Break",
    "SpinRate_Avg_roll3":    "3-Session Avg Spin Rate",
    "FB_Velo_Avg_lag1":      "Prev Session FB Velo",
    "VertApprAngle_Avg":     "Vertical Approach Angle",
    "VeloFade":              "Velocity Fade",
    "FB_Velo_Avg":           "Avg Fastball Velocity",
    "Strike_pct_lag1":       "Prev Session Strike %",
    "VertBreak_Avg":         "Vertical Break",
    "Whiff_pct_lag1":        "Prev Session Whiff %",
    "EffVelo_Avg":           "Effective Velocity",
    "RelSpeed_Avg":          "Avg Release Speed",
    "FB_Velo_Avg_roll3":     "3-Session Avg FB Velo",
    # CMJ features
    "CMJ_Jump Height (Imp-Mom) in Inches [in]": "CMJ Jump Height",
    "CMJ_Peak Power / BM [W/kg]":               "CMJ Peak Power / BW",
    "CMJ_RSI-modified (Imp-Mom) [m/s]":         "CMJ Reactive Strength",
    "CMJ_Countermovement Depth [cm]":           "CMJ Counter. Depth",
    "CMJ_Eccentric:Concentric Mean Force Ratio [%]": "CMJ Ecc:Con Ratio",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP plots (global + elite pitchers)",
    )
    parser.add_argument("--model", required=True, help="Path to strike_pct_model_lgbm.pkl")
    parser.add_argument("--data", required=True, help="Path to gmu_roster_session_data.csv")
    parser.add_argument("--outdir", default="shap_plots", help="Output directory for PNGs")
    parser.add_argument("--top-n", type=int, default=15, help="Features per pitcher plot")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load model ──
    print(f"Loading model from {args.model} ...")
    with open(args.model, "rb") as f:
        pkg = pickle.load(f)

    model = pkg["model"]
    scaler = pkg["scaler"]
    feature_cols = pkg["feature_cols"]

    # ── Load data ──
    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    available = [c for c in feature_cols if c in df.columns]

    X_raw = df[available].values.astype(np.float64)
    X_scaled = scaler.transform(X_raw)
    X_df = pd.DataFrame(X_scaled, columns=available)

    # ── Compute SHAP values ──
    print("Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    print(f"  SHAP matrix: {shap_values.shape}")

    # ── 1. Global SHAP summary ──
    print("\nGenerating global SHAP summary ...")
    X_display = X_df.rename(columns=FRIENDLY_NAMES)

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_display, show=False, max_display=20)
    plt.title(
        "Global SHAP Summary — What Drives Strike % Across All Pitchers",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    path = os.path.join(args.outdir, "shap_global_summary.png")
    plt.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # ── 2. Individual elite pitcher plots ──
    print("\nGenerating individual elite pitcher SHAP plots ...")

    for pitcher in ELITE_PITCHERS:
        pitcher_mask = df["Pitcher"] == pitcher
        if pitcher_mask.sum() == 0:
            print(f"  WARNING: No data found for {pitcher} — skipping")
            continue

        pitcher_indices = np.where(pitcher_mask)[0]
        pitcher_shap = shap_values[pitcher_indices]

        # Mean absolute SHAP for this pitcher — top N features
        mean_abs_shap = np.abs(pitcher_shap).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-args.top_n:][::-1]

        fig, ax = plt.subplots(figsize=(11, 8))

        top_names = [FRIENDLY_NAMES.get(available[i], available[i]) for i in top_idx]
        top_vals = [mean_abs_shap[i] * 100 for i in top_idx]  # convert to pp

        # Signed mean SHAP for color (helps vs hurts)
        mean_shap_signed = pitcher_shap.mean(axis=0)
        colors = ["#2E86AB" if mean_shap_signed[i] >= 0 else "#E74C3C" for i in top_idx]

        ax.barh(range(len(top_names)), top_vals[::-1],
                color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP value| (percentage points effect on Strike %)", fontsize=10)

        # Title with pitcher stats
        pitcher_strike = df.loc[pitcher_mask, "Strike_pct"].mean() * 100
        n_sessions = pitcher_mask.sum()
        ax.set_title(
            f"{pitcher}\nAvg Strike %: {pitcher_strike:.1f}%  |  Sessions: {n_sessions}",
            fontsize=13, fontweight="bold",
        )

        # Legend
        legend_elements = [
            Patch(facecolor="#2E86AB", label="Helps Strike %"),
            Patch(facecolor="#E74C3C", label="Hurts Strike %"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()

        safe_name = pitcher.split(",")[0].replace("'", "").replace(" ", "_")
        filename = f"shap_{safe_name}.png"
        fig.savefig(os.path.join(args.outdir, filename), dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}  ({n_sessions} sessions, avg {pitcher_strike:.1f}%)")

    print(f"\nDone! All SHAP plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()