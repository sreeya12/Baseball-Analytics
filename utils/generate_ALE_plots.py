#!/usr/bin/env python3
"""
generate_ale_plots.py — Standalone ALE plot generator for Diamond IQ

Generates individual ALE (Accumulated Local Effects) plots for the top N
features from a saved LightGBM model. Each plot is saved as a separate PNG
with a coach-friendly title and one-sentence practical summary.

Usage:
    python generate_ale_plots.py \
        --model  outputs/strike_pct_model_lgbm.pkl \
        --data   outputs/gmu_roster_session_data.csv \
        --outdir outputs/ale_plots \
        --top-n  10

Requirements:
    pip install lightgbm scikit-learn matplotlib pandas numpy
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "Zone_pct_lag1":         "Previous Session Zone %",
    "PitchTypes":            "Number of Pitch Types Used",
    "RelHeight_Avg":         "Release Height",
    "Zone_pct_roll3_cov":    "Zone % Consistency (3-Session)",
    "SpinRate_Avg_lag1":     "Previous Session Spin Rate",
    "SpinAxis_Avg":          "Spin Axis",
    "HorzBreak_Avg":         "Horizontal Break",
    "Extension_Avg":         "Extension",
    "pitcher_prior_strike_pct": "Career Strike % Prior",
    "SpinRate_Avg":          "Spin Rate",
    "Whiff_pct_roll3_cov":   "Whiff % Consistency (3-Session)",
    "FB_Velo_Max":           "Max Fastball Velocity",
    "Strike_pct_roll3_cov":  "Strike % Consistency (3-Session)",
    "IVB_Avg":               "Induced Vertical Break",
    "SpinRate_Avg_roll3":    "3-Session Avg Spin Rate",
    "FB_Velo_Avg_lag1":      "Previous Session FB Velocity",
    "VertApprAngle_Avg":     "Vertical Approach Angle",
    "VeloFade":              "Velocity Fade",
    "FB_Velo_Avg":           "Avg Fastball Velocity",
    "Strike_pct_lag1":       "Previous Session Strike %",
    "VertBreak_Avg":         "Vertical Break",
    "Whiff_pct_lag1":        "Previous Session Whiff %",
    "EffVelo_Avg":           "Effective Velocity",
    "RelSpeed_Avg":          "Avg Release Speed",
    "FB_Velo_Avg_roll3":     "3-Session Avg FB Velocity",
    # CMJ features (for when they're in the model)
    "CMJ_Jump Height (Imp-Mom) in Inches [in]": "CMJ Jump Height (inches)",
    "CMJ_Peak Power / BM [W/kg]":               "CMJ Peak Power / Bodyweight",
    "CMJ_RSI-modified (Imp-Mom) [m/s]":         "CMJ Reactive Strength Index",
    "CMJ_Countermovement Depth [cm]":           "CMJ Countermovement Depth",
    "CMJ_Eccentric:Concentric Mean Force Ratio [%]": "CMJ Ecc:Con Force Ratio",
}

# ─────────────────────────────────────────────────────────────────────
# One-sentence coaching summaries (auto-generated for unknown features)
# ─────────────────────────────────────────────────────────────────────
COACHING_SUMMARIES = {
    "FB_pct": "Pitchers who throw more fastballs (above ~60%) tend to have higher strike rates — leaning on your best fastball works.",
    "DaysSinceLast": "Strike rates are most stable with regular rest (7–30 days); long layoffs show unpredictable effects.",
    "RelSide_Avg": "Extreme release-side positions are linked to lower strike rates — a consistent, repeatable slot matters most.",
    "PitchCount": "Higher pitch counts (60+) are associated with slightly better strike rates, reflecting starters already commanding well.",
    "SpinRate_FB": "Higher fastball spin (above ~2200 RPM) gives a modest bump to strike percentage — more spin means more life on the ball.",
    "HorzApprAngle_Avg": "A neutral horizontal approach angle is ideal; extreme angles in either direction correlate with lower strike rates.",
    "Zone_pct_lag1": "Pitchers who located well in their previous outing tend to carry that command forward into the next session.",
    "PitchTypes": "Using 3+ pitch types is associated with higher strike rates — pitch mix diversity keeps hitters off balance.",
    "RelHeight_Avg": "Moderate release heights produce the best strike rates; very high or low slots tend to reduce command.",
    "Zone_pct_roll3_cov": "More consistent zone rates across recent sessions predicts better strike performance — steady command beats hot-and-cold.",
    "SpinRate_Avg_lag1": "Previous session spin rate carries forward as a predictor — physical readiness shows up in spin consistency.",
    "SpinAxis_Avg": "Spin axis influences pitch movement profile; certain axes produce more command-friendly movement patterns.",
    "HorzBreak_Avg": "Moderate horizontal break is associated with better strike rates — extreme movement can sacrifice control.",
    "Extension_Avg": "More extension (closer release to the plate) gives hitters less reaction time, supporting higher strike rates.",
    "pitcher_prior_strike_pct": "A pitcher's historical strike rate is one of the best predictors of future performance — track record matters.",
    "SpinRate_Avg": "Overall spin rate correlates with pitch quality, though the relationship is nonlinear and pitch-type dependent.",
    "Whiff_pct_roll3_cov": "Consistent whiff generation across sessions signals a pitcher whose stuff plays reliably, not just on good days.",
    "FB_Velo_Max": "Top-end velocity provides a ceiling effect — the harder you can throw when needed, the more room for command pitches.",
    "Strike_pct_roll3_cov": "Pitchers with consistent strike rates session-to-session are more reliable performers overall.",
    "IVB_Avg": "Induced vertical break (ride) on the fastball affects perceived location — more ride means higher effective strike zone.",
    "VeloFade": "Less velocity drop-off late in outings predicts better strike rates — conditioning and endurance show up here.",
    # CMJ summaries
    "CMJ_Jump Height (Imp-Mom) in Inches [in]": "Higher jump height reflects greater lower-body explosiveness, which supports force transfer through the pitching delivery.",
    "CMJ_Peak Power / BM [W/kg]": "Relative power output indicates how efficiently a pitcher generates force — higher values support better energy transfer to the ball.",
    "CMJ_RSI-modified (Imp-Mom) [m/s]": "Reactive strength captures how quickly a pitcher can produce force — faster loading translates to a more explosive delivery.",
    "CMJ_Countermovement Depth [cm]": "Loading depth reflects the pitcher's movement strategy — optimal depth varies, but consistency matters for repeatable mechanics.",
    "CMJ_Eccentric:Concentric Mean Force Ratio [%]": "The balance between absorbing and producing force affects delivery efficiency — a well-balanced ratio supports consistent output.",
}


# ─────────────────────────────────────────────────────────────────────
# ALE computation
# ─────────────────────────────────────────────────────────────────────

def compute_ale(predict_fn, X, feature_idx, n_bins=20):
    """Compute 1-D Accumulated Local Effects for one feature.

    Returns (bin_centers, ale_centered) in scaled units.
    """
    x_col = X[:, feature_idx]
    n_bins = min(n_bins, max(5, len(X) // 5))

    quantiles = np.unique(np.percentile(x_col[~np.isnan(x_col)], np.linspace(0, 100, n_bins + 1)))
    if len(quantiles) < 3:
        return np.array([]), np.array([])

    n_intervals = len(quantiles) - 1
    ale_vals = np.zeros(n_intervals)
    bin_counts = np.zeros(n_intervals)

    for k in range(n_intervals):
        lo, hi = quantiles[k], quantiles[k + 1]
        mask = (
            (x_col >= lo) & (x_col <= hi) if k == 0
            else (x_col > lo) & (x_col <= hi)
        )
        if mask.sum() == 0:
            continue

        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[:, feature_idx] = lo
        X_hi[:, feature_idx] = hi

        ale_vals[k] = float((predict_fn(X_hi) - predict_fn(X_lo)).mean())
        bin_counts[k] = mask.sum()

    ale_accum = np.cumsum(ale_vals)
    total = bin_counts.sum()
    weights = bin_counts / total if total > 0 else np.ones(n_intervals) / n_intervals
    ale_centred = ale_accum - float(np.average(ale_accum, weights=weights))

    bin_centers = (quantiles[:-1] + quantiles[1:]) / 2
    return bin_centers, ale_centred


def compute_permutation_importance(predict_fn, X, y, feature_cols):
    """Rank features by permutation importance (MAE increase when shuffled)."""
    base_mae = float(np.mean(np.abs(predict_fn(X) - y)))
    rng = np.random.default_rng(42)
    importances = {}

    for i, col in enumerate(feature_cols):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        importances[col] = float(np.mean(np.abs(predict_fn(X_perm) - y))) - base_mae

    return importances


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate individual ALE plots from a saved LightGBM model",
    )
    parser.add_argument("--model", required=True, help="Path to strike_pct_model_lgbm.pkl")
    parser.add_argument("--data", required=True, help="Path to gmu_roster_session_data.csv")
    parser.add_argument("--outdir", default="ale_plots", help="Output directory for PNGs")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top features to plot")
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

    predict_fn = lambda X: np.clip(model.predict(X), 0.0, 1.0)

    # ── Load data ──
    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    available = [c for c in feature_cols if c in df.columns]

    X_raw = df[available].values.astype(np.float64)
    X_scaled = scaler.transform(X_raw)
    y = df["Strike_pct"].values.astype(np.float64)

    # ── Rank features ──
    print("Computing permutation importance ...")
    importances = compute_permutation_importance(predict_fn, X_scaled, y, available)
    top_features = sorted(importances, key=lambda c: importances[c], reverse=True)[:args.top_n]

    print(f"\nTop {args.top_n} features by permutation importance:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:<35s}  +{importances[feat]*100:.2f}pp MAE")

    # ── Generate individual ALE plots ──
    print(f"\nGenerating ALE plots → {args.outdir}/")

    for rank, feat in enumerate(top_features, 1):
        feat_idx = available.index(feat)
        centers_scaled, ale = compute_ale(predict_fn, X_scaled, feat_idx, n_bins=20)
        if len(centers_scaled) < 2:
            print(f"  Skipping {feat} — not enough unique values")
            continue

        # Unscale x-axis to original units
        col_mean = scaler.mean_[feat_idx]
        col_scale = scaler.scale_[feat_idx]
        centers_orig = centers_scaled * col_scale + col_mean
        raw_orig = X_raw[:, feat_idx]

        ale_pp = ale * 100  # convert to percentage points

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.fill_between(centers_orig, ale_pp, 0, where=(ale_pp >= 0),
                        color="#90EE90", alpha=0.4)
        ax.fill_between(centers_orig, ale_pp, 0, where=(ale_pp < 0),
                        color="#FFB6B6", alpha=0.4)
        ax.plot(centers_orig, ale_pp, color="#1a1a2e", linewidth=2.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        # Rug plot (observed data distribution)
        valid_raw = raw_orig[~np.isnan(raw_orig)]
        ax.plot(valid_raw, np.full_like(valid_raw, ale_pp.min() - 0.3), "|",
                color="#666", alpha=0.3, markersize=6)

        friendly = FRIENDLY_NAMES.get(feat, feat)
        imp_pp = importances[feat] * 100
        ax.set_title(f"{friendly}   [+{imp_pp:.2f}pp MAE if shuffled]",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("Effect on Strike % (pp)", fontsize=11)

        # Coaching summary
        summary = COACHING_SUMMARIES.get(feat, f"This feature ranks #{rank} in importance for predicting strike percentage.")
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9, style="italic",
                 wrap=True, color="#444")

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.13)

        import re
        safe_feat = re.sub(r'[^\w\-.]', '_', feat)  # replace any non-alphanumeric chars
        filename = f"ale_{rank:02d}_{safe_feat}.png"
        fig.savefig(os.path.join(args.outdir, filename), dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")

    print(f"\nDone! {len(top_features)} ALE plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()