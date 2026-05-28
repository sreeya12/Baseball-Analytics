"""
visualizations.py — Plotting and reporting for strike-% model results.

Generates:
  1. Training loss curves (train vs validation)
  2. Actual vs Predicted scatter (test set)
  3. Per-pitcher bar chart (categorized pitchers)
  4. Category comparison box plot
  5. Pitcher summary CSV
  6. Console report table
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config.Settings import (
    ELITE_THRESHOLD,
    PITCHER_CATEGORIES,
)
from config.logging_config import get_logger
from models.Trainer import TrainResult

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────
CATEGORY_COLORS: Dict[str, str] = {
    "elite":    "#2ecc71",
    "offspeed": "#3498db",
    "fastball": "#e74c3c",
    "unknown":  "#95a5a6",
}

# ─────────────────────────────────────────────────────────────────────
# Friendly display names for features (used in charts and ALE plots)
# ─────────────────────────────────────────────────────────────────────
_FRIENDLY_FEATURE_NAMES: Dict[str, str] = {
    "FB_pct": "Fastball Usage %",
    "RelSide_Avg": "Release Side Position",
    "PitchCount": "Pitch Count",
    "SpinRate_FB": "Fastball Spin Rate (RPM)",
    "HorzApprAngle_Avg": "Horizontal Approach Angle",
    "Zone_pct_lag1": "Prev Session Zone %",
    "Zone_pct_roll3_cov": "Zone % Consistency (CoV)",
    "SpinAxis_Avg": "Spin Axis",
    "HorzBreak_Avg": "Horizontal Break",
    "Extension_Avg": "Extension",
    "SpinRate_Avg": "Spin Rate",
    "Whiff_pct_roll3_cov": "Whiff % Consistency (CoV)",
    "FB_Velo_Max": "Max Fastball Velocity",
    "Strike_pct_roll3_cov": "Strike % Consistency (CoV)",
    "IVB_Avg": "Induced Vertical Break",
    "VeloFade": "Velocity Fade",
    "FB_Velo_Avg": "Avg Fastball Velocity",
    "Strike_pct_lag1": "Prev Session Strike %",
    "Whiff_pct_lag1": "Prev Session Whiff %",
    "EffVelo_Avg": "Effective Velocity",
    "RelSpeed_Avg": "Avg Release Speed",
    "VertApprAngle_Avg": "Vertical Approach Angle",
    "FB_Velo_Avg_lag1": "Prev Session FB Velo",
    "FB_Velo_Avg_roll3": "3-Session Avg FB Velo",
    "SpinRate_Avg_roll3": "3-Session Avg Spin Rate",
    "SpinRate_Avg_lag1": "Prev Session Spin Rate",
    "CMJ_Jump Height (Imp-Mom) in Inches [in]": "CMJ Jump Height (in)",
    "CMJ_Peak Power / BM [W/kg]": "CMJ Peak Power / BW",
    "CMJ_RSI-modified (Imp-Mom) [m/s]": "CMJ Reactive Strength Index",
    "CMJ_Countermovement Depth [cm]": "CMJ Countermovement Depth",
    "CMJ_Eccentric:Concentric Mean Force Ratio [%]": "CMJ Ecc:Con Force Ratio",
    "CMJ_Concentric Mean Power / BM [W/kg]": "CMJ Concentric Mean Power / BW",
    "CMJ_Concentric Peak Force [N]": "CMJ Concentric Peak Force",
    "CMJ_Eccentric Deceleration Mean Force [N]": "CMJ Ecc Decel Mean Force",
    "CMJ_Eccentric Peak Power / BM [W/kg]": "CMJ Ecc Peak Power / BW",
    "CMJ_Eccentric Peak Velocity [m/s]": "CMJ Ecc Peak Velocity",
    "CMJ_Peak Landing Force / BM [N/kg]": "CMJ Peak Landing Force / BW",
    "IMTP_Peak Vertical Force / BM [N/kg]": "IMTP Peak Force / BW",
    "IMTP_Force at 200ms / BM [N/kg]": "IMTP Force at 200ms / BW",
    "IMTP_RFD - 200ms [N/s]": "IMTP Rate of Force Dev.",
}

# One-sentence coaching summaries for ALE plots
_COACHING_SUMMARIES: Dict[str, str] = {
    "FB_pct": "Pitchers throwing 60%+ fastballs see a clear positive effect on strike rate — leaning on your best pitch works.",
    "PitchCount": "Higher pitch counts are associated with slightly better strike rates, reflecting starters already commanding well.",
    "SpinRate_FB": "Higher fastball spin (above ~2200 RPM) gives a modest bump to strike percentage.",
    "HorzApprAngle_Avg": "A neutral horizontal approach angle is ideal; extreme angles correlate with lower strike rates.",
    "Zone_pct_lag1": "Pitchers who located well in their previous outing tend to carry that command forward.",
    "Zone_pct_roll3_cov": "Consistent zone rates across recent sessions predict better strike performance — steady command beats hot-and-cold.",
    "IVB_Avg": "Induced vertical break affects perceived fastball location — more ride supports a higher effective strike zone.",
    "HorzBreak_Avg": "Moderate horizontal break correlates with better strike rates — extreme movement can sacrifice control.",
    "RelSide_Avg": "Extreme release-side positions are linked to lower strike rates — a repeatable slot matters most.",
    "VeloFade": "Less velocity drop-off late in outings predicts better strike rates — conditioning shows up here.",
    "Strike_pct_roll3_cov": "Pitchers with consistent strike rates session-to-session are more reliable performers.",
    "VertApprAngle_Avg": "Steeper vertical approach angles affect pitch perception and swing decisions.",
    "CMJ_Jump Height (Imp-Mom) in Inches [in]": "Moderate jump heights (12–15 in) are associated with higher strike rates — optimal range of lower-body explosiveness.",
    "CMJ_Peak Power / BM [W/kg]": "Relative power output indicates how efficiently a pitcher generates force for energy transfer to the ball.",
    "CMJ_RSI-modified (Imp-Mom) [m/s]": "A sweet spot around 0.45–0.55 m/s where strike rates peak — faster loading supports explosive delivery.",
    "CMJ_Countermovement Depth [cm]": "Loading depth reflects movement strategy — optimal depth varies but consistency matters for repeatable mechanics.",
    "CMJ_Eccentric:Concentric Mean Force Ratio [%]": "The balance between absorbing and producing force affects delivery efficiency.",
    "CMJ_Concentric Mean Power / BM [W/kg]": "Concentric power relative to bodyweight reflects push-off efficiency through the delivery.",
    "CMJ_Concentric Peak Force [N]": "Peak force during the push-off phase relates to ground reaction forces in the delivery.",
    "CMJ_Eccentric Deceleration Mean Force [N]": "How effectively a pitcher absorbs and redirects force during loading — critical for energy transfer.",
    "CMJ_Eccentric Peak Power / BM [W/kg]": "Eccentric power captures ability to absorb force quickly — key for the loading phase of delivery.",
    "CMJ_Eccentric Peak Velocity [m/s]": "Eccentric velocity reflects how fast the pitcher loads — faster loading can support more explosive output.",
    "CMJ_Peak Landing Force / BM [N/kg]": "Landing force relative to bodyweight reflects how the body handles deceleration stress.",
}


class ReportGenerator:
    """Produce all visual and tabular outputs from a training result."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    # ────────────────── public API ──────────────────

    def generate_all(
        self,
        result: TrainResult,
        model_df: pd.DataFrame,
    ) -> None:
        """Run every report in sequence."""
        summary = self._build_pitcher_summary(result, model_df)
        self._save_summary_csv(summary)
        self._print_console_report(summary)
        self._plot_loss_curves(result)
        self._plot_actual_vs_predicted(result, model_df)
        self._plot_pitcher_bars(summary)
        self._plot_category_box(summary)
        self._plot_elite_threshold(summary)
        self._plot_pitcher_progression(result, model_df)
        self._plot_feature_importance(result, model_df)
        self._plot_residuals(result, model_df)
        self._plot_ale(result, model_df)
        logger.info("All reports saved to %s", self.output_dir)

    # ────────────────── summary table ──────────────────

    def _build_pitcher_summary(
        self, result: TrainResult, model_df: pd.DataFrame
    ) -> pd.DataFrame:
        records = pd.DataFrame({
            "Pitcher":  result.pitcher_names,
            "Actual":   model_df["Strike_pct"].values,
            "Predicted": result.predictions,
        })

        summary = (
            records.groupby("Pitcher")
            .agg(
                Sessions=("Actual", "count"),
                Actual_Strike_Pct=("Actual", "mean"),
                Predicted_Strike_Pct=("Predicted", "mean"),
            )
            .reset_index()
        )
        summary["Error"] = summary["Predicted_Strike_Pct"] - summary["Actual_Strike_Pct"]
        summary["Abs_Error"] = summary["Error"].abs()
        summary["Category"] = summary["Pitcher"].map(PITCHER_CATEGORIES).fillna("")
        summary["Is_Elite_Pred"] = summary["Predicted_Strike_Pct"] >= ELITE_THRESHOLD

        return summary.sort_values("Predicted_Strike_Pct", ascending=False)

    def _save_summary_csv(self, summary: pd.DataFrame) -> None:
        path = os.path.join(self.output_dir, "pitcher_predictions.csv")
        summary.to_csv(path, index=False, float_format="%.6f")
        logger.info("Saved pitcher summary → %s", path)

    # ────────────────── console report ──────────────────

    def _print_console_report(self, summary: pd.DataFrame) -> None:
        categorized = summary[summary["Category"] != ""].copy()

        print(f"\n{'=' * 85}")
        print(f"  PREDICTED STRIKE % — CATEGORIZED PITCHERS (GMU 2025)")
        print(f"{'=' * 85}")
        print(
            f"  {'Pitcher':<25} {'Category':<10} {'Sess':>5} "
            f"{'Actual':>9} {'Predicted':>10} {'Error':>8}"
        )
        print(f"  {'-' * 80}")

        for _, row in categorized.iterrows():
            parts = row["Pitcher"].split(",")
            short = parts[0] + ", " + parts[1].strip()[:1] + "."
            flag = " *" if row["Is_Elite_Pred"] else ""
            print(
                f"  {short:<25} {row['Category']:<10} {row['Sessions']:>5.0f} "
                f"{row['Actual_Strike_Pct'] * 100:>8.1f}% "
                f"{row['Predicted_Strike_Pct'] * 100:>9.1f}% "
                f"{row['Error'] * 100:>+7.1f}%{flag}"
            )

        print(f"\n  Elite threshold: {ELITE_THRESHOLD * 100:.0f}%  (* = predicted elite)")

        # Category averages
        print(f"\n  {'Category Averages':>30}")
        print(f"  {'-' * 60}")
        for cat in ["elite", "offspeed", "fastball"]:
            sub = categorized[categorized["Category"] == cat]
            if sub.empty:
                continue
            act = sub["Actual_Strike_Pct"].mean() * 100
            pred = sub["Predicted_Strike_Pct"].mean() * 100
            print(
                f"  {cat:<10}  Actual: {act:>5.1f}%  |  "
                f"Predicted: {pred:>5.1f}%  |  n={len(sub)}"
            )
        print()

    # ────────────────── plots ──────────────────

    def _plot_loss_curves(self, result: TrainResult) -> None:
        if not result.train_losses:
            logger.info("No training loss history (sklearn model) — skipping loss curve plot")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(result.train_losses) + 1)
        ax.plot(epochs, result.train_losses, label="Train Loss", linewidth=1.5)
        ax.plot(epochs, result.val_losses, label="Val Loss", linewidth=1.5)
        ax.axvline(result.best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"Best epoch ({result.best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        self._savefig(fig, "loss_curves.png")

    def _plot_actual_vs_predicted(
        self, result: TrainResult, model_df: pd.DataFrame
    ) -> None:
        actual = model_df["Strike_pct"].values * 100
        predicted = result.predictions * 100

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(actual, predicted, alpha=0.4, s=25, c="steelblue", edgecolors="none")

        lims = [
            max(15, min(actual.min(), predicted.min()) - 5),
            min(95, max(actual.max(), predicted.max()) + 5),
        ]
        ax.plot(lims, lims, "r--", lw=1.2, label="Perfect prediction")
        ax.axhline(ELITE_THRESHOLD * 100, color="green", linestyle=":", alpha=0.5,
                   label=f"Elite threshold ({ELITE_THRESHOLD * 100:.0f}%)")

        m = result.test_metrics
        ax.set_xlabel("Actual Strike %")
        ax.set_ylabel("Predicted Strike %")
        ax.set_title(
            f"Actual vs Predicted  |  "
            f"MAE={m['mae'] * 100:.2f}pp   R²={m['r2']:.3f}"
        )
        ax.legend(loc="upper left")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        plt.tight_layout()
        self._savefig(fig, "actual_vs_predicted.png")

    def _plot_pitcher_bars(self, summary: pd.DataFrame) -> None:
        categorized = summary[summary["Category"] != ""].sort_values(
            "Predicted_Strike_Pct", ascending=True
        )
        if categorized.empty:
            return

        short_names = [n.split(",")[0] for n in categorized["Pitcher"]]
        bar_colors = [CATEGORY_COLORS.get(c, "gray") for c in categorized["Category"]]
        y_pos = np.arange(len(short_names))

        fig, ax = plt.subplots(figsize=(10, max(6, len(short_names) * 0.45)))
        ax.barh(y_pos, categorized["Predicted_Strike_Pct"].values * 100,
                color=bar_colors, alpha=0.85, height=0.6)
        ax.scatter(categorized["Actual_Strike_Pct"].values * 100, y_pos,
                   color="black", zorder=5, s=50, marker="d", label="Actual")

        ax.axvline(ELITE_THRESHOLD * 100, color="green", linestyle="--",
                   alpha=0.7, label=f"Elite ({ELITE_THRESHOLD * 100:.0f}%)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_xlabel("Strike %")
        ax.set_title("Predicted vs Actual Strike % by Pitcher")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        legend_elements = [
            Patch(facecolor=CATEGORY_COLORS["elite"],    label="Elite"),
            Patch(facecolor=CATEGORY_COLORS["offspeed"], label="Off-Speed"),
            Patch(facecolor=CATEGORY_COLORS["fastball"], label="Fastball"),
            plt.Line2D([0], [0], marker="d", color="black", linestyle="",
                       markersize=7, label="Actual"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        plt.tight_layout()
        self._savefig(fig, "pitcher_bars.png")

    def _plot_category_box(self, summary: pd.DataFrame) -> None:
        categorized = summary[summary["Category"] != ""].copy()
        if categorized.empty:
            return

        categorized["Predicted_Pct"] = categorized["Predicted_Strike_Pct"] * 100
        order = ["elite", "offspeed", "fastball"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=categorized, x="Category", y="Predicted_Pct",
            order=order, palette=CATEGORY_COLORS, ax=ax, width=0.5,
        )
        sns.stripplot(
            data=categorized, x="Category", y="Predicted_Pct",
            order=order, color="black", size=7, ax=ax, jitter=0.15,
        )
        ax.axhline(ELITE_THRESHOLD * 100, color="green", linestyle="--",
                   alpha=0.6, label=f"Elite threshold ({ELITE_THRESHOLD * 100:.0f}%)")
        ax.set_ylabel("Predicted Strike %")
        ax.set_title("Predicted Strike % Distribution by Category")
        ax.legend()
        plt.tight_layout()
        self._savefig(fig, "category_boxplot.png")

    def _plot_elite_threshold(self, summary: pd.DataFrame) -> None:
        """Waterfall-style chart: distance from elite threshold per pitcher."""
        categorized = summary[summary["Category"] != ""].copy()
        if categorized.empty:
            return

        categorized["Delta"] = (
            categorized["Predicted_Strike_Pct"] - ELITE_THRESHOLD
        ) * 100
        categorized = categorized.sort_values("Delta", ascending=True)

        short_names = [n.split(",")[0] for n in categorized["Pitcher"]]
        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in categorized["Delta"]]

        fig, ax = plt.subplots(figsize=(10, max(5, len(short_names) * 0.4)))
        y_pos = np.arange(len(short_names))
        ax.barh(y_pos, categorized["Delta"].values, color=colors, alpha=0.85, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_xlabel(f"Percentage Points from Elite Threshold ({ELITE_THRESHOLD * 100:.0f}%)")
        ax.set_title("Distance from Elite Threshold (Predicted)")
        plt.tight_layout()
        self._savefig(fig, "elite_threshold_gap.png")

    def _plot_pitcher_progression(
        self, result: TrainResult, model_df: pd.DataFrame
    ) -> None:
        """Small-multiples: each pitcher's actual strike % over season sessions."""
        df = model_df[["Pitcher", "Date", "Strike_pct", "Category"]].copy()
        df["Predicted"] = result.predictions
        df = df.sort_values(["Pitcher", "Date"])

        pitchers = sorted(df["Pitcher"].unique())
        n = len(pitchers)
        if n == 0:
            return

        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(14, max(4, nrows * 3)),
            squeeze=False,
        )

        for i, pitcher in enumerate(pitchers):
            ax = axes[i // ncols][i % ncols]
            sub = df[df["Pitcher"] == pitcher].sort_values("Date")
            cat = sub["Category"].iloc[0] if not sub.empty else "unknown"
            color = CATEGORY_COLORS.get(cat, "#95a5a6")

            ax.plot(
                range(len(sub)), sub["Strike_pct"] * 100,
                "o-", color=color, linewidth=1.8, markersize=5, label="Actual",
            )
            ax.plot(
                range(len(sub)), sub["Predicted"] * 100,
                "s--", color="#7f8c8d", linewidth=1.1, markersize=3,
                alpha=0.75, label="Predicted",
            )
            ax.axhline(
                ELITE_THRESHOLD * 100, color="#27ae60",
                linestyle=":", linewidth=1, alpha=0.6,
            )
            short = pitcher.split(",")[0]
            ax.set_title(f"{short}  [{cat}]", fontsize=8, fontweight="bold")
            ax.set_ylim(30, 90)
            ax.set_xlabel("Session", fontsize=7)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.tick_params(labelsize=7)

        # Hide unused panels
        for j in range(i + 1, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        # Shared legend
        handles = [
            plt.Line2D([0], [0], color="steelblue", marker="o", label="Actual"),
            plt.Line2D([0], [0], color="#7f8c8d", marker="s",
                       linestyle="--", label="Predicted"),
            plt.Line2D([0], [0], color="#27ae60", linestyle=":",
                       label=f"Elite ({ELITE_THRESHOLD * 100:.0f}%)"),
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=3)
        fig.suptitle(
            "Strike % Progression by Pitcher (Session-by-Session)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        self._savefig(fig, "pitcher_progression.png")

    def _plot_feature_importance(
        self, result: TrainResult, model_df: pd.DataFrame
    ) -> None:
        """Permutation importance: shuffle each feature, measure MAE increase.

        Works for both PyTorch (nn) and sklearn/LightGBM models.
        Bars are always positive (larger = more important) and color-coded
        by feature source: TrackMan (gray), CMJ (blue), IMTP (orange).
        """
        feat_cols: List[str] = result.feature_cols
        available = [c for c in feat_cols if c in model_df.columns]
        if not available:
            return

        X = model_df[available].values.astype(np.float32)
        y = model_df["Strike_pct"].values.astype(np.float32)
        X_scaled = result.scaler.transform(X).astype(np.float32)

        model = result.model
        predict_fn = self._get_predict_fn(result)

        # ── Compute permutation importance (unified for all model types) ──
        base_mae = float(np.mean(np.abs(predict_fn(X_scaled) - y)))
        rng = np.random.default_rng(42)

        importances: Dict[str, float] = {}
        for i, col in enumerate(available):
            X_perm = X_scaled.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_mae = float(np.mean(np.abs(predict_fn(X_perm) - y)))
            # Positive value = shuffling hurts the model = feature is important
            importances[col] = perm_mae - base_mae

        # Sort ascending for horizontal bar chart (largest at top)
        imp = pd.Series(importances).sort_values(ascending=True)

        # ── Color-code by feature source ──
        def _feat_color(name: str) -> str:
            if "CMJ" in name:  return "#2E86AB"   # blue
            if "IMTP" in name: return "#E8A838"   # orange
            return "#555555"                       # gray (TrackMan)

        colors = [_feat_color(f) for f in imp.index]

        # ── Friendly display names ──
        friendly = _FRIENDLY_FEATURE_NAMES
        display_names = [friendly.get(f, f) for f in imp.index]

        fig, ax = plt.subplots(figsize=(12, max(6, len(imp) * 0.38)))
        ax.barh(
            range(len(imp)), imp.values * 100,
            color=colors, edgecolor="white", linewidth=0.5, height=0.7,
        )
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(display_names, fontsize=9)
        ax.set_xlabel("Increase in MAE when feature is shuffled (pp)", fontsize=10)

        model_label = getattr(result, "model_type", "model").upper()
        ax.set_title(
            f"Feature Importance — Permutation ({model_label})\n"
            "Larger bar = model relies more on this feature",
            fontsize=12, fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=9)

        # Legend for feature sources
        ax.legend(
            handles=[
                Patch(facecolor="#555555", label="TrackMan"),
                Patch(facecolor="#2E86AB", label="CMJ"),
                Patch(facecolor="#E8A838", label="IMTP"),
            ],
            loc="lower right", fontsize=9,
        )

        plt.tight_layout()
        self._savefig(fig, "feature_importance.png")

    def _plot_residuals(
        self, result: TrainResult, model_df: pd.DataFrame
    ) -> None:
        """Residual distribution and residuals-vs-predicted scatter."""
        residuals = (result.predictions - model_df["Strike_pct"].values) * 100
        predicted = result.predictions * 100
        categories = model_df["Category"].values

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: histogram ──
        ax = axes[0]
        ax.hist(residuals, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
        ax.axvline(
            float(np.mean(residuals)), color="orange", linestyle="-",
            linewidth=1.2, label=f"Bias={np.mean(residuals):.2f}pp",
        )
        ax.set_xlabel("Residual (Predicted − Actual, pp)", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution")
        ax.legend(fontsize=8)

        # ── Right: residuals vs predicted, colored by category ──
        ax = axes[1]
        for cat, color in CATEGORY_COLORS.items():
            mask = categories == cat
            if not mask.any():
                continue
            ax.scatter(
                predicted[mask], residuals[mask],
                color=color, alpha=0.6, s=35, edgecolors="none",
                label=cat.capitalize(),
            )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Predicted Strike %", fontsize=9)
        ax.set_ylabel("Residual (pp)")
        ax.set_title("Residuals vs Predicted (colored by category)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.legend(fontsize=8)

        plt.tight_layout()
        self._savefig(fig, "residuals.png")

    # ────────────────── ALE plots ──────────────────

    def _plot_ale(self, result: TrainResult, model_df: pd.DataFrame) -> None:
        """ALE (Accumulated Local Effects) for the top 10 most impactful features.

        Generates:
          - ale_plots.png: combined 2-column grid (backward compatible)
          - ale_plots/ale_01_<feature>.png: individual plots with coaching summaries

        Y-axis is in percentage points: +3pp means that value of the feature
        raises the model's predicted strike % by ~3 pp vs average.
        """
        import re

        feat_cols = result.feature_cols
        available = [c for c in feat_cols if c in model_df.columns]
        if not available:
            return

        X        = model_df[available].values.astype(np.float32)
        y        = model_df["Strike_pct"].values.astype(np.float32)
        X_scaled = result.scaler.transform(X).astype(np.float32)

        predict_fn = self._get_predict_fn(result)

        # ── Rank features by permutation importance ──
        base_mae = float(np.mean(np.abs(predict_fn(X_scaled) - y)))
        rng = np.random.default_rng(42)
        importances: Dict[str, float] = {}
        for i, col in enumerate(available):
            X_perm = X_scaled.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            importances[col] = float(np.mean(np.abs(predict_fn(X_perm) - y))) - base_mae

        top_features = sorted(importances, key=lambda c: importances[c], reverse=True)[:10]
        top_indices  = [available.index(f) for f in top_features]

        friendly = _FRIENDLY_FEATURE_NAMES
        summaries = _COACHING_SUMMARIES

        # ── Individual ALE plots (one PNG per feature) ──
        ale_dir = os.path.join(self.output_dir, "ale_plots")
        os.makedirs(ale_dir, exist_ok=True)

        for rank, (feat_name, feat_idx) in enumerate(zip(top_features, top_indices), 1):
            bin_centers_scaled, ale = self._compute_ale(predict_fn, X_scaled, feat_idx)
            if len(bin_centers_scaled) < 2:
                continue

            col_mean  = float(result.scaler.mean_[feat_idx])
            col_scale = float(result.scaler.scale_[feat_idx])
            bin_centers_orig = bin_centers_scaled * col_scale + col_mean
            orig_vals        = X[:, feat_idx]
            ale_pp = ale * 100

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(bin_centers_orig, ale_pp, 0,
                            where=(ale_pp >= 0), color="#90EE90", alpha=0.4)
            ax.fill_between(bin_centers_orig, ale_pp, 0,
                            where=(ale_pp < 0), color="#FFB6B6", alpha=0.4)
            ax.plot(bin_centers_orig, ale_pp, color="#1a1a2e", linewidth=2.5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

            valid_orig = orig_vals[~np.isnan(orig_vals)]
            ax.plot(valid_orig, np.full_like(valid_orig, ale_pp.min() - 0.3),
                    "|", color="#666", alpha=0.3, markersize=6)

            display_name = friendly.get(feat_name, feat_name)
            imp_pp = importances[feat_name] * 100
            ax.set_title(f"{display_name}   [+{imp_pp:.2f}pp MAE if shuffled]",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel(feat_name, fontsize=11)
            ax.set_ylabel("Effect on Strike % (pp)", fontsize=11)

            summary = summaries.get(feat_name,
                f"Ranked #{rank} in importance for predicting strike percentage.")
            fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                     style="italic", wrap=True, color="#444")
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.13)

            safe_feat = re.sub(r'[^\w\-.]', '_', feat_name)
            fig.savefig(
                os.path.join(ale_dir, f"ale_{rank:02d}_{safe_feat}.png"),
                dpi=200, bbox_inches="tight", facecolor="white",
            )
            plt.close(fig)

        logger.info("Individual ALE plots saved to %s/", ale_dir)

        # ── Combined grid (backward compatible) ──
        n     = len(top_features)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(14, nrows * 3.2),
            squeeze=False,
        )

        for plot_idx, (feat_name, feat_idx) in enumerate(zip(top_features, top_indices)):
            ax = axes[plot_idx // ncols][plot_idx % ncols]

            bin_centers_scaled, ale = self._compute_ale(predict_fn, X_scaled, feat_idx)
            if len(bin_centers_scaled) < 2:
                ax.set_visible(False)
                continue

            col_mean  = float(result.scaler.mean_[feat_idx])
            col_scale = float(result.scaler.scale_[feat_idx])
            bin_centers_orig = bin_centers_scaled * col_scale + col_mean
            orig_vals        = X[:, feat_idx]

            ale_pp = ale * 100

            ax.plot(bin_centers_orig, ale_pp, color="#2c3e50", linewidth=2, zorder=3)
            ax.fill_between(
                bin_centers_orig, ale_pp, 0,
                where=(ale_pp >= 0), alpha=0.18, color="#2ecc71", zorder=2,
            )
            ax.fill_between(
                bin_centers_orig, ale_pp, 0,
                where=(ale_pp < 0), alpha=0.18, color="#e74c3c", zorder=2,
            )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

            rug_y = ale_pp.min() - max(0.4, abs(ale_pp).max() * 0.08)
            valid_orig = orig_vals[~np.isnan(orig_vals)]
            ax.plot(
                valid_orig,
                np.full_like(valid_orig, rug_y),
                "|", color="#7f8c8d", alpha=0.35, markersize=4,
            )

            display_name = friendly.get(feat_name, feat_name)
            imp_pp = importances[feat_name] * 100
            ax.set_title(
                f"{display_name}   [+{imp_pp:.2f}pp MAE if shuffled]",
                fontsize=8, fontweight="bold",
            )
            ax.set_xlabel(feat_name, fontsize=7)
            ax.set_ylabel("ALE effect (pp)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

        for j in range(len(top_features), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(
            "Accumulated Local Effects (ALE) — Top 10 Features\n"
            "Each curve shows how that feature alone shifts predicted strike %\n"
            "relative to the average.  Rug marks = observed data distribution.",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        self._savefig(fig, "ale_plots.png")
        logger.info("ALE combined grid saved (top %d features)", len(top_features))

    @staticmethod
    def _compute_ale(
        predict_fn,
        X_scaled: np.ndarray,
        feat_idx: int,
        n_bins: int = 20,
    ):
        """Compute 1-D ALE for one feature.

        Returns (bin_centers, ale_centered) both as 1-D numpy arrays.
        bin_centers are in scaled units; the caller inverse-transforms them.

        Algorithm:
          1. Divide feature into quantile intervals.
          2. Within each interval, replace the feature with the interval's
             lower / upper edge and record the average prediction difference.
          3. Accumulate those local differences left-to-right.
          4. Centre by subtracting the observation-weighted mean so the curve
             passes through zero at the average feature value.
        """
        x_col  = X_scaled[:, feat_idx]
        # Handle NaN values (CMJ/IMTP features may have missing data)
        valid_mask = ~np.isnan(x_col)
        if valid_mask.sum() < 10:
            return np.array([x_col[valid_mask].min() if valid_mask.any() else 0,
                             x_col[valid_mask].max() if valid_mask.any() else 1]), np.array([0.0, 0.0])
        n_bins = min(n_bins, max(5, int(valid_mask.sum()) // 5))

        quantiles = np.unique(
            np.percentile(x_col[valid_mask], np.linspace(0, 100, n_bins + 1))
        )
        if len(quantiles) < 3:
            return np.array([x_col.min(), x_col.max()]), np.array([0.0, 0.0])

        n_intervals = len(quantiles) - 1
        ale_vals    = np.zeros(n_intervals)
        bin_counts  = np.zeros(n_intervals)

        for k in range(n_intervals):
            lo, hi = quantiles[k], quantiles[k + 1]
            # First bin is closed on both ends; subsequent bins open on left
            mask = (
                (x_col >= lo) & (x_col <= hi) if k == 0
                else (x_col > lo) & (x_col <= hi)
            )
            if mask.sum() == 0:
                continue

            X_lo = X_scaled[mask].copy()
            X_hi = X_scaled[mask].copy()
            X_lo[:, feat_idx] = lo
            X_hi[:, feat_idx] = hi

            ale_vals[k]   = float((predict_fn(X_hi) - predict_fn(X_lo)).mean())
            bin_counts[k] = mask.sum()

        ale_accum   = np.cumsum(ale_vals)
        total       = bin_counts.sum()
        weights     = bin_counts / total if total > 0 else np.ones(n_intervals) / n_intervals
        ale_centred = ale_accum - float(np.average(ale_accum, weights=weights))

        bin_centers = (quantiles[:-1] + quantiles[1:]) / 2
        return bin_centers, ale_centred

    def _get_predict_fn(self, result: TrainResult):
        """Return a unified numpy-in / numpy-out predict function for any model."""
        model = result.model
        if isinstance(model, torch.nn.Module):
            model.eval()
            def _torch_predict(X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return (
                        model(torch.tensor(X.astype(np.float32)))
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
            return _torch_predict
        # sklearn / LightGBM
        return lambda X: np.clip(model.predict(X), 0.0, 1.0)

    # ────────────────── helper ──────────────────

    def _savefig(self, fig, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved plot → %s", filename)