"""
loader.py — TrackMan CSV ingestion and session-level feature engineering.

Reads all CSV files from a directory, cleans the raw pitch-level data,
engineers session-level (pitcher × date) features including temporal
lag/rolling features and velocity fade, and returns a model-ready DataFrame.

Updated for full 63-file dataset.

v2 changes:
  - Split rolling features: rolling MEAN for velocity/spin (no leakage),
    rolling COEFFICIENT OF VARIATION for command metrics (Strike/Zone/Whiff)
    to avoid the autocorrelation shortcut where the model just predicts
    near a pitcher's recent strike-% average.
  - Removed SessionNum (not actionable for practitioners).

v3 changes:
  - CMJ/IMTP NaN handling: instead of dropping these feature groups when
    sparsely populated, keep them with NaNs intact for LightGBM's native
    missing-value routing. LightGBM learns optimal split directions for
    missing values during training — no imputation needed.
  - Sprint data still gets median-filled (static per pitcher, not time-varying).
"""

import glob
import os
import sys
from typing import Optional

# Ensure the Analytics root is on the path when this module is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.Settings import (
    CATEGORY_LIST,
    FEATURE_COLS,
    LAG_BASE_COLS,
    ROLL_MEAN_COLS,
    ROLL_COV_COLS,
    PITCHER_CATEGORIES,
    TARGET_COL,
    TRACKMAN_FASTBALL_TYPES,
    TRACKMAN_STRIKE_CALLS,
    TRACKMAN_BALL_CALLS,
)
from config.logging_config import get_logger

logger = get_logger(__name__)


class TrackManLoader:
    """Load TrackMan CSVs → engineer session features → return model-ready data."""

    def __init__(self, data_dir: str, vald_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.vald_dir = vald_dir
        self._raw: Optional[pd.DataFrame] = None
        self._sessions: Optional[pd.DataFrame] = None

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Full pipeline: read → clean → engineer → return model-ready df."""
        self._read_csvs()
        self._clean()
        self._engineer_session_features()
        self._add_velocity_fade()
        self._attach_categories()
        self._add_lag_rolling_features()
        self._merge_vald()
        self._merge_sixty_times()
        model_df = self._prepare_model_frame()
        return model_df

    def get_sessions(self) -> pd.DataFrame:
        """Return the full session DataFrame (before NaN-dropping)."""
        if self._sessions is None:
            raise RuntimeError("Call .load() first.")
        return self._sessions.copy()

    # ────────────────────────────────────────────────────────────────
    # 1. Read CSVs — handles nested directories too
    # ────────────────────────────────────────────────────────────────

    def _read_csvs(self) -> None:
        csv_paths = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "*.csv"), recursive=True)
        )
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}"
            )

        frames = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path, low_memory=False)
                df["_source_file"] = os.path.basename(path)
                frames.append(df)
                logger.info("Loaded %s  (%d rows)", os.path.basename(path), len(df))
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

        self._raw = pd.concat(frames, ignore_index=True)
        logger.info(
            "Total raw pitches: %s across %d file(s)",
            f"{len(self._raw):,}",
            len(frames),
        )

    # ────────────────────────────────────────────────────────────────
    # 2. Clean
    # ────────────────────────────────────────────────────────────────

    def _clean(self) -> None:
        df = self._raw
        assert df is not None

        # Parse date — format="mixed" handles the one file that uses M/D/YYYY
        # while the rest use YYYY-MM-DD (pandas 3.x infers format from first rows,
        # which locks out all other formats without this flag)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
        df = df.dropna(subset=["Date"])

        # Normalize pitcher names (strip extra whitespace)
        df["Pitcher"] = df["Pitcher"].astype(str).str.strip()

        # Remove junk rows (test accounts, blanks)
        junk_names = {"A, a", "Unknown", "", "nan"}
        df = df[~df["Pitcher"].isin(junk_names)]

        # Cast numeric columns (TrackMan sometimes exports blanks)
        numeric_cols = [
            "RelSpeed", "SpinRate", "SpinAxis", "Extension",
            "InducedVertBreak", "HorzBreak", "VertBreak",
            "RelHeight", "RelSide", "VertApprAngle", "HorzApprAngle",
            "EffectiveVelo", "PlateLocHeight", "PlateLocSide",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter implausible velocity (TrackMan noise)
        df = df[df["RelSpeed"].between(40, 110) | df["RelSpeed"].isna()]

        # Binary flags
        df["IsFB"] = df["TaggedPitchType"].isin(TRACKMAN_FASTBALL_TYPES).astype(int)

        df["IsStrike"] = df["PitchCall"].isin(TRACKMAN_STRIKE_CALLS).astype(int)
        df["IsBall"] = df["PitchCall"].isin(TRACKMAN_BALL_CALLS).astype(int)

        # Zone / whiff helpers
        df["IsInZone"] = (
            df["PlateLocHeight"].between(1.5, 3.5)
            & df["PlateLocSide"].between(-0.83, 0.83)
        ).astype(int)

        df["IsWhiff"] = (df["PitchCall"] == "StrikeSwinging").astype(int)

        self._raw = df
        logger.info(
            "Cleaned data: %s rows, %d unique pitchers",
            f"{len(df):,}",
            df["Pitcher"].nunique(),
        )

    # ────────────────────────────────────────────────────────────────
    # 3. Session-level feature engineering
    # ────────────────────────────────────────────────────────────────

    def _engineer_session_features(self) -> None:
        raw = self._raw
        assert raw is not None

        logger.info("Engineering session-level features …")

        session = (
            raw.groupby(["Pitcher", "Date"])
            .agg(
                # ── Core 8 features ──
                RelSpeed_Avg=("RelSpeed", "mean"),
                FB_Velo_Avg=(
                    "RelSpeed",
                    lambda x: x[raw.loc[x.index, "IsFB"] == 1].mean(),
                ),
                SpinRate_Avg=("SpinRate", "mean"),
                Extension_Avg=("Extension", "mean"),
                IVB_Avg=("InducedVertBreak", "mean"),
                HorzBreak_Avg=("HorzBreak", "mean"),
                SpinAxis_Avg=("SpinAxis", "mean"),
                RelHeight_Avg=("RelHeight", "mean"),
                # ── Extended ──
                FB_Velo_Max=(
                    "RelSpeed",
                    lambda x: x[raw.loc[x.index, "IsFB"] == 1].max(),
                ),
                SpinRate_FB=(
                    "SpinRate",
                    lambda x: x[raw.loc[x.index, "IsFB"] == 1].mean(),
                ),
                VertBreak_Avg=("VertBreak", "mean"),
                RelSide_Avg=("RelSide", "mean"),
                VertApprAngle_Avg=("VertApprAngle", "mean"),
                HorzApprAngle_Avg=("HorzApprAngle", "mean"),
                EffVelo_Avg=("EffectiveVelo", "mean"),
                Zone_pct=("IsInZone", "mean"),
                Whiff_pct=("IsWhiff", "mean"),
                PitchCount=("RelSpeed", "count"),
                PitchTypes=("TaggedPitchType", "nunique"),
                FB_pct=("IsFB", "mean"),
                # ── Target ──
                Strike_pct=("IsStrike", "mean"),
            )
            .reset_index()
        )

        self._sessions = session
        logger.info(
            "Sessions created: %d rows × %d cols", len(session), session.shape[1]
        )

    # ────────────────────────────────────────────────────────────────
    # 4. Velocity fade (early vs late in outing)
    # ────────────────────────────────────────────────────────────────

    def _add_velocity_fade(self) -> None:
        raw = self._raw
        assert raw is not None

        fb = raw[(raw["IsFB"] == 1) & raw["RelSpeed"].notna()].copy()
        fb["Rank"] = fb.groupby(["Pitcher", "Date"]).cumcount() + 1

        early = (
            fb[fb["Rank"] <= 20]
            .groupby(["Pitcher", "Date"])["RelSpeed"]
            .mean()
            .rename("VeloEarly")
        )
        late = (
            fb[fb["Rank"] > 20]
            .groupby(["Pitcher", "Date"])["RelSpeed"]
            .mean()
            .rename("VeloLate")
        )
        fade = (early - late).rename("VeloFade").reset_index()

        self._sessions = self._sessions.merge(
            fade, on=["Pitcher", "Date"], how="left"
        )
        logger.info("Added velocity fade feature")

    # ────────────────────────────────────────────────────────────────
    # 5. Category attachment
    # ────────────────────────────────────────────────────────────────

    def _attach_categories(self) -> None:
        session = self._sessions
        assert session is not None

        session["Category"] = (
            session["Pitcher"].map(PITCHER_CATEGORIES).fillna("unknown")
        )
        for cat in CATEGORY_LIST:
            session[f"cat_{cat}"] = (session["Category"] == cat).astype(int)

        self._sessions = session

    # ────────────────────────────────────────────────────────────────
    # 6. Lag and rolling features (temporal trends per pitcher)
    #
    # v2: Split rolling features by metric type to avoid autocorrelation
    # shortcut. Velocity/spin use rolling mean (level matters, no leakage).
    # Command metrics (Strike_pct, Zone_pct, Whiff_pct) use coefficient of
    # variation (std/mean) to capture CONSISTENCY rather than LEVEL —
    # preventing the model from just predicting near a pitcher's recent avg.
    # ────────────────────────────────────────────────────────────────

    def _add_lag_rolling_features(self) -> None:
        session = self._sessions
        assert session is not None

        session = session.sort_values(["Pitcher", "Date"]).reset_index(drop=True)
        g = session.groupby("Pitcher")

        # ── lag1: previous session's value (all base cols) ──────────────
        for col in LAG_BASE_COLS:
            if col not in session.columns:
                continue
            session[f"{col}_lag1"] = g[col].shift(1)

        # ── roll3 mean: velocity / spin rate ────────────────────────────
        # Level matters here — these aren't direct components of strike %
        # so using them as averages doesn't create a leakage shortcut.
        for col in ROLL_MEAN_COLS:
            if col not in session.columns:
                continue
            session[f"{col}_roll3"] = g[col].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )

        # ── roll3 CoV: command metrics ──────────────────────────────────
        # Strike_pct, Zone_pct, Whiff_pct — use coefficient of variation
        # (std / |mean|) to capture consistency rather than level. This
        # prevents the autocorrelation shortcut where the model just
        # predicts near a pitcher's recent strike-% average.
        #
        # Safeguards:
        #   - .shift(1) so the current session is excluded (no leakage)
        #   - min_periods=2: need at least 2 prior sessions for std
        #     (CoV of a single value is undefined)
        #   - abs(mean) in denominator avoids sign flips on near-zero
        #   - inf/NaN handled by _prepare_model_frame downstream
        for col in ROLL_COV_COLS:
            if col not in session.columns:
                continue

            shifted = g[col].transform(lambda x: x.shift(1))
            roll_mean = shifted.groupby(session["Pitcher"]).transform(
                lambda x: x.rolling(3, min_periods=2).mean()
            )
            roll_std = shifted.groupby(session["Pitcher"]).transform(
                lambda x: x.rolling(3, min_periods=2).std()
            )
            # CoV = std / |mean|; guard against divide-by-zero
            session[f"{col}_roll3_cov"] = (
                roll_std / roll_mean.abs().replace(0, np.nan)
            )

        # Days since last outing (workload context)
        session["DaysSinceLast"] = g["Date"].diff().dt.days

        # Pitcher prior: expanding mean of Strike_pct from all PREVIOUS sessions.
        # shift(1) ensures the current session is excluded — no leakage.
        # NaN on a pitcher's first session; filled downstream with dataset mean,
        # equivalent to a flat prior of "average pitcher" for first-time sessions.
        session["pitcher_prior_strike_pct"] = g["Strike_pct"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        self._sessions = session
        logger.info(
            "Added lag1 (%d cols), roll3 mean (%d cols), roll3 CoV (%d cols), pitcher prior",
            len(LAG_BASE_COLS), len(ROLL_MEAN_COLS), len(ROLL_COV_COLS),
        )

    # ────────────────────────────────────────────────────────────────
    # 7. Merge VALD force plate data (CMJ + IMTP)
    # ────────────────────────────────────────────────────────────────

    def _merge_vald(self) -> None:
        if self.vald_dir is None:
            logger.info("No VALD directory provided — skipping CMJ/IMTP merge")
            return

        from data.vald_loader import VALDLoader

        vald = VALDLoader(self.vald_dir)
        vald.load()
        self._sessions = vald.merge_into_sessions(self._sessions)
        logger.info("Merged VALD CMJ + IMTP features into sessions")

    # ────────────────────────────────────────────────────────────────
    # 8. Merge 60-yard sprint data (auto-detected from data_dir)
    # ────────────────────────────────────────────────────────────────

    def _merge_sixty_times(self) -> None:
        """Auto-detect BASE 60 TIMES .xlsx in data_dir and merge sprint features."""
        xlsx_files = glob.glob(
            os.path.join(self.data_dir, "**", "*60*.xlsx"), recursive=True
        ) + glob.glob(
            os.path.join(self.data_dir, "**", "*60*.xls"), recursive=True
        )
        if not xlsx_files:
            logger.info("No 60-times Excel file found in %s — skipping", self.data_dir)
            return

        xlsx_path = xlsx_files[0]
        logger.info("Found 60-times file: %s", os.path.basename(xlsx_path))

        from data.sixty_times_loader import SixtyTimesLoader

        loader = SixtyTimesLoader(xlsx_path)
        loader.load()
        self._sessions = loader.merge_into_sessions(self._sessions)

    # ────────────────────────────────────────────────────────────────
    # 9. Prepare model-ready DataFrame
    #
    # v3: CMJ/IMTP columns keep NaNs for LightGBM native handling
    #     instead of being dropped or median-filled.
    # ────────────────────────────────────────────────────────────────

    def _prepare_model_frame(self) -> pd.DataFrame:
        session = self._sessions
        assert session is not None

        keep_cols = FEATURE_COLS + [TARGET_COL, "Pitcher", "Date", "Category"]
        model_df = session[[c for c in keep_cols if c in session.columns]].copy()

        # Coerce features to numeric
        for col in FEATURE_COLS:
            if col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        # Drop rows missing the target
        before = len(model_df)
        model_df = model_df.dropna(subset=[TARGET_COL])
        dropped = before - len(model_df)
        if dropped:
            logger.info("Dropped %d rows with missing target", dropped)

        # Drop feature columns that are entirely NaN (e.g. CMJ/IMTP when
        # no VALD data was provided, or columns with no data at all)
        feat_present = [c for c in FEATURE_COLS if c in model_df.columns]
        all_nan_cols = [
            c for c in feat_present if model_df[c].isna().all()
        ]
        if all_nan_cols:
            logger.warning(
                "Dropping %d all-NaN feature columns: %s",
                len(all_nan_cols),
                ", ".join(all_nan_cols[:5]) + ("..." if len(all_nan_cols) > 5 else ""),
            )
            model_df = model_df.drop(columns=all_nan_cols)
            feat_present = [c for c in feat_present if c not in all_nan_cols]

        # ──────────────────────────────────────────────────────────
        # Handle optional feature groups (CMJ, IMTP, Sprint)
        #
        # v3: CMJ/IMTP keep NaNs for LightGBM native missing-value
        # handling. LightGBM routes missing values to the optimal
        # split direction during training — no imputation needed.
        # This preserves all sessions while giving the model real
        # biomechanical signal wherever VALD data exists.
        #
        # Sprint data is still median-filled since it's a single
        # static value per pitcher (not time-varying like force
        # plate tests).
        # ──────────────────────────────────────────────────────────
        from config.Settings import CMJ_FEATURE_COLS, IMTP_FEATURE_COLS, SIXTY_TIMES_FEATURES

        # Groups where LightGBM handles NaNs natively (time-varying tests)
        NATIVE_NAN_GROUPS = {
            "CMJ":  CMJ_FEATURE_COLS,
            "IMTP": IMTP_FEATURE_COLS,
        }
        # Groups that still get median-filled (static per pitcher)
        MEDIAN_FILL_GROUPS = {
            "Sprint": SIXTY_TIMES_FEATURES,
        }

        # CMJ/IMTP: only drop if completely empty (0% populated)
        for group_name, group_cols in NATIVE_NAN_GROUPS.items():
            present_in_group = [c for c in group_cols if c in feat_present]
            if not present_in_group:
                continue
            fill_rate = model_df[present_in_group].notna().mean().mean()
            if fill_rate == 0:
                logger.info(
                    "Dropping %s feature group (0%% populated — no data at all)",
                    group_name,
                )
                model_df = model_df.drop(columns=present_in_group)
                feat_present = [c for c in feat_present if c not in present_in_group]
            else:
                logger.info(
                    "%s feature group: %.0f%% populated — keeping NaNs for "
                    "LightGBM native missing-value handling",
                    group_name, fill_rate * 100,
                )

        # Sprint: drop if <50% populated, otherwise median-fill
        for group_name, group_cols in MEDIAN_FILL_GROUPS.items():
            present_in_group = [c for c in group_cols if c in feat_present]
            if not present_in_group:
                continue
            fill_rate = model_df[present_in_group].notna().mean().mean()
            if fill_rate < 0.50:
                logger.info(
                    "Dropping %s feature group (only %.0f%% populated — "
                    "median-fill would add noise, not signal)",
                    group_name, fill_rate * 100,
                )
                model_df = model_df.drop(columns=present_in_group)
                feat_present = [c for c in feat_present if c not in present_in_group]

        # Median-fill NON-CMJ/IMTP features only
        # (CMJ/IMTP keep their NaNs for LightGBM)
        cmj_imtp_cols = set(CMJ_FEATURE_COLS + IMTP_FEATURE_COLS)
        fill_cols = [c for c in feat_present if c not in cmj_imtp_cols]
        if fill_cols:
            model_df[fill_cols] = model_df[fill_cols].fillna(
                model_df[fill_cols].median()
            )

        # Replace inf with NaN (LightGBM treats both as missing)
        model_df[feat_present] = (
            model_df[feat_present]
            .replace([np.inf, -np.inf], np.nan)
        )

        # For non-CMJ/IMTP columns, fill any remaining NaN with 0
        remaining_fill = [c for c in feat_present if c not in cmj_imtp_cols]
        if remaining_fill:
            model_df[remaining_fill] = model_df[remaining_fill].fillna(0.0)

        logger.info(
            "Model-ready data: %d rows × %d features",
            len(model_df),
            len(feat_present),
        )
        return model_df