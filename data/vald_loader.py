"""
vald_loader.py — Load VALD force plate data (CMJ + IMTP).

Reads VALD CSVs, normalizes pitcher names to match TrackMan format,
and prepares the data for as-of merging with session data.

CMJ = Countermovement Jump (lower-body explosive power)
IMTP = Isometric Mid-Thigh Pull (max strength)

Tests are run ~2x per week, so we use as-of joining: for each pitching
session, we attach the most recent VALD test on or before that date.
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.Settings import CMJ_FEATURES, IMTP_FEATURES

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Name normalization
# ─────────────────────────────────────────────────────────────────────
# VALD uses "First Last", TrackMan uses "Last, First"
# We normalize VALD → TrackMan format and apply manual overrides for
# spelling differences (Elliott vs Elliot, O'Keefe vs Okeeffe, etc.)
# ─────────────────────────────────────────────────────────────────────

NAME_OVERRIDES = {
    "Daniel Elliott":  "Elliot, Daniel",
    "Shaun O'Keefe":   "Okeeffe, Shaun",
    "Shaun O\u2019Keefe": "Okeeffe, Shaun",  # curly apostrophe variant
    "Britt Yount":     "Yount, Britt",
    "Michael Madigan": "Madigan III, Michael",
    "Connor O'Hara":   "O'Hara, Connor",
}


def normalize_vald_name(name: str) -> str:
    """Convert 'First Last' (VALD) → 'Last, First' (TrackMan)."""
    if not isinstance(name, str):
        return ""

    name = name.strip().replace("  ", " ")

    # Apply manual overrides first
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]

    parts = name.split()
    if len(parts) < 2:
        return name

    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last}, {first}"


# ─────────────────────────────────────────────────────────────────────
# VALD Loader
# ─────────────────────────────────────────────────────────────────────


class VALDLoader:
    """Load and process CMJ + IMTP data from VALD CSVs."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cmj: Optional[pd.DataFrame] = None
        self._imtp: Optional[pd.DataFrame] = None

    # ────────────────── public API ──────────────────

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and clean CMJ + IMTP. Returns (cmj_df, imtp_df)."""
        self._cmj = self._load_test_type("CMJ")
        self._imtp = self._load_test_type("IMTP")
        return self._cmj, self._imtp

    def merge_into_sessions(
        self, sessions: pd.DataFrame
    ) -> pd.DataFrame:
        """As-of merge CMJ + IMTP into the session DataFrame.

        For each (Pitcher, Date) session row, attach the most recent
        VALD test of each type occurring on or before that date.
        Missing values are left as NaN (filled later by the loader).
        """
        if self._cmj is None or self._imtp is None:
            self.load()

        sessions = sessions.copy()
        sessions["Date"] = pd.to_datetime(sessions["Date"])
        sessions = sessions.sort_values("Date").reset_index(drop=True)

        # Merge CMJ
        if self._cmj is not None and not self._cmj.empty:
            sessions = self._asof_merge(sessions, self._cmj, CMJ_FEATURES, "CMJ")

        # Merge IMTP
        if self._imtp is not None and not self._imtp.empty:
            sessions = self._asof_merge(sessions, self._imtp, IMTP_FEATURES, "IMTP")

        return sessions

    # ────────────────── internals ──────────────────

    def _load_test_type(self, test_type: str) -> pd.DataFrame:
        """Find and concatenate all VALD CSVs matching a test type."""
        pattern = os.path.join(
            self.data_dir, "**", f"*{test_type}*.csv"
        )
        paths = sorted(glob.glob(pattern, recursive=True))

        if not paths:
            logger.warning("No %s CSVs found in %s", test_type, self.data_dir)
            return pd.DataFrame()

        frames = []
        for path in paths:
            try:
                # VALD files have a BOM and quoted columns
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
                frames.append(df)
                logger.info(
                    "Loaded %s  (%d rows)", os.path.basename(path), len(df)
                )
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Strip whitespace from column names (VALD adds trailing spaces)
        df.columns = [c.strip() for c in df.columns]

        # Normalize names to TrackMan format
        df["Pitcher"] = df["Name"].apply(normalize_vald_name)

        # Parse date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Pitcher"])

        # Filter to test type (defensive — files should already be type-pure)
        if "Test Type" in df.columns:
            df = df[df["Test Type"].str.upper() == test_type.upper()]

        # Numeric coercion for all metric columns
        non_numeric = {"Name", "ExternalId", "Test Type", "Date", "Time",
                       "Tags", "Pitcher"}
        for col in df.columns:
            if col not in non_numeric:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # If a pitcher has multiple tests on the same day, average them
        group_cols = ["Pitcher", "Date"]
        numeric_cols = [c for c in df.columns if c not in non_numeric and c not in group_cols]
        df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()

        logger.info(
            "%s data: %d rows, %d unique pitchers",
            test_type, len(df), df["Pitcher"].nunique(),
        )
        return df

    def _asof_merge(
        self,
        sessions: pd.DataFrame,
        vald: pd.DataFrame,
        feature_cols: List[str],
        prefix: str,
    ) -> pd.DataFrame:
        """Per-pitcher as-of merge.

        For each session, attach the most recent VALD test that
        happened on or before the session date for that pitcher.
        """
        # Only keep features that actually exist in the VALD data
        available = [c for c in feature_cols if c in vald.columns]
        if not available:
            logger.warning("No %s features found in VALD data", prefix)
            return sessions

        vald_subset = vald[["Pitcher", "Date"] + available].copy()
        vald_subset = vald_subset.sort_values("Date").reset_index(drop=True)

        # Rename to add prefix so CMJ + IMTP columns don't collide
        rename_map = {c: f"{prefix}_{c}" for c in available}
        vald_subset = vald_subset.rename(columns=rename_map)

        # Days-since-last-test (so the network knows freshness)
        merged = pd.merge_asof(
            sessions.sort_values("Date"),
            vald_subset,
            on="Date",
            by="Pitcher",
            direction="backward",
            allow_exact_matches=True,
        )

        # Track how many sessions got a match
        prefixed_cols = [f"{prefix}_{c}" for c in available]
        n_matched = merged[prefixed_cols[0]].notna().sum()
        logger.info(
            "%s merge: %d/%d sessions matched to a prior test",
            prefix, n_matched, len(merged),
        )

        return merged