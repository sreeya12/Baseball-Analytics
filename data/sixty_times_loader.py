"""
sixty_times_loader.py — Load BASE 60 TIMES Excel file.

Parses the multi-row-header sprint timing spreadsheet, normalizes athlete
names to "Last, First" format (matching TrackMan), and returns a DataFrame
with the most-recent 60-yard time, 30-yard time, and improvement over the
season for each athlete.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


def _normalize_name(name: str) -> str:
    """Convert 'First Last' (or 'First Middle Last') → 'Last, First Middle'."""
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    return parts[-1] + ", " + " ".join(parts[:-1])


class SixtyTimesLoader:
    """Parse BASE 60 TIMES .xlsx and expose sprint features per athlete."""

    def __init__(self, path: str):
        self.path = path
        self._df: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Return a DataFrame indexed by normalized pitcher name."""
        raw = pd.read_excel(self.path, header=None, engine="openpyxl")

        # ── Find the sub-header row that contains "ATHLETE" ──
        header_row = None
        athlete_col = None
        for r in range(min(6, len(raw))):
            for c in range(min(30, len(raw.columns))):
                cell = str(raw.iloc[r, c]).strip().upper()
                if cell == "ATHLETE":
                    header_row = r
                    athlete_col = c
                    break
            if header_row is not None:
                break

        if header_row is None:
            logger.warning("Could not find ATHLETE column in %s — skipping", self.path)
            self._df = pd.DataFrame()
            return self._df

        # ── Identify 60-yard time columns and 30-yard time columns ──
        col_labels = [
            str(v).strip().lower() for v in raw.iloc[header_row]
        ]
        sixty_time_cols = [
            c for c, lbl in enumerate(col_labels)
            if "60 yard" in lbl and "mph" not in lbl and "diff" not in lbl
        ]
        thirty_time_cols = [
            c for c, lbl in enumerate(col_labels)
            if "30 yard" in lbl and "mph" not in lbl and "diff" not in lbl
        ]
        sixty_mph_cols = [
            c for c, lbl in enumerate(col_labels)
            if ("mph" in lbl and "60" in lbl) or ("mph (60" in lbl)
        ]

        logger.info(
            "60-times: header row=%d, 60yd cols=%s, 30yd cols=%s",
            header_row, sixty_time_cols, thirty_time_cols,
        )

        # ── Extract data rows ──
        data = raw.iloc[header_row + 1:].copy()

        def _is_athlete_row(val: str) -> bool:
            v = val.strip().lower()
            return (
                len(v) > 1
                and v not in ("nan", "", "average", "avg")
                and not v.startswith("nan")
                and not v[0].isdigit()
            )

        records = []
        for _, row in data.iterrows():
            raw_name = str(row.iloc[athlete_col])
            if not _is_athlete_row(raw_name):
                continue

            norm_name = _normalize_name(raw_name)

            def _vals(cols):
                return [
                    v for c in cols
                    if c < len(row)
                    for v in [pd.to_numeric(row.iloc[c], errors="coerce")]
                    if not (isinstance(v, float) and np.isnan(v))
                ]

            times_60 = _vals(sixty_time_cols)
            times_30 = _vals(thirty_time_cols)
            mph_60   = _vals(sixty_mph_cols)

            records.append({
                "Pitcher": norm_name,
                "Sprint60_Time": times_60[-1] if times_60 else np.nan,
                "Sprint30_Time": times_30[-1] if times_30 else np.nan,
                "Sprint60_MPH":  mph_60[-1]   if mph_60   else np.nan,
                "Sprint60_Improvement": (
                    (times_60[-1] - times_60[0]) if len(times_60) >= 2 else np.nan
                ),
            })

        self._df = pd.DataFrame(records)
        logger.info(
            "Loaded %d athletes from %s", len(self._df), os.path.basename(self.path)
        )
        return self._df

    def merge_into_sessions(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Left-join sprint features into session DataFrame on Pitcher name."""
        if self._df is None or self._df.empty:
            return sessions

        feat_cols = ["Pitcher", "Sprint60_Time", "Sprint30_Time",
                     "Sprint60_MPH", "Sprint60_Improvement"]
        merged = sessions.merge(
            self._df[feat_cols], on="Pitcher", how="left"
        )
        matched = merged["Sprint60_Time"].notna().sum()
        logger.info(
            "60-times merge: %d / %d sessions matched sprint data",
            matched, len(merged),
        )
        return merged
