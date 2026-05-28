"""
settings.py — Central configuration for the pitcher strike-% model.

All pitcher categories, feature definitions, model hyperparameters,
and TrackMan column mappings live here so nothing is scattered.

v2: Roster-only model
  - Removed SessionNum from CONTEXT_FEATURES (not actionable)
  - Removed CATEGORY_FEATURES from FEATURE_COLS (label, not input)
  - Added GMU_ROSTER set for filtering
  - Scaled down architecture for smaller roster-only dataset

v3: CMJ integration
  - Added CMJ/IMTP features with native NaN handling in LightGBM
  - IMTP reduced to 3 features (Peak Force/BM, Force at 200ms/BM, RFD-200ms)

v4: Feature cleanup per supervisor review
  - Removed PitchTypes (not actionable for training/decision-making)
  - Removed DaysSinceLast (scheduling variable, adds noise)
  - Removed pitcher_prior_strike_pct (too similar to outcome variable)
  - Removed VertBreak_Avg (keep IVB_Avg only, per prior meetings)
  - Removed RelHeight_Avg (per prior meetings)
  - Expanded CMJ from 5 to 11 features per supervisor's full list
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

# ─────────────────────────────────────────────────────────────────────
# Pitcher categories (coach-assigned, 2025 season)
#   "elite" = both fastball AND off-speed command
# ─────────────────────────────────────────────────────────────────────
PITCHER_CATEGORIES: Dict[str, str] = {
    "Cassedy, Brandon":     "elite",
    "Meeks, Gardner":       "elite",
    "Lavin, Sam":           "elite",
    "O'Hara, Connor":       "elite",
    "Elliot, Daniel":       "elite",
    "Knox, Connor":         "elite",
    "Rumberg, Logan":       "elite",
    "Uchman, Ty":           "offspeed",
    "Stewart, Owen":        "offspeed",
    "McCarthy, Dylan":      "offspeed",
    "Okeeffe, Shaun":       "offspeed",
    "Egan, Cole":           "fastball",
    "Bilo, Michael":        "fastball",
    "Ament, Ryan":          "fastball",
    "Yount, Britt":         "fastball",
    "Drumm, Jake":          "fastball",
    "Madigan III, Michael": "fastball",
}

# Convenient set for filtering raw data to roster only
GMU_ROSTER: Set[str] = set(PITCHER_CATEGORIES.keys())

CATEGORY_LIST: List[str] = ["elite", "offspeed", "fastball", "unknown"]

# Elite strike-% threshold (from coaching staff)
ELITE_THRESHOLD: float = 0.61

# ─────────────────────────────────────────────────────────────────────
# TrackMan column mappings
# ─────────────────────────────────────────────────────────────────────
TRACKMAN_FASTBALL_TYPES = [
    "Fastball", "Four-Seam", "Sinker", "TwoSeamFastball",
    "TwoSeamFastBall", "Cutter",
]

TRACKMAN_STRIKE_CALLS = [
    "StrikeCalled", "StrikeSwinging", "FoulBall",
    "FoulBallNotFieldable", "FoulBallFieldable",
    "InPlay", "FoulTip",
]

TRACKMAN_BALL_CALLS = [
    "BallCalled", "BallinDirt", "HitByPitch",
    "BallIntentional",
]

# ─────────────────────────────────────────────────────────────────────
# Session-level features (engineered from raw pitch data)
#
# v4 removals:
#   - PitchTypes: not actionable for training or decision-making
#   - DaysSinceLast: scheduling variable, adds noise
#   - pitcher_prior_strike_pct: too similar to outcome variable
#   - VertBreak_Avg: removed per prior meetings (keep IVB only)
#   - RelHeight_Avg: removed per prior meetings
# ─────────────────────────────────────────────────────────────────────

# --- Core features (session aggregates) ---
# v4: RelHeight_Avg removed per supervisor request
CORE_FEATURES: List[str] = [
    "RelSpeed_Avg",
    "FB_Velo_Avg",
    "SpinRate_Avg",
    "Extension_Avg",
    "IVB_Avg",
    "HorzBreak_Avg",
    "SpinAxis_Avg",
    # "RelHeight_Avg",  — REMOVED v4: per prior meeting discussion
]

# --- Extended features ---
# v4: VertBreak_Avg, PitchTypes removed per supervisor request
EXTENDED_FEATURES: List[str] = [
    "FB_Velo_Max",
    "SpinRate_FB",
    # "VertBreak_Avg",  — REMOVED v4: keep IVB only, per prior meetings
    "RelSide_Avg",
    "VertApprAngle_Avg",
    "HorzApprAngle_Avg",
    "EffVelo_Avg",
    "PitchCount",
    # "PitchTypes",     — REMOVED v4: not actionable for training/decisions
    "FB_pct",
    "VeloFade",
]

# --- Category one-hot (kept for Loader compatibility, NOT in FEATURE_COLS) ---
CATEGORY_FEATURES: List[str] = [f"cat_{c}" for c in CATEGORY_LIST]

# --- Lag / rolling temporal features ---
LAG_BASE_COLS: List[str] = [
    "FB_Velo_Avg", "SpinRate_Avg", "Strike_pct", "Zone_pct", "Whiff_pct",
]

# lag1 = previous session's value (kept for all base cols)
LAG_FEATURES: List[str] = [f"{c}_lag1" for c in LAG_BASE_COLS]

# Rolling features split by metric type to avoid autocorrelation shortcut:
#   Velocity/spin: rolling MEAN (level matters, no leakage risk)
#   Command metrics: COEFFICIENT OF VARIATION (std/mean) — captures
#     CONSISTENCY rather than LEVEL, preventing the model from taking
#     the shortcut of predicting near a pitcher's recent average.
ROLL_MEAN_COLS: List[str] = ["FB_Velo_Avg", "SpinRate_Avg"]
ROLL_COV_COLS: List[str] = ["Strike_pct", "Zone_pct", "Whiff_pct"]

ROLLING_FEATURES: List[str] = (
    [f"{c}_roll3" for c in ROLL_MEAN_COLS]       # rolling mean
    + [f"{c}_roll3_cov" for c in ROLL_COV_COLS]  # rolling coefficient of variation
)

# --- Context features ---
# v4: DaysSinceLast removed (scheduling variable, adds noise)
#     pitcher_prior_strike_pct removed (too similar to outcome variable)
CONTEXT_FEATURES: List[str] = []

# ─────────────────────────────────────────────────────────────────────
# VALD CMJ features (countermovement jump — lower-body explosive power)
#
# v4: Expanded to 11 features per supervisor's full list.
#   Original 5:
#     Jump Height, Peak Power/BM, RSI-modified, CM Depth, Ecc:Con Ratio
#   Added 6 per supervisor request:
#     Concentric Mean Power/BM, Concentric Peak Force,
#     Eccentric Deceleration Mean Force, Eccentric Peak Power/BM,
#     Eccentric Peak Velocity, Peak Landing Force/BM
# ─────────────────────────────────────────────────────────────────────
CMJ_FEATURES: List[str] = [
    # Original 5
    "Jump Height (Imp-Mom) in Inches [in]",
    "Peak Power / BM [W/kg]",
    "RSI-modified (Imp-Mom) [m/s]",
    "Countermovement Depth [cm]",
    "Eccentric:Concentric Mean Force Ratio [%]",
    # Added 6 per supervisor request (v4)
    "Concentric Mean Power / BM [W/kg]",
    "Concentric Peak Force [N]",
    "Eccentric Deceleration Mean Force [N]",
    "Eccentric Peak Power / BM [W/kg]",
    "Eccentric Peak Velocity [m/s]",
    "Peak Landing Force / BM [N/kg]",
]

# ─────────────────────────────────────────────────────────────────────
# VALD IMTP features (isometric mid-thigh pull — max strength)
# ─────────────────────────────────────────────────────────────────────
IMTP_FEATURES: List[str] = [
    "Peak Vertical Force / BM [N/kg]",
    "Force at 200ms / BM [N/kg]",
    "RFD - 200ms [N/s]",
]

# Prefixed column names that the loader will produce
CMJ_FEATURE_COLS: List[str] = [f"CMJ_{c}" for c in CMJ_FEATURES]
IMTP_FEATURE_COLS: List[str] = [f"IMTP_{c}" for c in IMTP_FEATURES]

# --- Sprint / 60-yard features ---
SIXTY_TIMES_FEATURES: List[str] = [
    "Sprint60_Time",
    "Sprint30_Time",
    "Sprint60_MPH",
    "Sprint60_Improvement",
]

# --- Full feature vector ---
# v4: PitchTypes, DaysSinceLast, pitcher_prior_strike_pct,
#     VertBreak_Avg, RelHeight_Avg removed
#     CMJ expanded to 11 features
FEATURE_COLS: List[str] = (
    CORE_FEATURES
    + EXTENDED_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + CONTEXT_FEATURES
    + CMJ_FEATURE_COLS
    + IMTP_FEATURE_COLS
    + SIXTY_TIMES_FEATURES
)

TARGET_COL: str = "Strike_pct"


# ─────────────────────────────────────────────────────────────────────
# Model hyperparameters — scaled for roster-only dataset (~100 rows)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ModelParams:
    """All neural network and training hyperparameters in one place."""

    # Architecture — smaller for roster-only data
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float = 0.35
    use_batch_norm: bool = True
    use_residual: bool = True

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 32
    max_epochs: int = 500
    early_stop_patience: int = 40

    # Data split
    test_size: float = 0.20
    val_size: float = 0.15
    random_seed: int = 42

    # LR scheduler
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-6


MODEL_PARAMS = ModelParams()