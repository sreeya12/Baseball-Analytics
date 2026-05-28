#!/usr/bin/env python3
"""
main.py — GMU Pitcher Development: Strike % Neural Network Pipeline (v2)

v2 changes:
  - Filters training data to GMU rostered pitchers only
  - Exports session-level dataset CSV for staff inspection
  - Prints roster coverage diagnostics

Usage:
    python main.py --data-dir /path/to/trackman/csvs
    python main.py --data-dir ./data --cv-folds 5 --no-save

Full pipeline:
    1. Load & clean all TrackMan CSVs from --data-dir
    2. Filter to GMU roster pitchers only
    3. Engineer session-level features (pitcher × date)
    4. Train PyTorch neural network (train/val/test split)
    5. Optionally run K-fold cross-validation
    6. Generate predictions, reports, and visualizations
    7. Save model checkpoint + session dataset CSV
"""

import argparse
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PARAMS, ELITE_THRESHOLD, get_logger
from config.Settings import GMU_ROSTER, PITCHER_CATEGORIES
from data.Loader import TrackManLoader
from models.Trainer import SklearnTrainer, Trainer
from utils.Visualizations import ReportGenerator

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GMU Pitcher Strike-% Neural Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing TrackMan CSV files",
    )
    parser.add_argument(
        "--vald-dir",
        type=str,
        default=None,
        help="Directory containing VALD CMJ/IMTP CSV files (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (default: <data-dir>/nn_outputs)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (0 to skip CV)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the model checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["nn", "ridge", "lgbm"],
        help=(
            "Model type to train. "
            "'lgbm' (default) = LightGBM gradient-boosted trees; "
            "'ridge' = regularised linear regression; "
            "'nn' = PyTorch neural network (original)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    output_dir = args.output_dir or os.path.join(args.data_dir, "nn_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ── Override hyperparams if requested ──
    if args.epochs:
        MODEL_PARAMS.max_epochs = args.epochs
    if args.lr:
        MODEL_PARAMS.learning_rate = args.lr
    if args.batch_size:
        MODEL_PARAMS.batch_size = args.batch_size

    model_label = {"nn": "Neural Network", "ridge": "Ridge Regression", "lgbm": "LightGBM"}[args.model]
    print()
    print("=" * 70)
    print(f"  GMU PITCHER DEVELOPMENT — STRIKE % MODEL (v3)")
    print(f"  Model:           {model_label}")
    print(f"  Elite threshold: {ELITE_THRESHOLD * 100:.0f}%")
    print(f"  Mode:            ROSTER-ONLY ({len(GMU_ROSTER)} pitchers)")
    print("=" * 70)

    # ── 1. Load data ──
    logger.info("Loading TrackMan data from %s", args.data_dir)
    if args.vald_dir:
        logger.info("Loading VALD data from %s", args.vald_dir)
    loader = TrackManLoader(args.data_dir, vald_dir=args.vald_dir)
    model_df = loader.load()

    # ── 2. Filter to GMU roster only ──
    total_before = len(model_df)
    pitchers_before = model_df["Pitcher"].nunique()

    model_df = model_df[model_df["Pitcher"].isin(GMU_ROSTER)].copy()

    # Roster coverage diagnostics
    found = set(model_df["Pitcher"].unique()) & GMU_ROSTER
    missing = GMU_ROSTER - found

    print(f"\n  Roster filter: {total_before} -> {len(model_df)} sessions")
    print(f"  Pitchers: {pitchers_before} total -> {len(found)}/{len(GMU_ROSTER)} rostered")

    if missing:
        print(f"  ! Missing from data: {', '.join(sorted(missing))}")

    # Per-pitcher session counts
    print(f"\n  {'Pitcher':<28} {'Category':<10} {'Sessions':>8}")
    print(f"  {'-'*50}")
    counts = model_df.groupby("Pitcher").size().reset_index(name="Sessions")
    counts["Category"] = counts["Pitcher"].map(PITCHER_CATEGORIES)
    counts = counts.sort_values("Sessions", ascending=False)
    for _, row in counts.iterrows():
        print(f"  {row['Pitcher']:<28} {row['Category']:<10} {row['Sessions']:>8}")
    print()

    if len(model_df) < 20:
        logger.error(
            "Only %d sessions found — need at least 20 for meaningful training.",
            len(model_df),
        )
        sys.exit(1)

    # ── 3. Export session dataset for staff inspection ──
    session_csv_path = os.path.join(output_dir, "gmu_roster_session_data.csv")
    model_df.to_csv(session_csv_path, index=False)
    logger.info("Session dataset exported → %s (%d rows)", session_csv_path, len(model_df))

    # ── 4. Train ──
    logger.info("Training %s model …", args.model)
    if args.model == "nn":
        trainer = Trainer()
    else:
        trainer = SklearnTrainer(model_type=args.model)
    result = trainer.train(model_df)

    # ── 5. Cross-validate ──
    if args.cv_folds > 0:
        logger.info("Running %d-fold cross-validation …", args.cv_folds)
        cv = trainer.cross_validate(model_df, n_folds=args.cv_folds)
        print(f"\n  Cross-Validation ({args.cv_folds}-fold)")
        print(f"  {'─' * 45}")
        print(f"  MAE  = {cv.mae_mean * 100:.2f} ± {cv.mae_std * 100:.2f} pp")
        print(f"  RMSE = {cv.rmse_mean * 100:.2f} ± {cv.rmse_std * 100:.2f} pp")
        print(f"  R²   = {cv.r2_mean:.3f} ± {cv.r2_std:.3f}")
        print()

    # ── 6. Reports & plots ──
    logger.info("Generating reports …")
    reporter = ReportGenerator(output_dir)
    reporter.generate_all(result, model_df)

    # ── 7. Save model ──
    if not args.no_save:
        if args.model == "nn":
            ckpt_path = os.path.join(output_dir, "strike_pct_model.pt")
            trainer.save(result, ckpt_path)
        else:
            import pickle
            ckpt_path = os.path.join(output_dir, f"strike_pct_model_{args.model}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({"model": result.model, "scaler": result.scaler,
                             "feature_cols": result.feature_cols}, f)
            logger.info("Model saved → %s", ckpt_path)

    elapsed = time.time() - t0
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs -> {output_dir}")
    print()


if __name__ == "__main__":
    main()