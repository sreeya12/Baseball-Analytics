"""
trainer.py — Training loop, evaluation, cross-validation, and model I/O.

Handles:
  • Train / validation / test splitting (pitcher-grouped to prevent leakage)
  • Feature scaling (StandardScaler, persisted with the model)
  • Neural network training with early stopping and LR scheduling
  • Sklearn/LightGBM training (Ridge, LightGBM)
  • Cross-validation for robust metrics
  • Model saving / loading

Grouped splits (GroupShuffleSplit / GroupKFold keyed on Pitcher):
  A pitcher's sessions are either entirely in train or entirely in test —
  never split across both.  Without this, the model partially memorises each
  pitcher's average during training and the test error looks artificially good.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.logging_config import get_logger
from config.Settings import FEATURE_COLS, MODEL_PARAMS, TARGET_COL
from models.Network import StrikePctNetwork

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────


@dataclass
class TrainResult:
    """Holds everything produced by a single training run.

    model may be a StrikePctNetwork (nn) or any sklearn-compatible estimator
    (ridge, lgbm).  Downstream code checks isinstance(result.model, nn.Module)
    where torch-specific behaviour is needed.
    """

    model: Any                      # StrikePctNetwork OR sklearn estimator
    scaler: StandardScaler
    train_losses: List[float]       # empty for sklearn models
    val_losses: List[float]         # empty for sklearn models
    best_epoch: int                 # 0 for sklearn models
    test_metrics: Dict[str, float]
    predictions: np.ndarray         # on full dataset
    pitcher_names: np.ndarray
    feature_cols: List[str]         # ordered feature names the model was trained on
    model_type: str = "nn"          # "nn" | "ridge" | "lgbm"


@dataclass
class CVResult:
    """Aggregate cross-validation metrics."""

    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float


# ─────────────────────────────────────────────────────────────────────
# Shared metric helper
# ─────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_true - y_pred
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────
# Neural network trainer
# ─────────────────────────────────────────────────────────────────────


class Trainer:
    """Orchestrates PyTorch training, evaluation, and persistence."""

    def __init__(self, params=None):
        self.p = params or MODEL_PARAMS
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", self.device)

    # ────────────────── public API ──────────────────

    def train(self, df: pd.DataFrame) -> TrainResult:
        """Full train pipeline: grouped split → scale → fit → evaluate → predict."""
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feat_cols].values.astype(np.float64)
        y = df[TARGET_COL].values.astype(np.float64)
        pitcher_names  = df["Pitcher"].values
        pitcher_groups = df["Pitcher"].values

        # ── Pitcher-grouped split (no pitcher appears in both train and test) ──
        idx = np.arange(len(X))
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=self.p.test_size,
            random_state=self.p.random_seed,
        )
        idx_train_full, idx_test = next(
            gss_test.split(X, y, groups=pitcher_groups)
        )

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=self.p.val_size / (1 - self.p.test_size),
            random_state=self.p.random_seed + 1,
        )
        rel_train, rel_val = next(
            gss_val.split(
                X[idx_train_full], y[idx_train_full],
                groups=pitcher_groups[idx_train_full],
            )
        )
        idx_train = idx_train_full[rel_train]
        idx_val   = idx_train_full[rel_val]

        logger.info(
            "Grouped split → train=%d (%d pitchers)  val=%d  test=%d (%d pitchers)",
            len(idx_train), len(np.unique(pitcher_groups[idx_train])),
            len(idx_val),
            len(idx_test),  len(np.unique(pitcher_groups[idx_test])),
        )

        # ── Scale ──
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X[idx_train])
        X_val   = scaler.transform(X[idx_val])
        X_test  = scaler.transform(X[idx_test])
        X_all   = scaler.transform(X)

        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        # ── DataLoaders ──
        train_dl = self._make_loader(X_train, y_train, shuffle=True)
        val_dl   = self._make_loader(X_val,   y_val,   shuffle=False)

        # ── Model ──
        n_features = X_train.shape[1]
        model = StrikePctNetwork(n_features=n_features).to(self.device)
        logger.info("Model params: %s", f"{self._count_params(model):,}")

        # ── Optimizer + scheduler ──
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.p.learning_rate,
            weight_decay=self.p.weight_decay,
        )
        scheduler  = self._build_scheduler(optimizer)
        criterion  = nn.MSELoss()

        # ── Training loop ──
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_state    = None
        best_epoch    = 0
        patience_counter = 0

        for epoch in range(1, self.p.max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_losses.append(epoch_loss / len(idx_train))

            val_loss = self._evaluate_loss(model, val_dl, criterion)
            val_losses.append(val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch       = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 50 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %3d | train=%.6f  val=%.6f  lr=%.2e",
                    epoch, train_losses[-1], val_loss, lr_now,
                )

            if patience_counter >= self.p.early_stop_patience:
                logger.info(
                    "Early stop at epoch %d (best=%d, val=%.6f)",
                    epoch, best_epoch, best_val_loss,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(self.device)

        # ── Metrics & predictions ──
        test_metrics = _compute_metrics(y_test, self._predict_numpy(model, X_test))
        logger.info(
            "Test → MAE=%.4f  RMSE=%.4f  R²=%.4f",
            test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )

        predictions = self._predict_numpy(model, X_all)

        return TrainResult(
            model=model,
            scaler=scaler,
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            test_metrics=test_metrics,
            predictions=predictions,
            pitcher_names=pitcher_names,
            feature_cols=feat_cols,
            model_type="nn",
        )

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 5) -> CVResult:
        """Pitcher-grouped K-fold cross-validation."""
        feat_cols      = [c for c in FEATURE_COLS if c in df.columns]
        X              = df[feat_cols].values.astype(np.float64)
        y              = df[TARGET_COL].values.astype(np.float64)
        pitcher_groups = df["Pitcher"].values

        gkf = GroupKFold(n_splits=n_folds)
        mae_list, rmse_list, r2_list = [], [], []

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=pitcher_groups), 1
        ):
            scaler  = StandardScaler()
            X_tr    = scaler.fit_transform(X[train_idx])
            X_te    = scaler.transform(X[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]

            model = StrikePctNetwork(n_features=X_tr.shape[1]).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.p.learning_rate,
                weight_decay=self.p.weight_decay,
            )
            criterion = nn.MSELoss()
            train_dl  = self._make_loader(X_tr, y_tr, shuffle=True)

            model.train()
            for _ in range(min(200, self.p.max_epochs)):
                for xb, yb in train_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = criterion(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            metrics = _compute_metrics(y_te, self._predict_numpy(model, X_te))
            mae_list.append(metrics["mae"])
            rmse_list.append(metrics["rmse"])
            r2_list.append(metrics["r2"])
            logger.info(
                "Fold %d/%d → MAE=%.4f  RMSE=%.4f  R²=%.4f",
                fold, n_folds, metrics["mae"], metrics["rmse"], metrics["r2"],
            )

        return _make_cv_result(mae_list, rmse_list, r2_list)

    def save(self, result: TrainResult, path: str) -> None:
        """Persist model weights + scaler + config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        feat_cols = (
            [c for c in FEATURE_COLS if c in result.scaler.feature_names_in_]
            if hasattr(result.scaler, "feature_names_in_")
            else FEATURE_COLS
        )

        checkpoint = {
            "model_state":  result.model.state_dict(),
            "scaler":        result.scaler,
            "feature_cols":  feat_cols,
            "best_epoch":    result.best_epoch,
            "test_metrics":  result.test_metrics,
            "model_config": {
                "n_features":   len(feat_cols),
                "hidden_layers": self.p.hidden_layers,
                "dropout_rate":  self.p.dropout_rate,
                "use_batch_norm": self.p.use_batch_norm,
                "use_residual":  self.p.use_residual,
            },
        }
        torch.save(checkpoint, path)
        logger.info("Model saved → %s", path)

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> Tuple[StrikePctNetwork, StandardScaler]:
        """Load a saved checkpoint and return (model, scaler)."""
        dev  = torch.device(device or "cpu")
        ckpt = torch.load(path, map_location=dev, weights_only=False)

        cfg   = ckpt["model_config"]
        model = StrikePctNetwork(
            n_features=cfg["n_features"],
            hidden_layers=cfg["hidden_layers"],
            dropout_rate=cfg["dropout_rate"],
            use_batch_norm=cfg.get("use_batch_norm", True),
            use_residual=cfg.get("use_residual", True),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(dev).eval()

        return model, ckpt["scaler"]

    # ────────────────── private helpers ──────────────────

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.p.batch_size, shuffle=shuffle)

    def _evaluate_loss(self, model: nn.Module, dl: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += criterion(model(xb), yb).item() * len(xb)
                n += len(xb)
        return total / max(n, 1)

    def _predict_numpy(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return model(t).squeeze().cpu().numpy()

    def _build_scheduler(self, optimizer):
        if self.p.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.p.max_epochs, eta_min=self.p.min_lr
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    @staticmethod
    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────
# Sklearn / LightGBM trainer
# ─────────────────────────────────────────────────────────────────────


class SklearnTrainer:
    """Train Ridge or LightGBM with pitcher-grouped splits.

    Why these models beat a neural net at 139 sessions:
      - Ridge: linear model with strong L2 regularisation; very stable on
        small data, interpretable via coefficients.
      - LightGBM: gradient-boosted trees; handles non-linearities, missing
        values natively, and generalises well with proper min_child_samples.
    Both avoid the neural-net tendency to collapse predictions toward the
    dataset mean when training data is scarce.
    """

    def __init__(self, model_type: str = "lgbm", params=None):
        if model_type not in ("ridge", "lgbm"):
            raise ValueError(f"model_type must be 'ridge' or 'lgbm', got {model_type!r}")
        self.model_type = model_type
        self.p = params or MODEL_PARAMS

    # ────────────────── public API ──────────────────

    def train(self, df: pd.DataFrame) -> TrainResult:
        """Grouped split → scale → fit → evaluate → predict."""
        feat_cols      = [c for c in FEATURE_COLS if c in df.columns]
        X              = df[feat_cols].values.astype(np.float64)
        y              = df[TARGET_COL].values.astype(np.float64)
        pitcher_names  = df["Pitcher"].values
        pitcher_groups = df["Pitcher"].values

        # ── Pitcher-grouped train/test split ──
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.p.test_size,
            random_state=self.p.random_seed,
        )
        idx_train, idx_test = next(gss.split(X, y, groups=pitcher_groups))

        logger.info(
            "Grouped split → train=%d (%d pitchers)  test=%d (%d pitchers)",
            len(idx_train), len(np.unique(pitcher_groups[idx_train])),
            len(idx_test),  len(np.unique(pitcher_groups[idx_test])),
        )

        # ── Scale ──
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X[idx_train])
        X_test  = scaler.transform(X[idx_test])
        X_all   = scaler.transform(X)

        y_train, y_test = y[idx_train], y[idx_test]

        # ── Build and fit model ──
        model = self._build_model()
        model.fit(X_train, y_train)

        # ── Clip predictions to valid strike-% range [0, 1] ──
        y_pred_test = np.clip(model.predict(X_test), 0.0, 1.0)
        test_metrics = _compute_metrics(y_test, y_pred_test)
        logger.info(
            "Test → MAE=%.4f  RMSE=%.4f  R²=%.4f",
            test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )

        predictions = np.clip(model.predict(X_all), 0.0, 1.0)

        return TrainResult(
            model=model,
            scaler=scaler,
            train_losses=[],
            val_losses=[],
            best_epoch=0,
            test_metrics=test_metrics,
            predictions=predictions,
            pitcher_names=pitcher_names,
            feature_cols=feat_cols,
            model_type=self.model_type,
        )

    def cross_validate(self, df: pd.DataFrame, n_folds: int = 5) -> CVResult:
        """Pitcher-grouped K-fold cross-validation."""
        feat_cols      = [c for c in FEATURE_COLS if c in df.columns]
        X              = df[feat_cols].values.astype(np.float64)
        y              = df[TARGET_COL].values.astype(np.float64)
        pitcher_groups = df["Pitcher"].values

        # GroupKFold ensures each pitcher appears in exactly one test fold
        n_pitchers = len(np.unique(pitcher_groups))
        actual_folds = min(n_folds, n_pitchers)
        if actual_folds < n_folds:
            logger.warning(
                "Only %d unique pitchers — reducing CV folds from %d to %d",
                n_pitchers, n_folds, actual_folds,
            )

        gkf = GroupKFold(n_splits=actual_folds)
        mae_list, rmse_list, r2_list = [], [], []

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X, y, groups=pitcher_groups), 1
        ):
            scaler  = StandardScaler()
            X_tr    = scaler.fit_transform(X[train_idx])
            X_te    = scaler.transform(X[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]

            model = self._build_model()
            model.fit(X_tr, y_tr)

            y_pred = np.clip(model.predict(X_te), 0.0, 1.0)
            metrics = _compute_metrics(y_te, y_pred)
            mae_list.append(metrics["mae"])
            rmse_list.append(metrics["rmse"])
            r2_list.append(metrics["r2"])
            logger.info(
                "Fold %d/%d → MAE=%.4f  RMSE=%.4f  R²=%.4f",
                fold, actual_folds, metrics["mae"], metrics["rmse"], metrics["r2"],
            )

        return _make_cv_result(mae_list, rmse_list, r2_list)

    # ────────────────── private helpers ──────────────────

    def _build_model(self):
        if self.model_type == "ridge":
            from sklearn.linear_model import RidgeCV
            # RidgeCV auto-selects the best alpha via leave-one-out CV
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0])

        # lgbm
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            )
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,           # small tree depth — prevents overfitting
            min_child_samples=5,     # at least 5 sessions per leaf
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.p.random_seed,
            verbose=-1,
        )


# ─────────────────────────────────────────────────────────────────────
# Shared CV helper
# ─────────────────────────────────────────────────────────────────────

def _make_cv_result(
    mae_list: list, rmse_list: list, r2_list: list
) -> CVResult:
    result = CVResult(
        mae_mean=float(np.mean(mae_list)),
        mae_std=float(np.std(mae_list)),
        rmse_mean=float(np.mean(rmse_list)),
        rmse_std=float(np.std(rmse_list)),
        r2_mean=float(np.mean(r2_list)),
        r2_std=float(np.std(r2_list)),
    )
    logger.info(
        "CV Summary → MAE=%.4f+-%.4f  RMSE=%.4f+-%.4f  R2=%.4f+-%.4f",
        result.mae_mean, result.mae_std,
        result.rmse_mean, result.rmse_std,
        result.r2_mean, result.r2_std,
    )
    return result
