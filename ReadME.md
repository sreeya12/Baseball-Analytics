# GMU Pitcher Development — Strike % Model

## Project Structure

```
gmu-pitcher-nn/
├── config/
│   ├── __init__.py
│   ├── settings.py          # All constants, categories, thresholds, feature lists
│   └── logging_config.py    # Structured logging setup
├── data/
│   ├── __init__.py
│   └── loader.py            # CSV ingestion + session-level feature engineering
├── models/
│   ├── __init__.py
│   ├── network.py           # PyTorch neural network (MLP + residual connections)
│   └── trainer.py           # Training loop, evaluation, cross-validation
├── utils/
│   ├── __init__.py
│   └── visualizations.py    # All plotting / reporting utilities
├── outputs/                  # Generated artifacts (plots, CSVs, model weights)
├── main.py                   # CLI entrypoint — full pipeline
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Basic run — point at your TrackMan CSV directory
python main.py --data-dir /path/to/trackman/csvs

# Custom output location
python main.py --data-dir ./data --output-dir ./results

# Skip cross-validation for faster runs
python main.py --data-dir ./data --cv-folds 0

# Override hyperparameters
python main.py --data-dir ./data --epochs 300 --lr 0.0005 --batch-size 32
```
## Architecture

```
Input(36)
  → BatchNorm
  → Linear(256) → BN → ReLU → Dropout(0.3)  + residual skip
  → Linear(128) → BN → ReLU → Dropout(0.3)  + residual skip
  → Linear(64)  → BN → ReLU → Dropout(0.3)  + residual skip
  → Linear(32)  → BN → ReLU
  → Linear(1)   → Sigmoid
```

- **Residual connections**: Skip connections with learned projections
  between layers to prevent vanishing gradients in the deeper network
- **Loss**: MSE + L2 regularization (weight decay)
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Early stopping**: Patience of 40 epochs on validation loss
- **Gradient clipping**: Max norm 1.0

## Outputs

After training, the pipeline generates:

- `pitcher_predictions.csv` — per-pitcher actual vs predicted strike %
- `loss_curves.png` — training and validation loss over epochs
- `actual_vs_predicted.png` — scatter plot with R² and MAE
- `pitcher_bars.png` — horizontal bar chart by category
- `category_boxplot.png` — distribution comparison across categories
- `elite_threshold_gap.png` — distance from 61% elite threshold
- `strike_pct_model.pt` — saved model checkpoint (weights + scaler)
