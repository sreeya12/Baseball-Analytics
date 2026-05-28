# Neural Network Strike Prediction Model - Summary Report

## Project Overview
Built a neural network to predict baseball strike percentage using TrackMan pitch data from multiple datasets.

## Dataset Information
- **Total Samples**: 16,632 pitches (after cleaning)
- **Training Set**: 13,305 samples (80%)
- **Testing Set**: 3,327 samples (20%)
- **Target Variable**: Strike vs Non-Strike (Binary Classification)
- **Strike Percentage**: 58.14% (imbalanced toward strikes)
- **Features Used**: 13 pitch kinematic and trajectory features

## Data Processing
1. **Data Loading**: Combined CSV files from multiple folders (main datasets + TrackMan.csv subfolder)
2. **Strike Classification**: 
   - Strike = 1: StrikeSwinging, StrikeCalled, InPlay, FoulBall
   - Non-Strike = 0: BallCalled and other calls
3. **Feature Selection**: Removed columns with >50% missing values (e.g., Tilt was 100% null)
4. **Missing Values**: Imputed with median values
5. **Scaling**: StandardScaler normalization for optimal neural network performance

## Neural Network Architecture
```
Input Layer: 13 features
  ↓
Hidden Layer 1: 128 neurons (ReLU activation)
  ↓
Hidden Layer 2: 64 neurons (ReLU activation)
  ↓
Hidden Layer 3: 32 neurons (ReLU activation)
  ↓
Hidden Layer 4: 16 neurons (ReLU activation)
  ↓
Output Layer: 1 neuron (Sigmoid activation for binary classification)
```

## Model Configuration
- **Solver**: Adam optimizer
- **Learning Rate**: 0.001 (adaptive)
- **Max Iterations**: 500
- **Batch Size**: 32
- **Early Stopping**: Enabled with patience of 20
- **Validation Split**: 20% of training data

## Model Performance

### Training Set Metrics
- **Accuracy**: 98.92%
- **ROC-AUC Score**: 0.9860
- **Precision (Strike)**: 0.99
- **Recall (Strike)**: 0.99
- **F1-Score (Strike)**: 0.99

### Testing Set Metrics
- **Accuracy**: 93.63%
- **ROC-AUC Score**: 0.9394
- **Precision (Strike)**: 0.91
- **Recall (Strike)**: 0.90
- **F1-Score (Strike)**: 0.90

### Additional Metrics (Testing Set)
- **Mean Squared Error (MSE)**: 0.0637
- **Mean Absolute Error (MAE)**: 0.0637
- **R² Score**: 0.7892

### Strike Percentage Predictions
- **Actual Strike % (Test Set)**: 58.13%
- **Predicted Strike %**: 57.02%
- **Mean Strike Probability**: 57.15%

## Feature Importance Ranking
Based on mean absolute weights from the input layer:

1. **PlateLocSide** (0.247) - Horizontal position on home plate
2. **PlateLocHeight** (0.220) - Vertical position on home plate
3. **RelSpeed** (0.203) - Release speed of the pitch
4. **Extension** (0.195) - Distance from pitcher to plate
5. **InducedVertBreak** (0.193) - Vertical break movement
6. **SpinRate** (0.184) - Spin rate of the ball
7. **RelHeight** (0.182) - Release height
8. **HorzBreak** (0.175) - Horizontal break movement
9. **SpinAxis** (0.172) - Spin axis orientation
10. **RelSide** (0.156) - Horizontal release position
11. **VertApprAngle** (0.151) - Approach angle (vertical)
12. **VertBreak** (0.136) - Vertical break
13. **HorzApprAngle** (0.131) - Approach angle (horizontal)

## Key Insights

### Model Strengths
✓ Exceptional performance with 93.63% test accuracy
✓ High ROC-AUC (0.9394) indicates excellent discrimination
✓ Plate location (side and height) are the most important predictors
✓ Model generalizes well (only ~5% accuracy drop from training to testing)
✓ Well-separated probability distributions for strikes vs non-strikes

### Interpretation
- The model learns that where a pitch is located on home plate is crucial for strike prediction (as expected)
- Release speed, pitch movement, and spin characteristics also significantly influence strike probability
- The model provides well-calibrated probability estimates (most predictions cluster near 0.0 or 1.0)

## Model Output
- Saves predictions with strike probabilities (0-100%)
- Uses 0.5 probability threshold for binary classification
- Can be used to:
  - Predict strike probability for new pitches
  - Analyze pitcher effectiveness
  - Understand which pitch characteristics lead to strikes
  - Identify unusual pitch movement or release characteristics

## Files
- **Notebook**: `strike_prediction_neural_network.ipynb`
- **Report**: `MODEL_SUMMARY.md` (this file)

