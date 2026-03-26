# Credit Card Fraud Detection (Unsupervised Anomaly Detection)

This project investigates credit card fraud detection using **unsupervised anomaly detection**. Because fraud events are extremely rare, the goal is to learn the structure of *normal* transactions and flag deviations that may correspond to fraud.

## What’s inside

- `Anomaly_Detection.ipynb`
  - Data loading and integrity checks
  - Exploratory Data Analysis (EDA) of `Amount` and `Time`
  - Class-imbalance inspection (for offline evaluation)
  - Feature standardization
  - Representation diagnostics with PCA (visual inspection)
  - Unsupervised modeling:
    - `IsolationForest`
    - `OneClassSVM`
  - Evaluation vs. the known `Class` labels (offline research comparison)
  - Post-model interpretation:
    - How the detected anomalies differ in `Amount` and `Time`

## Dataset

This notebook expects a file named `creditcard.csv` in the same directory.

**Dataset source:** [Kaggle](https://www.kaggle.com/) (Credit Card Fraud Detection dataset commonly distributed as `creditcard.csv`).

**Dataset excluded from GitHub:** `creditcard.csv` is intentionally not committed/pushed to GitHub to keep the repository lightweight and avoid uploading proprietary or large data.

- `Amount`: transaction amount
- `Time`: seconds elapsed since the first transaction in the dataset
- `Class`: label (`1` = fraud, `0` = non-fraud)

> Note: The models are unsupervised, but evaluation uses `Class` as a reference to estimate recall/precision trade-offs.

## Project category

- **Machine Learning → Unsupervised Anomaly Detection**
- Domain: **Fraud Detection**

## How to run

1. Download `creditcard.csv` from Kaggle.
2. Place `creditcard.csv` in this folder (`ML Projects\classification\fraud-credit-detection`).
2. Start Jupyter Notebook (or JupyterLab).
3. Open `Anomaly_Detection.ipynb`.
4. Run all cells top-to-bottom.

## Dependencies

The notebook uses common Python ML/data libraries:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

## Models used (and key settings)

This notebook trains two unsupervised models and then maps their anomaly outputs onto your known labels (`Class`) for offline evaluation:

1. `IsolationForest`
   - `n_estimators=100`
   - `contamination='auto'`
   - `random_state=42`
   - Output mapping:
     - `1` in `IsolationForest` is converted to `0` (non-fraud)
     - `-1` is converted to `1` (fraud)

2. `OneClassSVM`
   - `nu=0.01`
   - `kernel='rbf'`
   - `gamma=0.01`
   - Output mapping:
     - `1` in `OneClassSVM` is converted to `0` (non-fraud)
     - `-1` is converted to `1` (fraud)

## Method summary (research-style)

1. **EDA** checks the marginal distributions of `Amount` and `Time`, motivating anomaly detection.
2. **Standardization** makes feature scales comparable (important for distance/boundary-based methods).
3. **PCA diagnostics** provide qualitative visualization of whether anomalies appear in lower-density regions.
4. **Unsupervised training**:
   - `IsolationForest` isolates rare points by random partitioning.
   - `OneClassSVM` learns a decision boundary around the normal region.
5. **Evaluation** uses the known label `Class` to compare models using:
   - confusion matrix and classification report (precision/recall/F1)
   - a recall-style detection percentage computed from `confusion_matrix(y_true, y_pred)[1, 1]` (true positives) and `confusion_matrix(y_true, y_pred)[1, 0]` (false negatives)
6. **Interpretation** compares detected anomalies vs. normal predictions using `Amount` and `Time`.

## Expected outputs when you run the notebook

Running top-to-bottom typically produces:

- Plots
  - `Amount` histogram + KDE
  - `Time` histogram + KDE (bimodality / cyclic behavior intuition)
  - Class distribution bar plot (high imbalance visualization)
  - Correlation heatmap of standardized features
  - PCA cumulative explained variance curve
  - PCA 3D scatter plot (qualitative separation: dense normal cloud vs dispersed anomalies)
  - Predicted-class comparison plots:
    - `Amount` distribution for predicted anomalies vs non-anomalies (from `iso_forest_pred`)
    - `Hour of day` distribution comparison (derived from `Time`)

- Console outputs
  - `Isolation Forest Classification Report` + `Isolation Forest Confusion Matrix`
  - `One-Class SVM Classification Report` + `One-Class SVM Confusion Matrix`
  - Recall/detection percentage printed using the confusion matrix values

- New dataframe artifact
  - `df['iso_forest_pred']` containing `0/1` predictions from the Isolation Forest run

## Notes / limitations

- Offline evaluation against `Class` does not guarantee real-world performance; deployment requires threshold calibration and monitoring for drift.
- Hyperparameters are set for initial comparison and should be tuned for production.

