# Store Demand Forecasting with Hybrid Neural Networks

Forecasting daily item demand for 10 stores and 50 items using classical time-series models and machine learning. The notebook `main_upd.ipynb` builds per-series baselines, a global ML model with lag features, and LSTM sequence models, then generates multi-step forecasts for the test period.

---

## Project overview

The goal is to predict future sales for each store-item pair based on historical daily sales. This repository provides:

- Per-series ARIMA baselines
- A global XGBoost model using lag and rolling features
- Per-series LSTM sequence modeling

All approaches generate forecasts for the `test.csv` horizon and save outputs to the `output/` folder.

---

## What is in `main_upd.ipynb`

### 1) ARIMA exploration (single series)
- Filters one store-item series to validate modeling choices
- Uses ADF, ACF, PACF to assess stationarity and autocorrelation
- Splits last 30 points as validation and evaluates MAE
- Performs a small grid search over $(p,d,q)$ for ARIMA

### 2) ARIMA for all store-item pairs
- Fits ARIMA(1,1,0) per store-item group
- Forecasts the full test horizon for each group
- Saves predictions to `output/ARIMA_out.csv`

### 3) XGBoost with lag and rolling features
- Builds time-series features per store-item:
  - Lags: 1, 7, 14
  - Rolling means: 7, 28 (using only past values)
  - Calendar features: day of week, month, day
- Trains a single global XGBoost model on $\log(1 + sales)$
- Recursively forecasts each store-item series
- Saves predictions to `output/XGB_out.csv`

### 4) LSTM per store-item series
- Builds 30-step sequences for each store-item group
- Trains a small LSTM model per group
- Recursively forecasts into the test horizon
- Saves predictions to `output/LSTM_out.csv`

---

## Dataset

Files are in the `data/` folder:

- `train.csv` with columns: `date`, `store`, `item`, `sales`
- `test.csv` with columns: `date`, `store`, `item`

Both files are expected to be daily time series for each store-item pair.

---

## Outputs

Generated predictions are written to:

- `output/ARIMA_out.csv`
- `output/XGB_out.csv`
- `output/LSTM_out.csv`

Each output file includes: `date`, `store`, `item`, `sales`.

---

## How to run

1. Open `main_upd.ipynb` in VS Code or Jupyter.
2. Update file paths if needed (the notebook uses absolute Windows paths).
3. Run cells top to bottom to generate forecasts.

---

## Tech stack

- Python
- Pandas, NumPy
- scikit-learn
- statsmodels
- XGBoost
- TensorFlow / Keras

---

## Notes

- `data/` and `output/` are ignored by Git as configured in `.gitignore`.
- The ARIMA grid search is intentionally small for speed.

