# Anticipatory Solar Irradiance Forecasting

Two-layer machine learning approach to forecast solar irradiance (GHI) and to anticipate, three hours ahead, the windows where the forecast is likely to fail. Built on five years of hourly weather data across five Saudi cities.

**Stack:** Python, Random Forest, LightGBM, SMOTE, SHAP

## Problem

Irradiance forecasts degrade during rapidly changing conditions (dust, wind and cloud variations). This project predicts irradiance and also flags the windows where the first model is likely to make large errors, so operators can be warned in advance.

## Data

- Hourly weather data, **2019–2024**, for **Riyadh, Jeddah, Abha, Dammam and Qassim**
- Variables: GHI, temperature, humidity, cloud cover, wind speed and direction, pressure, precipitation
- `data/raw/`: raw hourly series per city
- `data/processed/`: engineered features and a `split` column (train / test). The **test set is the year 2024**.

> **Data source and license:** _[to add: provider of the weather/solar data and its redistribution terms]_

## Method

1. **Layer 1: irradiance prediction.** A Random Forest baseline per city (`models/*_baseline_rf.pkl`).
2. **Layer 2: high-error window detection.** A LightGBM meta-model trained on the baseline's residuals predicts, three hours ahead, whether the next window will be a high-error window (`models/meta/` and `models/meta_improved/`).

### Feature engineering
- Temporal features: hour, month, day of year, season
- Atmospheric features and interactions: clear-sky ratio, dust proxy, hour × cloud, temperature × humidity, wind sin/cos, wind speed × cloud
- Wind, cloud, humidity and GHI trends (3 h) and 1 h changes
- Residual lags (`residual_lag1` to `residual_lag3`)

### Class imbalance and interpretation
- **SMOTE** to handle the imbalance of high-error windows
- **SHAP** to interpret the models

## Results (2024 test set)

Computed from the files in `results/`.

| City | R² (Random Forest) | RMSE (W/m²) | Recall on high-error windows | Precision |
|---|---|---|---|---|
| Riyadh | 0.9746 | 51.7 | 67.1% | 35.0% |
| Jeddah | 0.9769 | 49.5 | 49.2% | 28.2% |
| Abha | 0.9562 | 67.7 | 77.8% | 31.5% |
| Dammam | 0.9758 | 49.2 | 52.6% | 37.4% |
| Qassim | 0.9785 | 47.3 | 59.7% | 29.7% |

Recall is the share of actual high-error windows that the meta-model flagged. Precision is moderate (28–37%), so the alerts favor catching failures over avoiding false alarms.

## Repository structure

```
.
├── data/
│   ├── raw/                 # raw hourly data per city
│   └── processed/           # engineered features + train/test split
├── models/
│   ├── *_baseline_rf.pkl    # Layer 1: Random Forest baselines
│   ├── meta/                # Layer 2: LightGBM meta-models
│   └── meta_improved/       # Layer 2: improved meta-models
├── results/
│   ├── baseline/            # baseline predictions and residuals
│   ├── meta/                # meta-model results
│   └── meta_improved/       # improved meta-model results and predictions
└── colab_all_figures/       # figure notebooks and exported PNGs
```

## Status and limitations

- The repository contains the data, trained models and results. **The training scripts/notebooks are not included yet.**
- The notebooks in `colab_all_figures/` do not read the CSV files. They plot hard-coded summary values, and Figures 2, 5, 7, 8 and 10 use **simulated data for illustration** (for example, Figure 5 shows a simulated dust-storm event). They are not computed directly from `data/` or `results/`.

## Author

Marwa Hanini, Big Data Engineering student at EIDIA (Université Euromed de Fès).
LinkedIn: https://www.linkedin.com/in/marwa-hanini-9421663b5/
