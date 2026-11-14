# Anticipatory Solar Irradiance Forecasting

Two-layer machine learning approach that forecasts **Global Horizontal Irradiance (GHI)** and **anticipates when that forecast is likely to be wrong**, across five Saudi cities.

- **Layer 1: Baseline forecaster (Random Forest).** Predicts hourly GHI from weather and engineered features.
- **Layer 2: Meta-model (LightGBM).** Learns from the baseline's past residuals and flags **high-error windows three hours ahead**, so operators can be warned before the forecast degrades.

Cities: **Abha · Dammam · Jeddah · Qassim · Riyadh**

---

## Results (test set: 2024)

Computed from `results/meta_improved/*_improved_results.csv` (`split == "test"`).

| City | R² (GHI) | RMSE | MAE | Alert recall | Alert precision | Alert rate |
|---|---|---|---|---|---|---|
| Abha | 0.956 | 67.7 | 39.9 | 77.8% | 31.5% | 11.7% |
| Dammam | 0.976 | 49.2 | 25.5 | 52.6% | 37.4% | 5.3% |
| Jeddah | 0.977 | 49.5 | 25.9 | 49.2% | 28.2% | 5.7% |
| Qassim | 0.978 | 47.3 | 24.7 | 59.7% | 29.7% | 5.2% |
| Riyadh | 0.975 | 51.7 | 25.6 | 67.1% | 35.0% | 5.3% |

**Reading the alert metrics:** high-error windows are rare (about 5–12% of hours), so recall and precision matter more than accuracy here. The meta-model favours **recall** (catching most risky windows) at the cost of false alarms.

---

## Method

### Data split (chronological, no shuffling)
| Split | Years |
|---|---|
| Train | 2019–2022 |
| Validation | 2023 |
| Test | 2024 |

Splitting by time avoids leakage from future observations into training.

### Feature engineering
- **Temporal:** hour, month, day of year, season
- **Atmospheric:** temperature, humidity, cloud cover, pressure, precipitation, wind speed and direction (encoded as `wind_sin` / `wind_cos`)
- **Interactions:** `hour_x_cloud`, `temp_x_humidity`, `wind_speed_x_cloud`, `clear_sky_ratio`
- **Dust proxy:** `dust_proxy`
- **Trends and changes:** 3-hour trends and 1-hour changes for cloud, wind, humidity and GHI, plus `pressure_diff`
- **Meta-model inputs:** baseline residual lags (`residual_lag1` to `residual_lag3`)

### Imbalance and interpretability
- **SMOTE** to handle the rarity of high-error windows when training the meta-model
- **SHAP** to explain which features drive the alerts

---

## Repository structure

```
├── data/
│   ├── raw/                 # hourly weather + GHI per city
│   └── processed/           # engineered features with train/val/test split
├── models/
│   ├── *_baseline_rf.pkl    # Layer 1: Random Forest per city
│   ├── meta/                # Layer 2: first meta-model version (LightGBM)
│   └── meta_improved/       # Layer 2: improved meta-model (LightGBM)
├── results/
│   ├── baseline/            # baseline predictions and residuals
│   ├── meta/                # first meta-model outputs
│   └── meta_improved/       # final outputs (alert probabilities, predictions)
└── colab_all_figures/       # one notebook per figure (fig1–fig10) + exported PNGs
```

---

## Reproducing the figures

Each notebook in `colab_all_figures/` regenerates one figure from the saved results:

| Notebook | Figure |
|---|---|
| `fig1_correlation_matrix` | Feature correlation matrix |
| `fig2_dust_proxy` | Dust proxy analysis |
| `fig3_performance_comparison` | RMSE, MAE, R², improvement, recall, bias |
| `fig4_recall_threshold` | Recall vs. alert threshold |
| `fig5_alert_timeseries` | Alerts over time |
| `fig6_diurnal_error` | Error by hour of day |
| `fig7_clearsky_residual` | Clear-sky residuals |
| `fig8_seasonal_residuals` | Residuals by season |
| `fig9_radar_chart` | Per-city radar comparison |
| `fig10_alert_heatmap` | Alert heatmap |

Run cells in order. The notebooks were written for Google Colab.

---

## Tech stack
Python · pandas · scikit-learn (Random Forest) · LightGBM · imbalanced-learn (SMOTE) · SHAP · matplotlib

## Author
**Marwa Hanini**, Big Data Engineering student, EIDIA, Université Euromed de Fès
[GitHub](https://github.com/HANINIMarwa) · [LinkedIn](https://www.linkedin.com/in/marwa-hanini-9421663b5/)
