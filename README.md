# Regime-Adaptive Temporal Fusion for Macroeconomic-Aware Volatility Forecasting

Code for CIKM 2026 submission.

## Installation
pip install -r requirements.txt

## Data
Download SPY OHLCV from Yahoo Finance and macro series 
(VIXCLS, DGS10, BAMLH0A0HYM2) from FRED. 
Place in the base directory and update `base_path` in each script.

## Usage
Run scripts in order:
1_data_preprocess.py  → feature engineering and windowing
2_model_Trans_TCN.py  → train and evaluate Trans-TCN
3_baselines.py        → train all baselines
4_ablation.py         → ablation study
5_multi_seed.py       → multi-seed stability
6_sensitivity.py      → window size sensitivity
7_param_sensitivity.py → hyperparameter sensitivity
