"""
1_data_preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

base_path = r"D:\CIKM"

spy        = pd.read_csv(f"{base_path}\\spy_us_d (1).csv",    parse_dates=["DATE"])
sp500      = pd.read_csv(f"{base_path}\\SP500.csv",            parse_dates=["DATE"])
vix        = pd.read_csv(f"{base_path}\\VIXCLS.csv",           parse_dates=["DATE"])
dgs10      = pd.read_csv(f"{base_path}\\DGS10.csv",            parse_dates=["DATE"])
corp_spread= pd.read_csv(f"{base_path}\\BAMLH0A0HYM2.csv",    parse_dates=["DATE"])

sp500      = sp500.rename(columns={"SP500":        "CLOSE_FRED"})
vix        = vix.rename(columns={"VIXCLS":         "VIX"})
dgs10      = dgs10.rename(columns={"DGS10":         "YIELD_10Y"})
corp_spread= corp_spread.rename(columns={"BAMLH0A0HYM2": "CORP_SPREAD"})

for d in [spy, sp500, vix, dgs10, corp_spread]:
    d.set_index("DATE", inplace=True)

df = spy.copy()
df = df.join([sp500, vix, dgs10, corp_spread], how="inner")

macro_cols = ["VIX", "YIELD_10Y", "CORP_SPREAD"]
for col in macro_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[macro_cols] = df[macro_cols].ffill()


df["LOG_HIGH_LOW"]   = np.log(df["HIGH"]  / df["LOW"])
df["LOG_CLOSE_OPEN"] = np.log(df["CLOSE"] / df["OPEN"])
df["VOLATILITY"] = (0.5  * df["LOG_HIGH_LOW"]**2
                    - (2 * np.log(2) - 1) * df["LOG_CLOSE_OPEN"]**2)
df["VOLATILITY"] = np.log(df["VOLATILITY"] + 1e-9)


df["RETURN"]     = df["CLOSE"].pct_change()
df["RETURN_ABS"] = df["RETURN"].abs()

df["VOLUME_LOG"] = np.log(df["VOLUME"] + 1)


df["MA5"]  = df["CLOSE"].rolling(5).mean()
df["MA10"] = df["CLOSE"].rolling(10).mean()

delta    = df["CLOSE"].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))


df["VIX_ROC"]         = df["VIX"].diff()                           # VIX 变动率（拐点信号）
df["RET_VOL_INTERACT"] = df["RETURN_ABS"] * df["VOLUME_LOG"]       # 趋势强度交叉项
df["RV_5"]            = df["VOLATILITY"].rolling(5).mean()         # 周级别波动记忆
df["RV_22"]           = df["VOLATILITY"].rolling(22).mean()        # 月级别波动记忆

df = df.dropna()

FEATURES = [
    "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_LOG",
    "MA5", "MA10", "RSI",
    "RETURN", "RETURN_ABS",
    "VIX", "YIELD_10Y", "CORP_SPREAD",
    "VIX_ROC", "RET_VOL_INTERACT", "RV_5", "RV_22"
]
TARGET = "VOLATILITY"

X_raw = df[FEATURES].values
y_raw = df[TARGET].values.reshape(-1, 1)

WINDOW = 60
X_windows, y_labels = [], []
for i in range(len(X_raw) - WINDOW):
    X_windows.append(X_raw[i:i + WINDOW])
    y_labels.append(y_raw[i + WINDOW])

X_windows = np.array(X_windows)   # [N, 60, 17]
y_labels  = np.array(y_labels)    # [N, 1]

total     = len(X_windows)
train_end = int(total * 0.7)
val_end   = int(total * 0.8)

X_train = X_windows[:train_end]
y_train = y_labels[:train_end]
X_val   = X_windows[train_end:val_end]
y_val   = y_labels[train_end:val_end]
X_test  = X_windows[val_end:]
y_test  = y_labels[val_end:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

n_feat = len(FEATURES)
scaler_X.fit(X_train.reshape(-1, n_feat))
scaler_y.fit(y_train)

def transform_windows(data, scaler):
    b, t, f = data.shape
    return scaler.transform(data.reshape(-1, f)).reshape(b, t, f)

X_train_s = transform_windows(X_train, scaler_X)
X_val_s   = transform_windows(X_val,   scaler_X)
X_test_s  = transform_windows(X_test,  scaler_X)

y_train_s = scaler_y.transform(y_train).squeeze()
y_val_s   = scaler_y.transform(y_val).squeeze()
y_test_s  = scaler_y.transform(y_test).squeeze()

np.save(f"{base_path}\\X_train.npy",   X_train_s)
np.save(f"{base_path}\\X_val.npy",     X_val_s)
np.save(f"{base_path}\\X_test.npy",    X_test_s)
np.save(f"{base_path}\\y_train.npy",   y_train_s)
np.save(f"{base_path}\\y_val.npy",     y_val_s)
np.save(f"{base_path}\\y_test.npy",    y_test_s)
np.save(f"{base_path}\\split_idx.npy", np.array([train_end, val_end]))

joblib.dump(scaler_X, f"{base_path}\\scaler_X.pkl")
joblib.dump(scaler_y, f"{base_path}\\scaler_y.pkl")

df.reset_index().to_csv(f"{base_path}\\final_processed_data.csv", index=False)
df["RETURN"].to_csv(f"{base_path}\\returns_for_garch.csv")

print("=" * 55)
print("    数据预处理完成")
print(f"   特征数量  : {n_feat}")
print(f"   总样本数  : {total}")
print(f"   训练 / 验证 / 测试 : {train_end} / {val_end - train_end} / {total - val_end}")
print(f"   窗口大小  : {WINDOW} 个交易日")
print("=" * 55)