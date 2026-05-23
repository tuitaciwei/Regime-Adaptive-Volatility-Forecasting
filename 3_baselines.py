"""
3_baselines.py
==============
所有 baseline 统一文件：GARCH / HAR-RV / BiLSTM / TCN / Transformer
- 统一使用 utils.py 的指标和 DM 检验（QLIKE-based）
- 统一的数据加载、训练、评估接口
- 运行：python 3_baselines.py --model [garch|har|bilstm|tcn|transformer|all]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from utils import (set_seed, compute_all_metrics, print_metrics,
                   dm_test, dm_test_hac, random_walk_pred, plot_pred_vs_true)

# ==========================
# 配置
# ==========================
set_seed(42)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"D:\CIKM"
N_FEAT    = 17

# ==========================
# 公共：数据加载
# ==========================
def load_tensors():
    def _t(p, unsq=False):
        arr = torch.tensor(np.load(p), dtype=torch.float32)
        return arr.unsqueeze(-1) if unsq else arr
    return (_t(f"{base_path}\\X_train.npy"),
            _t(f"{base_path}\\X_val.npy"),
            _t(f"{base_path}\\X_test.npy"),
            _t(f"{base_path}\\y_train.npy", unsq=True),
            _t(f"{base_path}\\y_val.npy",   unsq=True),
            _t(f"{base_path}\\y_test.npy",  unsq=True))

def make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te, bs=32):
    tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    vl = DataLoader(TensorDataset(X_vl, y_vl), batch_size=bs, shuffle=False)
    te = DataLoader(TensorDataset(X_te, y_te), batch_size=bs, shuffle=False)
    return tr, vl, te

def inverse(arr, scaler_y):
    return np.exp(scaler_y.inverse_transform(arr)).flatten()

# ==========================
# 公共：神经网络训练循环
# ==========================
def train_nn(model, train_loader, val_loader, save_name,
             epochs=150, lr=5e-5):
    model.to(device)
    criterion  = nn.HuberLoss(delta=0.1)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, 'min', patience=5, factor=0.5)
    best_val   = float('inf')
    patience   = 0
    save_path  = os.path.join(base_path, save_name)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl_loss += criterion(model(xb), yb).item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)
        vl_loss /= len(val_loader.dataset)
        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            patience = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= 15:
                print(f"  Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train={tr_loss:.6f} | "
                  f"Val={vl_loss:.6f} | LR={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# ==========================
# 公共：神经网络评估
# ==========================
def eval_nn(model, test_loader, scaler_y):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.cpu().numpy())
    p_real = inverse(np.concatenate(preds),   scaler_y)
    t_real = inverse(np.concatenate(targets), scaler_y)
    return t_real, p_real

# ==========================
# 公共：打印 + 保存
# ==========================
def report_and_save(name, t_real, p_real, save_file):
    metrics = compute_all_metrics(t_real, p_real)
    rw      = random_walk_pred(t_real)
    dm_r    = dm_test(t_real, p_real, rw)
    dm_h    = dm_test_hac(t_real, p_real, rw)
    print_metrics(name, metrics, dm_raw=dm_r, dm_hac=dm_h)
    np.save(os.path.join(base_path, save_file), p_real)
    print(f"  ✅ 预测已保存：{save_file}\n")
    return metrics

# ============================================================
# BASELINE 1：GARCH(1,1) — Expanding Window
# ============================================================
def run_garch():
    from arch import arch_model

    split_idx = np.load(os.path.join(base_path, "split_idx.npy"))
    val_end   = int(split_idx[1])
    scaler_y  = joblib.load(os.path.join(base_path, "scaler_y.pkl"))
    y_test_s  = np.load(os.path.join(base_path, "y_test.npy")).flatten()
    t_real    = inverse(y_test_s.reshape(-1, 1), scaler_y)

    df      = pd.read_csv(os.path.join(base_path, "final_processed_data.csv"), index_col=0)
    returns = df["RETURN"].values

    preds       = np.zeros(len(t_real))
    last_model  = None

    for i in range(len(t_real)):
        idx   = val_end + i
        r_win = returns[:idx]
        if len(r_win) < 50:
            preds[i] = np.var(r_win) if len(r_win) > 1 else 1e-6
            continue
        try:
            if i % 5 == 0 or last_model is None:
                am = arch_model(r_win * 100, mean='Zero', vol='GARCH',
                                p=1, q=1, dist='Normal')
                last_model = am.fit(disp='off', show_warning=False)
            var_s    = last_model.forecast(horizon=1).variance.iloc[-1, 0]
            preds[i] = max(var_s / 10000, 1e-12)
        except Exception:
            preds[i] = np.var(r_win)

    preds = pd.Series(preds).ffill().clip(lower=1e-12).values

    print(f"\n  GARCH 预测均值: {preds.mean():.8f} | 真实均值: {t_real.mean():.8f}")
    report_and_save("GARCH(1,1) — Expanding Window", t_real, preds, "preds_garch.npy")

# ============================================================
# BASELINE 2：HAR-RV — Expanding Window
# ============================================================
def run_har():
    split_idx = np.load(os.path.join(base_path, "split_idx.npy"))
    val_end   = int(split_idx[1])
    scaler_y  = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    y_tr = np.load(os.path.join(base_path, "y_train.npy"))
    y_vl = np.load(os.path.join(base_path, "y_val.npy"))
    y_te = np.load(os.path.join(base_path, "y_test.npy"))
    y_all = np.concatenate([y_tr, y_vl, y_te]).reshape(-1, 1)

    full_vol = inverse(y_all, scaler_y)   # 真实量纲全量波动率
    t_real   = full_vol[val_end:]

    from sklearn.linear_model import LinearRegression

    preds = np.zeros(len(t_real))
    for i in range(len(t_real)):
        hist = full_vol[:val_end + i]
        if len(hist) < 30:
            preds[i] = hist[-1] if len(hist) > 0 else 1e-10
            continue
        X_h, y_h = [], []
        for t in range(22, len(hist)):
            X_h.append([hist[t-1], hist[t-5:t].mean(), hist[t-22:t].mean()])
            y_h.append(hist[t])
        try:
            reg    = LinearRegression(fit_intercept=True).fit(X_h, y_h)
            preds[i] = reg.predict([[hist[-1],
                                     hist[-5:].mean(),
                                     hist[-22:].mean()]])[0]
        except Exception:
            preds[i] = hist[-1]

    preds = pd.Series(preds).ffill().fillna(1e-10).clip(lower=1e-10).values

    report_and_save("HAR-RV — Expanding Window", t_real, preds, "preds_har.npy")

# ============================================================
# BASELINE 3：Enhanced BiLSTM
# ============================================================
class EnhancedBiLSTM(nn.Module):
    def __init__(self, feature_dim=N_FEAT, hidden_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.LayerNorm(64))
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x      = self.proj(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_bilstm():
    X_tr, X_vl, X_te, y_tr, y_vl, y_te = load_tensors()
    tr, vl, te = make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te)
    scaler_y   = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    model  = EnhancedBiLSTM(feature_dim=N_FEAT)
    model  = train_nn(model, tr, vl, "best_model_BiLSTM.pth")
    t_real, p_real = eval_nn(model, te, scaler_y)
    report_and_save("Enhanced BiLSTM", t_real, p_real, "preds_bilstm.npy")

# ============================================================
# BASELINE 4：Pure TCN
# ============================================================
class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, ks, dil, drop=0.2):
        super().__init__()
        pad      = (ks - 1) * dil
        self.conv= nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_in, n_out, ks, padding=pad, dilation=dil))
        self.net = nn.Sequential(self.conv, Chomp1d(pad), nn.ReLU(), nn.Dropout(drop))
        self.relu= nn.ReLU()
    def forward(self, x): return self.relu(self.net(x))

class PureTCN(nn.Module):
    def __init__(self, feature_dim=N_FEAT, channels=(64, 64, 64), ks=3):
        super().__init__()
        self.proj    = nn.Linear(feature_dim, channels[0])
        self.network = nn.Sequential(*[
            TemporalBlock(channels[i-1] if i > 0 else channels[0],
                          channels[i], ks, 2**i)
            for i in range(len(channels))])
        self.fc      = nn.Linear(channels[-1], 1)
    def forward(self, x):
        x   = self.proj(x).transpose(1, 2)
        out = self.network(x)
        return self.fc(out.transpose(1, 2)[:, -1, :])

def run_tcn():
    X_tr, X_vl, X_te, y_tr, y_vl, y_te = load_tensors()
    tr, vl, te = make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te)
    scaler_y   = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    model  = PureTCN(feature_dim=N_FEAT)
    model  = train_nn(model, tr, vl, "best_model_TCN.pth")
    t_real, p_real = eval_nn(model, te, scaler_y)
    report_and_save("Pure TCN", t_real, p_real, "preds_tcn.npy")

# ============================================================
# BASELINE 5：Pure Transformer（FIXED：DM 改为 QLIKE-based）
# ============================================================
class PureTransformer(nn.Module):
    def __init__(self, feature_dim=N_FEAT, d_model=128, nhead=4):
        super().__init__()
        self.emb = nn.Linear(feature_dim, d_model)
        enc_layer= nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fc  = nn.Linear(d_model, 1)
    def forward(self, x):
        return self.fc(self.transformer(self.emb(x))[:, -1, :])

def run_transformer():
    X_tr, X_vl, X_te, y_tr, y_vl, y_te = load_tensors()
    tr, vl, te = make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te)
    scaler_y   = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    model  = PureTransformer(feature_dim=N_FEAT)
    model  = train_nn(model, tr, vl, "best_model_Transformer.pth")
    t_real, p_real = eval_nn(model, te, scaler_y)
    # ✅ DM 统一使用 QLIKE（原来是 MSE，已修正）
    report_and_save("Pure Transformer (FIXED)", t_real, p_real, "preds_transformer.npy")

# ==========================
# 补充到 3_baselines.py 末尾
# 在 if __name__ == '__main__': 的最后调用 run_efficiency_comparison()
# 依赖：所有 baseline 已经跑完并保存了 .pth 文件
# ==========================

def run_efficiency_comparison():
    """
    训练效率对比：参数量 + 训练时间 + 推理时间 + 测试集指标
    所有模型统一在相同硬件环境下计时，保证公平性
    结果输出论文表格格式
    """
    import time
    import torch
    import numpy as np
    import os
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils import set_seed, set_plot_style, compute_all_metrics

    set_seed(42)
    set_plot_style()
    scaler_y = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    # ── 加载数据（只需要一次）──
    X_tr = torch.tensor(np.load(os.path.join(base_path, "X_train.npy")), dtype=torch.float32)
    X_vl = torch.tensor(np.load(os.path.join(base_path, "X_val.npy")),   dtype=torch.float32)
    X_te = torch.tensor(np.load(os.path.join(base_path, "X_test.npy")),  dtype=torch.float32)
    y_tr = torch.tensor(np.load(os.path.join(base_path, "y_train.npy")), dtype=torch.float32).unsqueeze(-1)
    y_vl = torch.tensor(np.load(os.path.join(base_path, "y_val.npy")),   dtype=torch.float32).unsqueeze(-1)
    y_te = torch.tensor(np.load(os.path.join(base_path, "y_test.npy")),  dtype=torch.float32).unsqueeze(-1)

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=32, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=32, shuffle=False)

    # ── 定义所有要统计的神经网络模型 ──
    # GARCH 和 HAR-RV 单独处理（非神经网络）
    nn_models = {
        "BiLSTM":      EnhancedBiLSTM(feature_dim=N_FEAT),
        "TCN":         PureTCN(feature_dim=N_FEAT),
        "Transformer": PureTransformer(feature_dim=N_FEAT),
        "Trans-TCN\n(Ours)": None,   # 从主模型文件加载
    }

    # 导入主模型
    import sys
    sys.path.insert(0, base_path)

    # 直接在此处重定义主模型（避免循环导入）
    class FeatureAttention(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.Sigmoid())
        def forward(self, x):
            return x * self.attn(x.mean(dim=1)).unsqueeze(1)

    class LightTransformer(torch.nn.Module):
        def __init__(self, dim, d=128):
            super().__init__()
            self.proj = torch.nn.Linear(dim, d)
            self.attn = torch.nn.MultiheadAttention(d, num_heads=4, batch_first=True)
            self.out  = torch.nn.Linear(d, 64)
        def forward(self, x):
            h, _ = self.attn(self.proj(x), self.proj(x), self.proj(x))
            return self.out(h[:, -1, :])

    class _Chomp(torch.nn.Module):
        def __init__(self, s): super().__init__(); self.s = s
        def forward(self, x): return x[:, :, :-self.s].contiguous()

    class _TBlock(torch.nn.Module):
        def __init__(self, ni, no, ks, dil):
            super().__init__()
            pad = (ks-1)*dil
            self.net = torch.nn.Sequential(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(ni, no, ks, padding=pad, dilation=dil)),
                _Chomp(pad), torch.nn.ReLU(), torch.nn.Dropout(0.2))
            self.relu = torch.nn.ReLU()
        def forward(self, x): return self.relu(self.net(x))

    class _TCN(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            chs = [64,64,64]
            self.tcn = torch.nn.Sequential(*[
                _TBlock(dim if i==0 else chs[i-1], chs[i], 3, 2**i)
                for i in range(3)])
            self.linear = torch.nn.Linear(64, 64)
        def forward(self, x):
            return self.linear(self.tcn(x.transpose(1,2))[:,:,-1])

    class MainModel(torch.nn.Module):
        def __init__(self, dim=N_FEAT):
            super().__init__()
            self.fa   = FeatureAttention(dim)
            self.tr   = LightTransformer(dim)
            self.tcn  = _TCN(dim)
            self.gate = torch.nn.Sequential(torch.nn.Linear(128,1), torch.nn.Sigmoid())
            self.out  = torch.nn.Linear(64, 1)
        def forward(self, x):
            x = self.fa(x)
            ht = self.tr(x); htc = self.tcn(x)
            g  = self.gate(torch.cat([ht, htc], dim=-1))
            return self.out(g*ht + (1-g)*htc)

    nn_models["Trans-TCN\n(Ours)"] = MainModel()

    # ── 统计函数 ──
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def time_training(model, train_loader, val_loader, max_epochs=30):
        """计时：固定跑 max_epochs 轮，取平均每 epoch 时间 × 实际收敛 epoch 数估算"""
        model.to(device)
        crit = torch.nn.HuberLoss(delta=0.1)
        opt  = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for ep in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                # 兼容返回 tuple 或 tensor 的模型
                pred = out[0] if isinstance(out, tuple) else out
                crit(pred, yb).backward()
                opt.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed / max_epochs   # 每 epoch 平均秒数

    def time_inference(model, test_loader):
        """推理时间：整个测试集跑一遍"""
        model.eval()
        model.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000   # ms

    def get_test_metrics(model_name):
        """从已保存的预测文件读取指标（避免重复训练）"""
        fname_map = {
            "BiLSTM":           "preds_bilstm.npy",
            "TCN":              "preds_tcn.npy",
            "Transformer":      "preds_transformer.npy",
            "Trans-TCN\n(Ours)":"preds_trans_tcn.npy",
        }
        fname = fname_map.get(model_name)
        if fname is None:
            return None
        fpath = os.path.join(base_path, fname)
        if not os.path.exists(fpath):
            return None
        preds   = np.load(fpath).flatten()
        targets = np.exp(scaler_y.inverse_transform(
                    np.load(os.path.join(base_path, "y_test.npy")).reshape(-1,1)
                  )).flatten()
        n = min(len(preds), len(targets))
        return compute_all_metrics(targets[:n], preds[:n])

    # ── 主统计循环 ──
    records = []

    # 统计神经网络模型
    for name, model in nn_models.items():
        print(f"  ⏱  统计 {name.replace(chr(10),' ')} ...")
        set_seed(42)

        n_params      = count_params(model)
        sec_per_epoch = time_training(model, train_loader, val_loader, max_epochs=20)
        infer_ms      = time_inference(model, test_loader)
        metrics       = get_test_metrics(name)

        records.append({
            "Model":        name.replace("\n", " "),
            "#Params":      n_params,
            "Train(s/ep)":  sec_per_epoch,
            "Infer(ms)":    infer_ms,
            "R2":           metrics["R2"]    if metrics else float('nan'),
            "QLike":        metrics["QLike"] if metrics else float('nan'),
            "RMSE":         metrics["RMSE"]  if metrics else float('nan'),
        })
        print(f"     Params={n_params:,} | {sec_per_epoch:.2f}s/ep | "
              f"{infer_ms:.1f}ms | R²={records[-1]['R2']:.4f}")

    # 统计 GARCH 和 HAR-RV（无参数量，只计推理时间）
    stat_models = {
        "GARCH":   "preds_garch.npy",
        "HAR-RV":  "preds_har.npy",
    }
    for name, fname in stat_models.items():
        fpath   = os.path.join(base_path, fname)
        if not os.path.exists(fpath):
            continue
        preds   = np.load(fpath).flatten()
        targets = np.exp(scaler_y.inverse_transform(
                    np.load(os.path.join(base_path, "y_test.npy")).reshape(-1,1)
                  )).flatten()
        n       = min(len(preds), len(targets))
        metrics = compute_all_metrics(targets[:n], preds[:n])
        records.append({
            "Model":        name,
            "#Params":      0,        # 统计模型无神经网络参数
            "Train(s/ep)":  float('nan'),
            "Infer(ms)":    float('nan'),
            "R2":           metrics["R2"],
            "QLike":        metrics["QLike"],
            "RMSE":         metrics["RMSE"],
        })

    # ── 打印论文格式表格 ──
    print("\n" + "=" * 95)
    print("📊 Efficiency & Performance Comparison Table")
    print("=" * 95)
    print(f"  {'Model':<22} | {'#Params':>10} | {'Train(s/ep)':>12} | "
          f"{'Infer(ms)':>10} | {'R²↑':>8} | {'QLike↓':>8} | {'RMSE↓':>12}")
    print("  " + "-" * 91)

    for r in records:
        params_str = f"{r['#Params']:,}" if r['#Params'] > 0 else "N/A"
        train_str  = f"{r['Train(s/ep)']:.2f}" if not np.isnan(r['Train(s/ep)']) else "N/A"
        infer_str  = f"{r['Infer(ms)']:.1f}"   if not np.isnan(r['Infer(ms)'])   else "N/A"
        mark       = " ★" if "Ours" in r['Model'] or "Trans-TCN" in r['Model'] else ""
        print(f"  {r['Model']:<22} | {params_str:>10} | {train_str:>12} | "
              f"{infer_str:>10} | {r['R2']:>8.4f} | {r['QLike']:>8.4f} | "
              f"{r['RMSE']:>12.4e}{mark}")
    print("=" * 95)
    print("  ★ = proposed model")
    print("  N/A = statistical model, no gradient-based training")

    # ── 保存 CSV ──
    df = pd.DataFrame(records)
    csv_path = os.path.join(base_path, "efficiency_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ 效率对比已保存：{csv_path}")

    # ── 绘图：参数量 vs R² 气泡图 ──
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    colors_map = {
        "GARCH":          "#7f7f7f",
        "HAR-RV":         "#8c564b",
        "BiLSTM":         "#1f77b4",
        "TCN":            "#2ca02c",
        "Transformer":    "#9467bd",
        "Trans-TCN Ours": "#ff7f0e",
    }

    for r in records:
        if r['#Params'] == 0 or np.isnan(r['Train(s/ep)']):
            continue
        name   = r['Model'].replace("\n"," ")
        color  = colors_map.get(name, "#333333")
        size   = r['Train(s/ep)'] * 800   # 气泡大小 = 训练时间
        ax.scatter(r['#Params'], r['R2'], s=size, color=color,
                   alpha=0.75, edgecolors='white', linewidth=1.5,
                   label=name, zorder=3)
        ax.annotate(name,
                    (r['#Params'], r['R2']),
                    textcoords='offset points',
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Number of Parameters", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("Efficiency vs. Performance\n(bubble size = training time per epoch)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    bubble_path = os.path.join(base_path, "Efficiency_Bubble.pdf")
    plt.savefig(bubble_path, dpi=300)
    plt.show(); plt.close()
    print(f"  ✅ 气泡图已保存：{bubble_path}")

    # ── 绘图：训练时间柱状图（神经网络模型）──
    nn_records = [r for r in records if not np.isnan(r['Train(s/ep)'])]
    names  = [r['Model'].replace("\n"," ") for r in nn_records]
    times  = [r['Train(s/ep)'] for r in nn_records]
    colors_bar = ['#ff7f0e' if 'Trans-TCN' in n or 'Ours' in n
                  else '#1f77b4' for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, times, color=colors_bar, edgecolor='white', alpha=0.85)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{t:.2f}s', ha='center', fontsize=10)
    ax.set_ylabel("Training Time (s / epoch)")
    ax.set_title("Training Time Comparison Across Neural Models")
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    bar_path = os.path.join(base_path, "Efficiency_TrainTime.pdf")
    plt.savefig(bar_path, dpi=300)
    plt.show(); plt.close()
    print(f"  ✅ 训练时间柱状图已保存：{bar_path}")


# ==========================
# 在 3_baselines.py 的 if __name__ == '__main__': 最后加入：
#
#   print("\n" + "="*65)
#   print("📊 Running: EFFICIENCY COMPARISON")
#   print("="*65)
#   run_efficiency_comparison()
#
# ==========================



# ==========================
# 入口
# ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
        choices=['garch', 'har', 'bilstm', 'tcn', 'transformer', 'all'])
    args = parser.parse_args()

    runners = {
        'garch':       run_garch,
        'har':         run_har,
        'bilstm':      run_bilstm,
        'tcn':         run_tcn,
        'transformer': run_transformer,
    }

    if args.model == 'all':
        for name, fn in runners.items():
            print(f"\n{'='*65}")
            print(f"🚀  Running: {name.upper()}")
            print(f"{'='*65}")
            fn()
    else:
        runners[args.model]()

    print("\n" + "="*65)
    print("📊 Running: EFFICIENCY COMPARISON")
    print("="*65)
    run_efficiency_comparison()
