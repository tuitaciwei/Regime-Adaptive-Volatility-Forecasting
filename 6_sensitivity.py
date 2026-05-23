"""
6_sensitivity.py
================
窗口敏感性实验：验证 window=60 是最优选择
- 对 [10, 20, 30, 60, 120] 五个窗口完整跑一遍
- 每次重新构建数据集（滑窗大小不同）、训练、评估
- 结果保存 npy，供主模型 plot_window_sensitivity() 调用
- 生成论文级三联图（R² / QLike / RMSE）

运行：python 6_sensitivity.py
     python 6_sensitivity.py --windows 10 20 30 60 120   （自定义窗口）
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from utils import set_seed, set_plot_style, compute_all_metrics

# ==========================
# 配置
# ==========================
SEED      = 42          # 唯一变量是窗口，其他全部固定
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"D:\CIKM"

FEATURES = [
    "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_LOG",
    "MA5", "MA10", "RSI",
    "RETURN", "RETURN_ABS",
    "VIX", "YIELD_10Y", "CORP_SPREAD",
    "VIX_ROC", "RET_VOL_INTERACT", "RV_5", "RV_22"
]
TARGET   = "VOLATILITY"
N_FEAT   = len(FEATURES)

# ==========================
# 模型（与主模型完全相同）
# ==========================
class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        w = self.attn(x.mean(dim=1))
        return x * w.unsqueeze(1)

class LightweightTransformer(nn.Module):
    def __init__(self, dim, d_model=128):
        super().__init__()
        self.proj = nn.Linear(dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.out  = nn.Linear(d_model, 64)
    def forward(self, x):
        h, _ = self.attn(self.proj(x), self.proj(x), self.proj(x))
        return self.out(h[:, -1, :])

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

class TCNModel(nn.Module):
    def __init__(self, dim, channels=(64,64,64), ks=3):
        super().__init__()
        layers = []
        for i, oc in enumerate(channels):
            ic = dim if i == 0 else channels[i-1]
            layers.append(TemporalBlock(ic, oc, ks, 2**i))
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(channels[-1], 64)
    def forward(self, x):
        return self.linear(self.tcn(x.transpose(1,2))[:,:,-1])

class GatedFusionModel(nn.Module):
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.fa   = FeatureAttention(dim)
        self.tr   = LightweightTransformer(dim)
        self.tcn  = TCNModel(dim)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)
    def forward(self, x):
        x  = self.fa(x)
        ht = self.tr(x); htc = self.tcn(x)
        g  = self.gate(torch.cat([ht, htc], dim=-1))
        return self.out(g * ht + (1-g) * htc), g

# ==========================
# 针对给定 window 重新构建数据集
# ==========================
def build_for_window(window):
    """
    从 final_processed_data.csv 重新滑窗
    只有 window 不同，其余（特征列/标准化/划分比例）完全一致
    """
    df   = pd.read_csv(os.path.join(base_path, "final_processed_data.csv"),
                       index_col=0).dropna()
    X_raw = df[FEATURES].values
    y_raw = df[TARGET].values.reshape(-1, 1)

    # 滑窗
    Xw, yl = [], []
    for i in range(len(X_raw) - window):
        Xw.append(X_raw[i:i+window])
        yl.append(y_raw[i+window])
    Xw = np.array(Xw); yl = np.array(yl)

    # 时序划分
    n       = len(Xw)
    tr_end  = int(n * 0.7)
    vl_end  = int(n * 0.8)

    X_tr, y_tr = Xw[:tr_end],        yl[:tr_end]
    X_vl, y_vl = Xw[tr_end:vl_end],  yl[tr_end:vl_end]
    X_te, y_te = Xw[vl_end:],        yl[vl_end:]

    # 标准化（只 fit train）
    sx = StandardScaler(); sy = StandardScaler()
    sx.fit(X_tr.reshape(-1, N_FEAT)); sy.fit(y_tr)

    def tx(d): b,t,f=d.shape; return sx.transform(d.reshape(-1,f)).reshape(b,t,f)

    X_tr_s = tx(X_tr); X_vl_s = tx(X_vl); X_te_s = tx(X_te)
    y_tr_s = sy.transform(y_tr).squeeze()
    y_vl_s = sy.transform(y_vl).squeeze()
    y_te_s = sy.transform(y_te).squeeze()

    def dl(X, y, sh):
        return DataLoader(TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1)),
            batch_size=32, shuffle=sh)

    return dl(X_tr_s,y_tr_s,True), dl(X_vl_s,y_vl_s,False), dl(X_te_s,y_te_s,False), sy

# ==========================
# 训练
# ==========================
def train(model, tr, vl, window, epochs=100):
    model.to(device)
    crit = nn.HuberLoss(delta=0.1)
    opt  = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
    best = float('inf'); pat = 0
    path = os.path.join(base_path, f"_tmp_w{window}.pth")

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            crit(pred, yb).backward()
            opt.step()

        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                vl_loss += crit(pred, yb).item() * xb.size(0)
        vl_loss /= len(vl.dataset)
        sch.step(vl_loss)

        if vl_loss < best:
            best = vl_loss; pat = 0
            torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= 15:
                print(f"    Early stop @ epoch {ep}"); break

        if ep % 20 == 0:
            print(f"    Epoch {ep:3d} | Val={vl_loss:.6f}")

    model.load_state_dict(torch.load(path, map_location=device))
    if os.path.exists(path): os.remove(path)
    return model

# ==========================
# 评估
# ==========================
def evaluate(model, te, sy):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for xb, yb in te:
            pred, _ = model(xb.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    p = np.exp(sy.inverse_transform(np.concatenate(preds))).flatten()
    t = np.exp(sy.inverse_transform(np.concatenate(targets))).flatten()
    return compute_all_metrics(t, p)

# ==========================
# 绘图（三联图，学术规范）
# ==========================
def plot_sensitivity(windows, results, save_path):
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    configs = [
        ("R2",    "R² Score",  "#B22222", 'o', True),
        ("QLike", "QLike",     "#1f77b4", 's', False),
        ("RMSE",  "RMSE",      "#2ca02c", '^', False),
    ]

    for ax, (key, ylabel, color, marker, higher_better) in zip(axes, configs):
        vals = results[key]
        ax.plot(windows, vals, marker=marker, color=color, lw=2.5, ms=8)
        ax.set_title(f'{ylabel} vs. Window Size', fontsize=12)
        ax.set_xlabel('Window Size (days)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        for w, v in zip(windows, vals):
            ax.annotate(f'{v:.4f}', (w, v),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=9)

        best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
        ax.axvline(windows[best_idx], color='gray', ls='--', lw=1.5, alpha=0.6,
                   label=f'Best w={windows[best_idx]}')
        ax.legend(fontsize=9)

    plt.suptitle('Window Size Sensitivity Analysis (Trans-TCN, seed=42)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"✅ 敏感性分析图已保存：{save_path}")

# ==========================
# 打印表格
# ==========================
def print_table(windows, results):
    print(f"\n{'='*75}")
    print("📊 Window Sensitivity Results")
    print(f"{'='*75}")
    print(f"  {'Window':<8} | {'R² ↑':>10} | {'QLike ↓':>10} | "
          f"{'RMSE ↓':>12} | {'MAE ↓':>12} | {'DA ↑':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for i, w in enumerate(windows):
        best_r2   = "✓" if results['R2'][i]   == max(results['R2'])    else " "
        best_ql   = "✓" if results['QLike'][i] == min(results['QLike']) else " "
        print(f"  {w:<8} | {results['R2'][i]:>10.4f}{best_r2}| "
              f"{results['QLike'][i]:>10.4f}{best_ql}| "
              f"{results['RMSE'][i]:>12.4e} | "
              f"{results['MAE'][i]:>12.4e} | "
              f"{results['DA'][i]:>8.2f}%")
    print(f"{'='*75}")
    print(f"  ✓ = best in column")
    best_w = windows[int(np.argmax(results['R2']))]
    print(f"\n  ✅ 最优窗口（R² 最高）: {best_w} 个交易日")
    print(f"     与主模型 window=60 {'一致 ✅' if best_w == 60 else '不一致，建议更新主模型配置'}")

# ==========================
# 主程序
# ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', type=int, nargs='+',
                        default=[10, 20, 30, 60, 120])
    args = parser.parse_args()

    windows = args.windows
    print(f"✅ 设备: {device}")
    print(f"✅ 测试窗口: {windows}")
    print(f"✅ 固定种子: {SEED}（唯一变量是窗口大小）\n")

    results = {k: [] for k in ['R2', 'QLike', 'RMSE', 'MAE', 'DA']}

    for w in windows:
        print(f"\n{'─'*55}")
        print(f"🔄  Window = {w}")
        print(f"{'─'*55}")
        set_seed(SEED)

        tr, vl, te, sy = build_for_window(w)
        print(f"  样本数: train={len(tr.dataset)} | val={len(vl.dataset)} | test={len(te.dataset)}")

        model   = GatedFusionModel(dim=N_FEAT)
        model   = train(model, tr, vl, w)
        metrics = evaluate(model, te, sy)

        for k in results:
            results[k].append(metrics[k])

        print(f"  ✅ R²={metrics['R2']:.4f} | QLike={metrics['QLike']:.4f} | "
              f"RMSE={metrics['RMSE']:.2e} | DA={metrics['DA']:.2f}%")

    # —— 保存原始结果（主模型 plot_window_sensitivity() 直接 load 此文件）——
    save_path = os.path.join(base_path, "window_sensitivity_results.npy")
    np.save(save_path, results)
    print(f"\n✅ 原始结果已保存：{save_path}")
    print(f"   主模型调用方式：")
    print(f"   res = np.load('window_sensitivity_results.npy', allow_pickle=True).item()")
    print(f"   plot_window_sensitivity({windows}, res['R2'], res['QLike'])")

    # —— 打印表格 ——
    print_table(windows, results)

    # —— 画图 ——
    fig_path = os.path.join(base_path, "Window_Sensitivity.pdf")
    plot_sensitivity(windows, results, fig_path)

    print("\n🎉 窗口敏感性实验完成！")