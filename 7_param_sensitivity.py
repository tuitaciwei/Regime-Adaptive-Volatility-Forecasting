"""
7_param_sensitivity.py
======================
超参数敏感性实验：验证 d_model 和 TCN channel 的选择合理性
实验设计：
  - d_model   ∈ {64, 128, 256}         （Transformer 嵌入维度）
  - tcn_ch    ∈ {32, 64, 128}          （TCN 每层 channel 数）
  - 其余超参全部固定（seed=42, lr=5e-5, window=60）

运行：
    python 7_param_sensitivity.py
    python 7_param_sensitivity.py --param dmodel   （只跑 d_model）
    python 7_param_sensitivity.py --param tcn       （只跑 TCN channel）
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

from utils import set_seed, set_plot_style, compute_all_metrics

# ==========================
# 配置
# ==========================
SEED      = 42
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"D:\CIKM"
N_FEAT    = 17

# 搜索空间
DMODEL_GRID  = [64, 128, 256]
TCN_CH_GRID  = [32, 64, 128]

# ==========================
# 可配置主模型
# ==========================
class FeatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.attn(x.mean(dim=1)).unsqueeze(1)

class LightweightTransformer(nn.Module):
    def __init__(self, dim, d_model):
        super().__init__()
        # nhead 必须能整除 d_model，自动选最大合法 nhead
        for nhead in [8, 4, 2, 1]:
            if d_model % nhead == 0:
                break
        self.proj = nn.Linear(dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
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
    def __init__(self, dim, ch):
        super().__init__()
        channels = [ch, ch, ch]
        layers   = []
        for i, oc in enumerate(channels):
            ic = dim if i == 0 else channels[i-1]
            layers.append(TemporalBlock(ic, oc, 3, 2**i))
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(ch, 64)
    def forward(self, x):
        return self.linear(self.tcn(x.transpose(1,2))[:,:,-1])

class ConfigurableGatedModel(nn.Module):
    """
    可配置版主模型：d_model 和 TCN channel 均可调
    其余结构与主模型完全相同
    """
    def __init__(self, dim=N_FEAT, d_model=128, tcn_ch=64):
        super().__init__()
        self.fa   = FeatureAttention(dim)
        self.tr   = LightweightTransformer(dim, d_model)
        self.tcn  = TCNModel(dim, tcn_ch)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)

        # 记录参数量（论文用）
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x  = self.fa(x)
        ht = self.tr(x); htc = self.tcn(x)
        g  = self.gate(torch.cat([ht, htc], dim=-1))
        return self.out(g * ht + (1-g) * htc), g

# ==========================
# 数据加载
# ==========================
def load_loaders():
    def _dl(Xp, yp, sh):
        X = torch.tensor(np.load(os.path.join(base_path, Xp)), dtype=torch.float32)
        y = torch.tensor(np.load(os.path.join(base_path, yp)), dtype=torch.float32).unsqueeze(-1)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=sh)
    return (_dl("X_train.npy","y_train.npy",True),
            _dl("X_val.npy",  "y_val.npy",  False),
            _dl("X_test.npy", "y_test.npy",  False))

# ==========================
# 训练
# ==========================
def train_config(model, tr, vl, tag, epochs=100):
    model.to(device)
    crit = nn.HuberLoss(delta=0.1)
    opt  = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
    best = float('inf'); pat = 0
    path = os.path.join(base_path, f"_tmp_{tag}.pth")

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
            if pat >= 15: break

    model.load_state_dict(torch.load(path, map_location=device))
    if os.path.exists(path): os.remove(path)
    return model

# ==========================
# 评估
# ==========================
def evaluate(model, te, scaler_y):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for xb, yb in te:
            pred, _ = model(xb.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    p = np.exp(scaler_y.inverse_transform(np.concatenate(preds))).flatten()
    t = np.exp(scaler_y.inverse_transform(np.concatenate(targets))).flatten()
    return compute_all_metrics(t, p)

# ==========================
# 绘图：双参数敏感性热力图 + 折线图
# ==========================
def plot_sensitivity_grid(results_dict, param_name, param_vals, save_path):
    """
    results_dict: { param_val: metrics_dict }
    """
    set_plot_style()
    metrics_to_show = ["R2", "QLike", "RMSE"]
    labels = {"R2": "R² ↑", "QLike": "QLike ↓", "RMSE": "RMSE ↓"}
    colors = {"R2": "#B22222", "QLike": "#1f77b4", "RMSE": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric in zip(axes, metrics_to_show):
        vals = [results_dict[v][metric] for v in param_vals]
        ax.plot(param_vals, vals, marker='o', color=colors[metric], lw=2.5, ms=8)

        for pv, mv in zip(param_vals, vals):
            ax.annotate(f'{mv:.4f}', (pv, mv),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=9)

        # 高亮最优值
        best_idx = int(np.argmax(vals) if metric == "R2" else np.argmin(vals))
        ax.axvline(param_vals[best_idx], color='gray', ls='--', lw=1.5, alpha=0.6,
                   label=f'Best={param_vals[best_idx]}')

        ax.set_title(f'{labels[metric]} vs. {param_name}', fontsize=12)
        ax.set_xlabel(param_name)
        ax.set_ylabel(labels[metric])
        ax.set_xticks(param_vals)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(f'Hyperparameter Sensitivity: {param_name}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"✅ 敏感性图已保存：{save_path}")

def plot_combined_heatmap(dmodel_results, tcn_results, save_path):
    """
    双参数结果对比热力图（论文附图）
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row_idx, (results, param_vals, param_name) in enumerate([
        (dmodel_results, DMODEL_GRID, "d_model"),
        (tcn_results,    TCN_CH_GRID, "TCN Channels"),
    ]):
        for col_idx, metric in enumerate(["R2", "QLike", "RMSE"]):
            ax   = axes[row_idx][col_idx]
            vals = [results[v][metric] for v in param_vals]
            colors_bar = ['#ff7f0e' if (metric == "R2" and v == max(vals))
                          or (metric != "R2" and v == min(vals))
                          else '#1f77b4' for v in vals]
            bars = ax.bar([str(p) for p in param_vals], vals,
                          color=colors_bar, edgecolor='white', alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                        f'{v:.4f}', ha='center', fontsize=9)
            ax.set_title(f'{param_name} — {metric}', fontsize=10)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"✅ 组合热力图已保存：{save_path}")

# ==========================
# 打印表格
# ==========================
def print_param_table(results, param_vals, param_name, default_val):
    print(f"\n{'='*70}")
    print(f"📊 {param_name} Sensitivity Results  (default={default_val} ★)")
    print(f"{'='*70}")
    print(f"  {param_name:<12} | {'R² ↑':>10} | {'QLike ↓':>10} | "
          f"{'RMSE ↓':>12} | {'#Params':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    for v in param_vals:
        m   = results[v]
        tag = " ★" if v == default_val else ""
        print(f"  {str(v):<12} | {m['R2']:>10.4f} | {m['QLike']:>10.4f} | "
              f"{m['RMSE']:>12.4e} | {m.get('n_params', 0):>10,}{tag}")
    print(f"{'='*70}")

# ==========================
# 主程序
# ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='all',
                        choices=['dmodel', 'tcn', 'all'])
    args = parser.parse_args()

    print(f"✅ 设备: {device}")
    print(f"✅ 固定种子: {SEED}（唯一变量是超参）\n")

    tr, vl, te   = load_loaders()
    scaler_y     = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    dmodel_results = {}
    tcn_results    = {}

    # ── d_model 实验 ──
    if args.param in ('dmodel', 'all'):
        print("=" * 55)
        print("🔄 实验1：d_model 敏感性")
        print("=" * 55)
        for dm in DMODEL_GRID:
            set_seed(SEED)
            print(f"\n  d_model = {dm}")
            model   = ConfigurableGatedModel(d_model=dm, tcn_ch=64)
            n_param = model.n_params
            model   = train_config(model, tr, vl, f"dmodel{dm}")
            metrics = evaluate(model, te, scaler_y)
            metrics['n_params'] = n_param
            dmodel_results[dm]  = metrics
            print(f"  R²={metrics['R2']:.4f} | QLike={metrics['QLike']:.4f} | "
                  f"RMSE={metrics['RMSE']:.2e} | Params={n_param:,}")

        print_param_table(dmodel_results, DMODEL_GRID, "d_model", default_val=128)
        plot_sensitivity_grid(dmodel_results, "d_model", DMODEL_GRID,
                              os.path.join(base_path, "Sensitivity_dmodel.pdf"))
        np.save(os.path.join(base_path, "sensitivity_dmodel.npy"), dmodel_results)

    # ── TCN channel 实验 ──
    if args.param in ('tcn', 'all'):
        print("\n" + "=" * 55)
        print("🔄 实验2：TCN Channel 敏感性")
        print("=" * 55)
        for ch in TCN_CH_GRID:
            set_seed(SEED)
            print(f"\n  TCN channels = {ch}")
            model   = ConfigurableGatedModel(d_model=128, tcn_ch=ch)
            n_param = model.n_params
            model   = train_config(model, tr, vl, f"tcnch{ch}")
            metrics = evaluate(model, te, scaler_y)
            metrics['n_params'] = n_param
            tcn_results[ch]     = metrics
            print(f"  R²={metrics['R2']:.4f} | QLike={metrics['QLike']:.4f} | "
                  f"RMSE={metrics['RMSE']:.2e} | Params={n_param:,}")

        print_param_table(tcn_results, TCN_CH_GRID, "TCN Channels", default_val=64)
        plot_sensitivity_grid(tcn_results, "TCN Channels", TCN_CH_GRID,
                              os.path.join(base_path, "Sensitivity_TCN.pdf"))
        np.save(os.path.join(base_path, "sensitivity_tcn.npy"), tcn_results)

    # ── 组合图（两个都跑完才生成）──
    if args.param == 'all':
        plot_combined_heatmap(dmodel_results, tcn_results,
                              os.path.join(base_path, "Sensitivity_Combined.pdf"))
        print("\n🎉 参数敏感性实验完成！")
        print("   d_model 结果 → Sensitivity_dmodel.pdf")
        print("   TCN 结果    → Sensitivity_TCN.pdf")
        print("   组合图      → Sensitivity_Combined.pdf")