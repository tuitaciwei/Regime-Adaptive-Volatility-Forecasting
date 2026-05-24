"""
5_multi_seed.py
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
import warnings
warnings.filterwarnings("ignore")

from utils import (set_seed, set_plot_style,
                   compute_all_metrics, dm_test, dm_test_hac,
                   random_walk_pred)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"D:\CIKM"
N_FEAT    = 17

DEFAULT_SEEDS = [42, 1029, 52, 305, 1002]

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

class M_NoAttn(nn.Module):
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.tr   = LightweightTransformer(dim)
        self.tcn  = TCNModel(dim)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)
    def forward(self, x):
        ht = self.tr(x); htc = self.tcn(x)
        g  = self.gate(torch.cat([ht, htc], dim=-1))
        return self.out(g * ht + (1-g) * htc), g

def load_data():
    def _np(p): return np.load(os.path.join(base_path, p))
    return (_np("X_train.npy"), _np("X_val.npy"),   _np("X_test.npy"),
            _np("y_train.npy"), _np("y_val.npy"),   _np("y_test.npy"))

def make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te):
    def _dl(X, y, shuffle):
        return DataLoader(TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1)),
            batch_size=32, shuffle=shuffle)
    return _dl(X_tr,y_tr,True), _dl(X_vl,y_vl,False), _dl(X_te,y_te,False)

def train_one_seed(ModelClass, train_loader, val_loader, seed,
                   save_name, epochs=100, lr=5e-5):
    set_seed(seed)
    model = ModelClass().to(device)
    crit  = nn.HuberLoss(delta=0.1)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
    best  = float('inf'); pat = 0
    path  = os.path.join(base_path, f"_tmp_{save_name}_s{seed}.pth")

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            crit(pred, yb).backward()
            opt.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                vl += crit(pred, yb).item() * xb.size(0)
        vl /= len(val_loader.dataset)
        sch.step(vl)

        if vl < best:
            best = vl; pat = 0
            torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= 15:
                break

    model.load_state_dict(torch.load(path, map_location=device))
    if os.path.exists(path): os.remove(path)
    return model

def eval_one(model, test_loader, scaler_y):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred, _ = model(xb.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    p = np.exp(scaler_y.inverse_transform(np.concatenate(preds))).flatten()
    t = np.exp(scaler_y.inverse_transform(np.concatenate(targets))).flatten()
    return t, p

def run_multi_seed(ModelClass, seeds, label, data):
    X_tr, X_vl, X_te, y_tr, y_vl, y_te = data
    scaler_y = joblib.load(os.path.join(base_path, "scaler_y.pkl"))
    tr, vl, te = make_loaders(X_tr, X_vl, X_te, y_tr, y_vl, y_te)

    all_metrics = []
    seed_records = []

    print(f"\n{'='*65}")
    print(f" Multi-Seed Experiment: {label}")
    print(f"    Seeds: {seeds}")
    print(f"{'='*65}")

    for seed in seeds:
        print(f"\n  ── Seed {seed} ──")
        save_name = label.replace(" ", "_").lower()
        model  = train_one_seed(ModelClass, tr, vl, seed, save_name)
        t_real, p_real = eval_one(model, te, scaler_y)
        metrics = compute_all_metrics(t_real, p_real)
        all_metrics.append(metrics)

        rw = random_walk_pred(t_real)
        dm_stat, dm_p = dm_test_hac(t_real, p_real, rw)
        sig = "***" if dm_p < 0.001 else "**" if dm_p < 0.01 else "*" if dm_p < 0.05 else "n.s."

        print(f"    R²={metrics['R2']:.4f} | QLike={metrics['QLike']:.4f} | "
              f"RMSE={metrics['RMSE']:.2e} | DA={metrics['DA']:.2f}% | "
              f"DM(HAC) p={dm_p:.4f} {sig}")
        seed_records.append({
            "Seed": seed,
            "R2":   metrics['R2'],
            "QLike":metrics['QLike'],
            "MAE":  metrics['MAE'],
            "RMSE": metrics['RMSE'],
            "DA":   metrics['DA'],
            "DM_stat": dm_stat,
            "DM_p":    dm_p,
        })

    return all_metrics, seed_records

# ==========================
# 统计汇总
# ==========================
def summarize(label, all_metrics, seed_records):
    keys = ["R2", "QLike", "MAE", "RMSE", "DA"]
    stats = {k: {"mean": np.mean([m[k] for m in all_metrics]),
                 "std":  np.std([m[k] for m in all_metrics], ddof=1),
                 "min":  np.min([m[k] for m in all_metrics]),
                 "max":  np.max([m[k] for m in all_metrics])}
             for k in keys}

    print(f"\n{'='*65}")
    print(f"  Summary: {label}")
    print(f"{'='*65}")
    print(f"  {'Metric':<8} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for k in keys:
        s = stats[k]
        print(f"  {k:<8} | {s['mean']:>10.4f} | {s['std']:>10.4f} | "
              f"{s['min']:>10.4f} | {s['max']:>10.4f}")
    print(f"{'='*65}")


    print(f"\n   （均值 ± 标准差）：")
    for k in keys:
        s = stats[k]
        print(f"     {k}: {s['mean']:.4f} ± {s['std']:.4f}")

    return stats

def plot_seed_results(results_dict, save_path):

    set_plot_style()
    metrics_to_plot = ["R2", "QLike", "RMSE", "DA"]
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        data_to_plot = []
        labels       = []
        for label, metric_list in results_dict.items():
            vals = [m[metric] for m in metric_list]
            data_to_plot.append(vals)
            labels.append(label)

        bp = ax.boxplot(data_to_plot, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis='y')


        for i, vals in enumerate(data_to_plot):
            ax.scatter(i + 1, np.mean(vals), marker='D',
                       color='black', zorder=5, s=30, label='Mean' if i == 0 else '')
        if metric == metrics_to_plot[0]:
            ax.legend(fontsize=8)

    plt.suptitle('Multi-Seed Stability Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"   箱线图已保存：{save_path}")

def plot_seed_bar(seed_records_dict, metric="R2", save_path=None):

    set_plot_style()
    n_models = len(seed_records_dict)
    fig, ax  = plt.subplots(figsize=(max(10, 3 * n_models), 5))

    all_labels  = list(seed_records_dict.keys())
    all_records = list(seed_records_dict.values())
    n_seeds     = len(all_records[0])
    seeds_used  = [r["Seed"] for r in all_records[0]]
    x           = np.arange(n_seeds)
    width       = 0.8 / n_models

    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    for i, (label, records) in enumerate(zip(all_labels, all_records)):
        vals = [r[metric] for r in records]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=colors[i % len(colors)], alpha=0.8, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=7)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"Seed={s}" for s in seeds_used])
    ax.set_ylabel(metric)
    ax.set_title(f'Per-Seed {metric} Comparison')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(base_path, f"MultiSeed_{metric}.pdf")
    plt.savefig(save_path, dpi=300)
    plt.show(); plt.close()
    print(f"   逐种子柱状图已保存：{save_path}")

def save_seed_table(seed_records_dict, filepath):
    rows = []
    for label, records in seed_records_dict.items():
        for r in records:
            row = {"Model": label}
            row.update(r)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  详细结果已保存：{filepath}")

def print_paper_table(all_stats):

    print(f"\n{'='*80}")
    print("（均值 ± 标准差）")
    print(f"{'='*80}")
    print(f"{'Model':<30} | {'R² ↑':>16} | {'QLike ↓':>16} | {'RMSE ↓':>16} | {'DA ↑':>10}")
    print("-" * 80)
    for label, stats in all_stats.items():
        r2   = f"{stats['R2']['mean']:.4f} ± {stats['R2']['std']:.4f}"
        ql   = f"{stats['QLike']['mean']:.4f} ± {stats['QLike']['std']:.4f}"
        rm   = f"{stats['RMSE']['mean']:.2e} ± {stats['RMSE']['std']:.2e}"
        da   = f"{stats['DA']['mean']:.2f} ± {stats['DA']['std']:.2f}"
        print(f"  {label:<28} | {r2:>16} | {ql:>16} | {rm:>16} | {da:>10}")
    print(f"{'='*80}")
    print("\n  LaTeX 示例：")
    for label, stats in all_stats.items():
        r2 = stats['R2']
        print(f"  {label}: $R^2 = {r2['mean']:.4f} \\pm {r2['std']:.4f}$")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',    type=int, default=5,
                        help='种子数量（从 DEFAULT_SEEDS 取前N个）')
    parser.add_argument('--ablation', action='store_true',
                        help='同时对关键消融变体（w/o Attn）跑多种子')
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[:args.seeds]
    data  = load_data()

    print(f"设备: {device}")
    print(f"种子: {seeds}")

    main_metrics, main_records = run_multi_seed(
        GatedFusionModel, seeds, "Trans-TCN (Ours)", data)
    main_stats = summarize("Trans-TCN (Ours)", main_metrics, main_records)

    all_metrics_dict = {"Trans-TCN": main_metrics}
    all_records_dict = {"Trans-TCN": main_records}
    all_stats_dict   = {"Trans-TCN": main_stats}

    if args.ablation:
        noattn_metrics, noattn_records = run_multi_seed(
            M_NoAttn, seeds, "w/o Attn", data)
        noattn_stats = summarize("w/o Attn", noattn_metrics, noattn_records)

        all_metrics_dict["w/o Attn"] = noattn_metrics
        all_records_dict["w/o Attn"] = noattn_records
        all_stats_dict["w/o Attn"]   = noattn_stats

    print_paper_table(all_stats_dict)

    box_path = os.path.join(base_path, "MultiSeed_Boxplot.pdf")
    plot_seed_results(all_metrics_dict, box_path)
    plot_seed_bar(all_records_dict, metric="R2")
    plot_seed_bar(all_records_dict, metric="QLike")

    csv_path = os.path.join(base_path, "multi_seed_results.csv")
    save_seed_table(all_records_dict, csv_path)

    np.save(os.path.join(base_path, "multi_seed_stats.npy"), all_stats_dict)

    print("\n  多种子实验完成！")
    print(f"   箱线图   → MultiSeed_Boxplot.pdf")
    print(f"   详细结果 → multi_seed_results.csv")
