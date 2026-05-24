"""
utils.py
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# 随机种子
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 绘图全局配置
def set_plot_style():
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    sns.set_palette("colorblind")


# 指标函数
def r2_score_np(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)

def mae(true, pred):
    return np.mean(np.abs(true - pred))

def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

def mape(true, pred, eps=1e-9):
    return np.mean(np.abs((true - pred) / (true + eps))) * 100

def qlike(true, pred, eps=1e-10):

    true = np.clip(true, eps, None)
    pred = np.clip(pred, eps, None)
    return np.mean((true / pred) - np.log(true / pred) - 1)

def directional_accuracy(true, pred):
    true_dir = np.sign(true[1:] - true[:-1])
    pred_dir = np.sign(pred[1:] - true[:-1])
    return np.mean(true_dir == pred_dir) * 100

def compute_all_metrics(true, pred):
    return {
        "R2":   r2_score_np(true, pred),
        "QLike": qlike(true, pred),
        "MAE":  mae(true, pred),
        "RMSE": rmse(true, pred),
        "DA":   directional_accuracy(true, pred),
    }


# DM 检验
def _qlike_vec(true, pred, eps=1e-10):
    """逐样本 QLIKE 损失序列（用于 DM 差序列）"""
    true = np.clip(true, eps, None)
    pred = np.clip(pred, eps, None)
    return (true / pred) - np.log(true / pred) - 1

def dm_test(true, pred1, pred2):

    d  = _qlike_vec(true, pred1) - _qlike_vec(true, pred2)
    n  = len(d)
    se = np.sqrt(np.var(d, ddof=1) / n + 1e-12)
    stat = np.mean(d) / se
    p    = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p

def dm_test_hac(true, pred1, pred2, max_lag=None):

    d = _qlike_vec(true, pred1) - _qlike_vec(true, pred2)
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    d_dm   = d - np.mean(d)
    var_est = np.var(d_dm, ddof=1)
    for lag in range(1, max_lag + 1):
        gamma  = np.cov(d_dm[lag:], d_dm[:-lag])[0, 1]
        weight = 1 - lag / (max_lag + 1)
        var_est += 2 * weight * gamma

    se   = np.sqrt(max(var_est, 1e-12) / n)
    stat = np.mean(d) / se
    p    = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p

def random_walk_pred(true):
    return np.concatenate([[true[0]], true[:-1]])

def print_metrics(name, metrics, dm_raw=None, dm_hac=None):
    print("=" * 65)
    print(f"📊  {name}")
    print(f"    R²    : {metrics['R2']:.6f}")
    print(f"    QLike : {metrics['QLike']:.6f}")
    print(f"    MAE   : {metrics['MAE']:.6f}")
    print(f"    RMSE  : {metrics['RMSE']:.6f}")
    print(f"    DA    : {metrics['DA']:.4f}%")
    if dm_raw is not None:
        stat, p = dm_raw
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    DM (Raw)  : stat={stat:.4f}, p={p:.4f}  {sig}")
    if dm_hac is not None:
        stat, p = dm_hac
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    DM (HAC)  : stat={stat:.4f}, p={p:.4f}  {sig}")
    print("=" * 65)

def plot_pred_vs_true(true, pred, title, save_path):
    set_plot_style()
    plt.figure(figsize=(14, 5))
    plt.plot(true, label='True Volatility',  lw=2,   c='#000080')
    plt.plot(pred, label=f'{title} Pred',    lw=1.8, c='#B22222', alpha=0.75)
    plt.title(f'{title}: Predicted vs. Real Volatility')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()