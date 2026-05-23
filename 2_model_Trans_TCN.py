
"""
2_model_Trans_TCN.py
====================
主模型：Gated Fusion（Transformer + TCN）+ Feature Attention
完整流程：加载数据 → 训练 → 评估 → 可视化 → 保存预测
"""

import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, TensorDataset
from utils import (set_seed, set_plot_style,
                   compute_all_metrics, print_metrics,
                   dm_test, dm_test_hac, random_walk_pred,
                   plot_pred_vs_true)

# ==========================
# 配置
# ==========================
set_seed(42)
set_plot_style()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = (r"D:\CIKM")

FEATURE_NAMES = [
    "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_LOG",
    "MA5", "MA10", "RSI",
    "RETURN", "RETURN_ABS",
    "VIX", "YIELD_10Y", "CORP_SPREAD",
    "VIX_ROC", "RET_VOL_INTERACT", "RV_5", "RV_22"
]
N_FEATURES = len(FEATURE_NAMES)   # 17

# ==========================
# 数据加载
# ==========================
def load_data():
    def _t(path, unsqueeze=False):
        arr = torch.tensor(np.load(path), dtype=torch.float32)
        return arr.unsqueeze(-1) if unsqueeze else arr

    X_train = _t(f"{base_path}\\X_train.npy")
    X_val   = _t(f"{base_path}\\X_val.npy")
    X_test  = _t(f"{base_path}\\X_test.npy")
    y_train = _t(f"{base_path}\\y_train.npy", unsqueeze=True)
    y_val   = _t(f"{base_path}\\y_val.npy",   unsqueeze=True)
    y_test  = _t(f"{base_path}\\y_test.npy",  unsqueeze=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader

# ==========================
# 模型组件
# ==========================
class FeatureAttention(nn.Module):
    """
    特征级注意力：对每个特征计算时序均值后，学习一组 [0,1] 权重，
    动态压制低信息量特征，强化高相关特征（如 VIX、RV_5）
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, T, F]
        weights = self.attn(x.mean(dim=1))          # [B, F]
        return x * weights.unsqueeze(1), weights    # [B, T, F], [B, F]


class LightweightTransformer(nn.Module):
    """
    轻量级 Transformer：捕捉长程依赖和跨时间步的全局注意力
    d_model=128, 4头注意力，单层编码器（防止过拟合）
    """
    def __init__(self, feature_dim, d_model=128):
        super().__init__()
        self.proj    = nn.Linear(feature_dim, d_model)
        self.attn    = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.out     = nn.Linear(d_model, 64)
    def forward(self, x):
        h          = self.proj(x)
        attn_out, _= self.attn(h, h, h)
        return self.out(attn_out[:, -1, :])   # 取最后时间步


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 基本单元：因果膨胀卷积 + weight_norm + Dropout"""
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_in, n_out, kernel_size,
                      padding=padding, dilation=dilation))
        self.net = nn.Sequential(
            self.conv, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x))


class TCNModel(nn.Module):
    """
    三层 TCN：感受野 = 1 + (kernel-1)*(1+2+4) = 15 个时间步
    配合 window=60，捕捉中短期局部时序模式
    """
    def __init__(self, input_size, num_channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_c in enumerate(num_channels):
            in_c = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_c, out_c, kernel_size, 2 ** i, dropout))
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 64)
    def forward(self, x):
        # x: [B, T, F] → 转置为 [B, F, T]
        out = self.tcn(x.transpose(1, 2))
        return self.linear(out[:, :, -1])


class GatedFusionModel(nn.Module):
    """
    主模型：Feature Attention → Transformer + TCN → Gated Fusion → 输出
    门控机制自适应决定：高波动期偏向 TCN（局部），低波动期偏向 Transformer（全局）
    """
    def __init__(self, feature_dim=N_FEATURES):
        super().__init__()
        self.feature_attn = FeatureAttention(feature_dim)
        self.transformer  = LightweightTransformer(feature_dim)
        self.tcn          = TCNModel(feature_dim)
        self.gate         = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.output       = nn.Linear(64, 1)

    def forward(self, x):
        x, attn_w = self.feature_attn(x)
        h_t  = self.transformer(x)
        h_tc = self.tcn(x)
        g    = self.gate(torch.cat([h_t, h_tc], dim=-1))
        fused = g * h_t + (1 - g) * h_tc
        return self.output(fused), g, attn_w

# ==========================
# 训练
# ==========================
def train_model(model, train_loader, val_loader, epochs=100, lr=5e-5):
    model.to(device)
    criterion  = nn.HuberLoss(delta=0.1)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='min', patience=5, factor=0.5)

    best_val       = float('inf')
    patience_count = 0
    early_stop     = 15
    save_path      = os.path.join(base_path, "best_model_Transformer_TCN.pth")

    for epoch in range(1, epochs + 1):
        # —— 训练 ——
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # —— 验证 ——
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _ = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= early_stop:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Train={train_loss:.6f} | "
                  f"Val={val_loss:.6f} | LR={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"  ✅ Best val loss: {best_val:.6f}")
    return model

# ==========================
# 推理：收集预测值 + 门控值 + 注意力权重
# ==========================
def predict(model, loader):
    model.eval()
    preds, targets, gates, attns = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, g, attn = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
            gates.append(g.cpu().numpy())
            attns.append(attn.cpu().numpy())
    return (np.concatenate(preds),
            np.concatenate(targets),
            np.concatenate(gates).flatten(),
            np.concatenate(attns))

# ==========================
# 还原真实量纲
# ==========================
def inverse_transform(arr_scaled, scaler_y):
    """标准化逆变换 + exp 还原（目标是 log-volatility）"""
    return np.exp(scaler_y.inverse_transform(arr_scaled)).flatten()

# ==========================
# 绘图：DM 对比表（全模型）
# ==========================
def run_dm_table(t_real, p_real):
    baselines = {
        "GARCH":       "preds_garch.npy",
        "HAR-RV":      "preds_har.npy",
        "BiLSTM":      "preds_bilstm.npy",
        "TCN":         "preds_tcn.npy",
        "Transformer": "preds_transformer.npy",
    }
    print("\n" + "=" * 95)
    print("📊 DIEBOLD-MARIANO TEST (QLIKE-based) | Trans-TCN vs Baselines")
    print("   Raw DM + HAC-Newey-West Adjusted")
    print("=" * 95)
    print(f"{'Comparison':<22} | {'DM-Raw':>8} | {'p-Raw':>8} | "
          f"{'DM-HAC':>8} | {'p-HAC':>8} | {'Sig-Raw':>8} | {'Sig-HAC'}")
    print("-" * 95)

    def sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "n.s."

    for name, fn in baselines.items():
        fp = os.path.join(base_path, fn)
        if not os.path.exists(fp):
            print(f"  ⚠️  {fn} 不存在，跳过")
            continue
        bp = np.load(fp).flatten()
        n  = min(len(t_real), len(p_real), len(bp))
        t, o, b = t_real[:n], p_real[:n], bp[:n]

        stat_r, p_r = dm_test(t, o, b)
        stat_h, p_h = dm_test_hac(t, o, b)

        print(f"  Ours vs {name:<13} | {stat_r:>8.4f} | {p_r:>8.4f} | "
              f"{stat_h:>8.4f} | {p_h:>8.4f} | {sig(p_r):>8} | {sig(p_h)}")
    print("=" * 95)

# ==========================
# 绘图函数集
# ==========================
def plot_zoom(true, pred):
    """局部放大对比（稳定期 vs 高波动期）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    mid = len(true) // 3
    ax1.plot(true[mid:mid+200], c='#000080', lw=2, label='True')
    ax1.plot(pred[mid:mid+200], c='#B22222', lw=1.8, alpha=0.8, label='Pred')
    ax1.set_title('Stable Period (zoom)')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(true[:200], c='#000080', lw=2, label='True')
    ax2.plot(pred[:200], c='#B22222', lw=1.8, alpha=0.8, label='Pred')
    ax2.set_title('High-Volatility Period (zoom)')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{base_path}\\Zoom_Comparison.pdf", dpi=300)
    plt.show(); plt.close()

def plot_gate(gate, true):
    """门控权重 + 真实波动率双轴图"""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(gate, c='#B22222', lw=1.2, label='Gate weight (→Transformer)')
    ax1.axhline(0.5, c='k', ls='--', alpha=0.5, lw=1)
    ax1.set_ylabel('Gate weight', color='#B22222')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(true, c='#1f77b4', alpha=0.4, lw=1, label='True Volatility')
    ax2.set_ylabel('True Volatility', color='#1f77b4')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Gate Weight Evolution vs. True Volatility')
    plt.tight_layout()
    plt.savefig(f"{base_path}\\Gate_Volatility.pdf", dpi=300)
    plt.show(); plt.close()

def plot_attn_heatmap(attn_weights):
    """特征注意力热力图（前300个测试样本）"""
    plt.figure(figsize=(16, 6))
    sns.heatmap(attn_weights[:300].T,
                cmap='magma',
                yticklabels=FEATURE_NAMES,
                cbar_kws={'label': 'Attention Weight'})
    plt.title('Feature Attention Heatmap (Test Period, first 300 steps)')
    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig(f"{base_path}\\Attention_Heatmap.pdf", dpi=300)
    plt.show(); plt.close()

def plot_rmse_bar(true, ours_pred):
    """RMSE 对比柱状图"""
    from sklearn.metrics import mean_squared_error
    baselines = {
        "GARCH":       "preds_garch.npy",
        "HAR-RV":      "preds_har.npy",
        "BiLSTM":      "preds_bilstm.npy",
        "TCN":         "preds_tcn.npy",
        "Transformer": "preds_transformer.npy",
    }
    names, rmses = [], []
    for name, fn in baselines.items():
        fp = os.path.join(base_path, fn)
        if os.path.exists(fp):
            pred = np.load(fp).flatten()
            n = min(len(true), len(pred))
            names.append(name)
            rmses.append(np.sqrt(mean_squared_error(true[:n], pred[:n])))
    names.append("Trans-TCN\n(Ours)")
    rmses.append(np.sqrt(mean_squared_error(true, ours_pred)))

    colors = ['#1f77b4'] * (len(names) - 1) + ['#ff7f0e']
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, rmses, color=colors, edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                 f'{h:.2e}', ha='center', fontsize=9)
    plt.title('RMSE Comparison Across Models')
    plt.ylabel('RMSE')
    plt.xticks(rotation=15)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{base_path}\\RMSE_Barplot.pdf", dpi=300)
    plt.show(); plt.close()

def plot_window_sensitivity(windows, r2_list, qlike_list):
    """
    窗口敏感性分析图
    windows / r2_list / qlike_list 来自 sensitivity_experiment.py 的真实结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(windows, r2_list, marker='o', c='#B22222', lw=2)
    ax1.set_title('R² vs. Window Size')
    ax1.set_xlabel('Window Size (days)')
    ax1.set_ylabel('R²')
    ax1.grid(alpha=0.3)
    for w, v in zip(windows, r2_list):
        ax1.annotate(f'{v:.3f}', (w, v), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=9)

    ax2.plot(windows, qlike_list, marker='s', c='#1f77b4', lw=2)
    ax2.set_title('QLike vs. Window Size')
    ax2.set_xlabel('Window Size (days)')
    ax2.set_ylabel('QLike')
    ax2.grid(alpha=0.3)
    for w, v in zip(windows, qlike_list):
        ax2.annotate(f'{v:.3f}', (w, v), xytext=(0, 7),
                     textcoords='offset points', ha='center', fontsize=9)

    plt.suptitle('Window Size Sensitivity Analysis', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{base_path}\\Window_Sensitivity.pdf", dpi=300)
    plt.show(); plt.close()

# ==========================
# 补充到 2_model_Trans_TCN.py 末尾
# 在 if __name__ == '__main__': 的最后调用 plot_case_study(t_real, p_real)
# ==========================

def plot_case_study(true, pred, save_dir=base_path):
    """
    案例分析：对比主模型在两个极端行情期的预测表现
    需要 final_processed_data.csv 提供日期索引

    调用方式（加在主程序末尾）：
        plot_case_study(t_real, p_real)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from utils import set_plot_style
    set_plot_style()

    # ── 读取日期索引 ──
    df        = pd.read_csv(f"{save_dir}\\final_processed_data.csv", index_col=0,
                            parse_dates=True)
    df        = df.dropna()
    split_idx = np.load(f"{save_dir}\\split_idx.npy")
    val_end   = int(split_idx[1])
    WINDOW    = 60

    # # 测试集对应的日期（滑窗后索引从 WINDOW 开始）
    # test_dates = df.index[val_end + WINDOW:]
    # n          = min(len(test_dates), len(true))
    # test_dates = test_dates[:n]
    # true       = true[:n]
    # pred       = pred[:n]

    # 替换原来的 test_dates 计算部分
    df = pd.read_csv(f"{save_dir}\\final_processed_data.csv",
                     index_col=0, parse_dates=True).dropna()
    split_idx = np.load(f"{save_dir}\\split_idx.npy")
    val_end = int(split_idx[1])
    WINDOW = 60

    # 原始 df 中，测试集第一个样本对应的标签位置是 val_end + WINDOW
    # 取从那里开始、长度等于 true 的日期序列
    test_start = val_end + WINDOW
    test_dates = df.index[test_start: test_start + len(true)]

    # 保险：再做一次长度对齐
    n = min(len(test_dates), len(true), len(pred))
    test_dates = test_dates[:n]
    true = true[:n]
    pred = pred[:n]

    # 打印确认
    print(f"✅ 测试集日期: {test_dates[0].date()} → {test_dates[-1].date()}，共 {n} 个交易日")

    # ── 定义两个高波动事件区间 ──

    events = [
        {
            "label": "2024 US Election Volatility",
            "start": "2024-10-01",
            "end": "2024-11-30",
            "color": "#d62728",
            "caption": "2024 U.S. Election Period (Oct–Nov 2024)"
        },
        {
            "label": "2025 Market Turbulence",
            "start": "2025-03-01",
            "end": "2025-05-31",
            "color": "#1f77b4",
            "caption": "2025 Tariff Shock & Market Correction (Mar–May 2025)"
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, ev in zip(axes, events):
        mask = (test_dates >= ev["start"]) & (test_dates <= ev["end"])
        if mask.sum() < 5:
            ax.text(0.5, 0.5, f"No test data in\n{ev['label']}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(ev["caption"])
            continue

        dates_ev = test_dates[mask]
        true_ev  = true[mask]
        pred_ev  = pred[mask]

        # —— 主曲线 ——
        ax.plot(dates_ev, true_ev, label='True Volatility',
                color='#000080', lw=2.2, zorder=3)
        ax.plot(dates_ev, pred_ev, label='Trans-TCN Pred',
                color=ev["color"], lw=2, alpha=0.85, linestyle='--', zorder=3)

        # —— 误差填充 ——
        ax.fill_between(dates_ev, true_ev, pred_ev,
                        alpha=0.12, color=ev["color"], label='Error')

        # —— 局部指标 ——
        ss_res = np.sum((true_ev - pred_ev) ** 2)
        ss_tot = np.sum((true_ev - np.mean(true_ev)) ** 2)
        r2_ev  = 1 - ss_res / (ss_tot + 1e-12)
        rmse_ev= np.sqrt(np.mean((true_ev - pred_ev) ** 2))

        ax.set_title(f"{ev['caption']}\n"
                     f"R²={r2_ev:.3f}  RMSE={rmse_ev:.2e}",
                     fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.suptitle("Case Study: Model Performance During Extreme Market Events",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = f"{save_dir}\\Case_Study.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Case study 图已保存：{save_path}")

    # ── 打印两段区间的局部指标（论文用）──
    print("\n📊 Case Study 局部指标：")
    print(f"{'Event':<30} | {'N':>5} | {'R²':>8} | {'RMSE':>12} | {'MAE':>12}")
    print("-" * 75)
    for ev in events:
        mask = (test_dates >= ev["start"]) & (test_dates <= ev["end"])
        if mask.sum() < 5:
            continue
        t, p = true[mask], pred[mask]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2   = 1 - ss_res / (ss_tot + 1e-12)
        rmse = np.sqrt(np.mean((t - p) ** 2))
        mae  = np.mean(np.abs(t - p))
        print(f"  {ev['label']:<28} | {mask.sum():>5} | {r2:>8.4f} | "
              f"{rmse:>12.4e} | {mae:>12.4e}")

# ==========================
# 主程序
# ==========================
if __name__ == '__main__':
    print(f"✅ Device: {device}")
    print(f"✅ Features: {N_FEATURES}")

    # 1. 加载数据
    train_loader, val_loader, test_loader = load_data()
    scaler_y = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    # 2. 训练
    print("\n── 训练中 ──")
    model = GatedFusionModel(feature_dim=N_FEATURES)
    model = train_model(model, train_loader, val_loader)

    # 3. 推理
    preds_s, targets_s, gates, attn_w = predict(model, test_loader)
    p_real = inverse_transform(preds_s,   scaler_y)
    t_real = inverse_transform(targets_s, scaler_y)

    # 4. 指标
    metrics = compute_all_metrics(t_real, p_real)
    rw      = random_walk_pred(t_real)
    dm_r    = dm_test(t_real, p_real, rw)
    dm_h    = dm_test_hac(t_real, p_real, rw)
    print_metrics("Trans-TCN (Ours)", metrics, dm_raw=dm_r, dm_hac=dm_h)

    # 5. DM 对比表
    run_dm_table(t_real, p_real)

    # 6. 绘图
    plot_pred_vs_true(t_real, p_real, "Trans-TCN",
                      f"{base_path}\\Pred_vs_True.pdf")
    plot_zoom(t_real, p_real)
    plot_gate(gates, t_real)
    plot_attn_heatmap(attn_w)
    plot_rmse_bar(t_real, p_real)

    # 7. 窗口敏感性图（从真实实验结果中加载；先跑 sensitivity_experiment.py）
    sens_path = os.path.join(base_path, "window_sensitivity_results.npy")
    if os.path.exists(sens_path):
        res = np.load(sens_path, allow_pickle=True).item()
        print(res.keys())  # 看清楚键名再用

        plot_window_sensitivity([10, 20, 30, 60, 120], res['R2'], res['QLike'])
    else:
        print("⚠️  窗口敏感性结果未找到，请先运行 sensitivity_experiment.py")

    plot_case_study(t_real, p_real)

    # 8. 保存预测
    np.save(os.path.join(base_path, "preds_trans_tcn.npy"), p_real)
    pd.DataFrame(p_real, columns=['Pred']).to_csv(
        f"{base_path}\\submission_Final.csv", index=False)
    np.save(os.path.join(base_path, "test_alphas.npy"), attn_w)  # 已有 attn_w
    np.save(os.path.join(base_path, "test_gates.npy"), gates)  # 已有 gates

    print("\n🎉 全部完成！")
