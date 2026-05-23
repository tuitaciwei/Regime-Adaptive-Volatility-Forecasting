"""
4_ablation.py
=============
消融实验：系统验证主模型各组件的贡献
共 6 个变体，对比维度：
  A. 特征维度：17维(全) vs 13维(无高级特征)
  B. 融合机制：有门控 vs 无门控(Concat)
  C. 特征注意力：有 vs 无
  D. 骨干网络：TCN vs BiLSTM

运行：python 4_ablation.py
     python 4_ablation.py --variant no_attn   （只跑某个变体）
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

from utils import (set_seed, compute_all_metrics, print_metrics,
                   dm_test, dm_test_hac, random_walk_pred)

# ==========================
# 配置
# ==========================
set_seed(42)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = r"D:\CIKM"
N_FEAT    = 17

# ==========================
# 公共组件（与主模型完全相同）
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
        h, _= self.attn(self.proj(x), self.proj(x), self.proj(x))
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

class BiLSTM(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.lstm   = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden*2, 64)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# ==========================
# 六个消融变体
# ==========================

# ── 完整模型（对照组，与主模型完全一致）──
class M_Full(nn.Module):
    """Trans-TCN + FeatureAttn + Gate（完整）"""
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.fa   = FeatureAttention(dim)
        self.tr   = LightweightTransformer(dim)
        self.tcn  = TCNModel(dim)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)
    def forward(self, x):
        x = self.fa(x)
        g = self.gate(torch.cat([self.tr(x), self.tcn(x)], dim=-1))
        return self.out(g * self.tr(x) + (1-g) * self.tcn(x))

# ── A. 无特征注意力 ──
class M_NoAttn(nn.Module):
    """Trans-TCN + Gate，去掉 FeatureAttention"""
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.tr   = LightweightTransformer(dim)
        self.tcn  = TCNModel(dim)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)
    def forward(self, x):
        g = self.gate(torch.cat([self.tr(x), self.tcn(x)], dim=-1))
        return self.out(g * self.tr(x) + (1-g) * self.tcn(x))

# ── B. 无门控（Concat 融合）──
class M_NoGate(nn.Module):
    """Trans-TCN + FeatureAttn，门控换为 Concat"""
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.fa  = FeatureAttention(dim)
        self.tr  = LightweightTransformer(dim)
        self.tcn = TCNModel(dim)
        self.out = nn.Linear(128, 1)   # 64+64 直接拼接
    def forward(self, x):
        x = self.fa(x)
        return self.out(torch.cat([self.tr(x), self.tcn(x)], dim=-1))

# ── C. 仅用13维基础特征（无高级特征 VIX_ROC / RET_VOL_INTERACT / RV_5 / RV_22）──
class M_13Feats(nn.Module):
    """Trans-TCN + FeatureAttn + Gate，输入改为13维"""
    def __init__(self, dim=13):
        super().__init__()
        self.fa   = FeatureAttention(dim)
        self.tr   = LightweightTransformer(dim)
        self.tcn  = TCNModel(dim)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out  = nn.Linear(64, 1)
    def forward(self, x):
        x  = self.fa(x)
        ht = self.tr(x);  htc = self.tcn(x)
        g  = self.gate(torch.cat([ht, htc], dim=-1))
        return self.out(g * ht + (1-g) * htc)

# ── D. 骨干替换：BiLSTM 替代 TCN（有门控）──
class M_TransBiLSTM(nn.Module):
    """Trans-BiLSTM + FeatureAttn + Gate"""
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.fa     = FeatureAttention(dim)
        self.tr     = LightweightTransformer(dim)
        self.bilstm = BiLSTM(dim)
        self.gate   = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.out    = nn.Linear(64, 1)
    def forward(self, x):
        x  = self.fa(x)
        ht = self.tr(x);  hl = self.bilstm(x)
        g  = self.gate(torch.cat([ht, hl], dim=-1))
        return self.out(g * ht + (1-g) * hl)

# ── E. 骨干替换：BiLSTM 替代 TCN（无门控）──
class M_TransBiLSTM_NoGate(nn.Module):
    """Trans-BiLSTM + FeatureAttn + Concat"""
    def __init__(self, dim=N_FEAT):
        super().__init__()
        self.fa     = FeatureAttention(dim)
        self.tr     = LightweightTransformer(dim)
        self.bilstm = BiLSTM(dim)
        self.out    = nn.Linear(128, 1)
    def forward(self, x):
        x = self.fa(x)
        return self.out(torch.cat([self.tr(x), self.bilstm(x)], dim=-1))

# ==========================
# 训练 + 评估（统一接口）
# ==========================
def _make_loader(X, y, shuffle):
    return DataLoader(TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1)),
        batch_size=32, shuffle=shuffle)

def train_variant(model, tr_loader, vl_loader, save_name, epochs=130):
    model.to(device)
    crit  = nn.HuberLoss(delta=0.1)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    best  = float('inf')
    pat   = 0
    path  = os.path.join(base_path, save_name)

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item() * xb.size(0)
        vl /= len(vl_loader.dataset)
        sch.step(vl)

        if vl < best:
            best = vl; pat = 0
            torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= 15:
                print(f"  Early stop @ epoch {ep}"); break

        if ep % 20 == 0:
            print(f"  Epoch {ep:3d} | Val={vl:.6f}")

    model.load_state_dict(torch.load(path, map_location=device))
    return model

def eval_variant(model, te_loader, scaler_y):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.cpu().numpy())
    p = np.exp(scaler_y.inverse_transform(np.concatenate(preds))).flatten()
    t = np.exp(scaler_y.inverse_transform(np.concatenate(targets))).flatten()
    return t, p

# ==========================
# 单个变体的完整运行
# ==========================
VARIANTS = {
    "full":              (M_Full,              N_FEAT, "ablation_full.pth",
                          "完整模型（对照）"),
    "no_attn":           (M_NoAttn,            N_FEAT, "ablation_no_attn.pth",
                          "w/o Feature Attention"),
    "no_gate":           (M_NoGate,            N_FEAT, "ablation_no_gate.pth",
                          "w/o Gate (Concat)"),
    "13feats":           (M_13Feats,           13,     "ablation_13feats.pth",
                          "w/o Advanced Features (13-dim)"),
    "trans_bilstm":      (M_TransBiLSTM,       N_FEAT, "ablation_bilstm.pth",
                          "Trans+BiLSTM (w/ Gate)"),
    "trans_bilstm_nogate":(M_TransBiLSTM_NoGate,N_FEAT,"ablation_bilstm_nogate.pth",
                           "Trans+BiLSTM (Concat)"),
}

def run_variant(key):
    ModelClass, feat_dim, save_name, label = VARIANTS[key]

    scaler_y = joblib.load(os.path.join(base_path, "scaler_y.pkl"))

    # 加载数据（13维变体截取前13列）
    X_tr = np.load(os.path.join(base_path, "X_train.npy"))
    X_vl = np.load(os.path.join(base_path, "X_val.npy"))
    X_te = np.load(os.path.join(base_path, "X_test.npy"))
    y_tr = np.load(os.path.join(base_path, "y_train.npy"))
    y_vl = np.load(os.path.join(base_path, "y_val.npy"))
    y_te = np.load(os.path.join(base_path, "y_test.npy"))

    if feat_dim == 13:
        X_tr = X_tr[:, :, :13]
        X_vl = X_vl[:, :, :13]
        X_te = X_te[:, :, :13]

    tr = _make_loader(X_tr, y_tr, shuffle=True)
    vl = _make_loader(X_vl, y_vl, shuffle=False)
    te = _make_loader(X_te, y_te, shuffle=False)

    model  = ModelClass(dim=feat_dim)
    model  = train_variant(model, tr, vl, save_name)
    t_real, p_real = eval_variant(model, te, scaler_y)

    metrics = compute_all_metrics(t_real, p_real)
    rw      = random_walk_pred(t_real)
    print_metrics(f"消融：{label}", metrics,
                  dm_raw=dm_test(t_real, p_real, rw),
                  dm_hac=dm_test_hac(t_real, p_real, rw))
    return key, metrics

# ==========================
# 汇总打印（论文表格格式）
# ==========================
def print_summary_table(all_results):
    print("\n" + "=" * 90)
    print("📊 Ablation Study Summary Table")
    print("=" * 90)
    print(f"{'Variant':<35} | {'R²':>8} | {'QLike':>8} | {'MAE':>10} | {'RMSE':>10} | {'DA':>8}")
    print("-" * 90)
    for key, m in all_results:
        label = VARIANTS[key][3]
        marker = " ← best" if key == "full" else ""
        print(f"  {label:<33} | {m['R2']:>8.4f} | {m['QLike']:>8.4f} | "
              f"{m['MAE']:>10.6f} | {m['RMSE']:>10.6f} | {m['DA']:>8.2f}%{marker}")
    print("=" * 90)

# ==========================
# 入口
# ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='all',
        choices=list(VARIANTS.keys()) + ['all'])
    args = parser.parse_args()

    if args.variant == 'all':
        all_results = []
        for key in VARIANTS:
            print(f"\n{'='*65}")
            print(f"🔬  Ablation: {VARIANTS[key][3]}")
            print(f"{'='*65}")
            result = run_variant(key)
            all_results.append(result)
        print_summary_table(all_results)
    else:
        run_variant(args.variant)