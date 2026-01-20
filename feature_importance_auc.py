# feature_importance_auc.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import json

ART = Path("artifacts_parquet")
MODEL_PATH = "gru_dual_threshold_model.pth"
BATCH_SIZE = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUCls(nn.Module):
    def __init__(self, in_dim, hid=128, num_layers=1, dropout=0.3, n_classes=2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hid,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.bn = nn.BatchNorm1d(hid)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid, n_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = self.bn(out[:, -1, :])
        return self.fc(last)


def load_data():
    X_df = pd.read_parquet(ART / "test_X.parquet")
    y_df = pd.read_parquet(ART / "test_y.parquet")

    if "event_id" in X_df.columns:
        X_df = X_df.set_index("event_id")
    if "event_id" in y_df.columns:
        y_df = y_df.set_index("event_id")

    idx = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[idx]
    y_df = y_df.loc[idx]

    features = [c for c in X_df.columns if c != "event_id"]
    X = X_df[features].values.astype(np.float32)
    y = y_df["Label"].values.astype(np.int64)
    return X, y, features


def predict_proba_pos(model, X_arr):
    model.eval()
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)

    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)   # (N, 1, F)

    probs = []
    with torch.no_grad():
        for i in range(0, len(X_arr), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE].to(DEVICE)
            logits = model(batch)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def main():
    ART.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    model = GRUCls(
        in_dim=cfg["in_dim"], hid=cfg["hid"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        n_classes=cfg["n_classes"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X, y, feat_names = load_data()

    baseline_probs = predict_proba_pos(model, X)
    baseline_auc = roc_auc_score(y, baseline_probs)
    print(f"\nBaseline AUC = {baseline_auc:.6f}\n")

    results = []
    for i, fn in enumerate(tqdm(feat_names, desc="AUC test")):
        X_mask = X.copy()
        # 표준화 이후 0.0 ≈ 평균 → 해당 피처 영향 제거 시뮬레이션
        X_mask[:, i] = 0.0

        probs = predict_proba_pos(model, X_mask)
        auc = roc_auc_score(y, probs)
        drop = baseline_auc - auc

        results.append({"feature": fn, "auc": auc, "drop": drop})

    df = pd.DataFrame(results).sort_values("drop", ascending=False)

    # Top 10
    top10 = df.head(10)

    # ---------- 핵심 피처 core_features.json 저장 ----------
    CORE_FEATURES = top10["feature"].tolist()
    importance_map = {row.feature: float(row.drop) for row in df.itertuples()}

    out_json = {
        "source": "auc_drop_leave_one_out",
        "baseline_auc": float(baseline_auc),
        "core_top_n": 10,
        "core_features": CORE_FEATURES,
        "feature_importance_auc_drop": importance_map,
    }

    core_path = ART / "core_features.json"
    with open(core_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print("\n[+] Saved core_features.json!")
    print("    →", core_path)
    print("    Top core features:", CORE_FEATURES)

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x="drop", y="feature", palette="Reds_r")
    plt.title("Top 10 Features by ROC-AUC Drop")
    plt.xlabel("ROC-AUC Drop (Baseline - Masked)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plot_path = ART / "feature_importance_top10_auc.png"
    plt.savefig(plot_path)
    print(f"\n[+] Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
