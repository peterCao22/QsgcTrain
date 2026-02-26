"""快速诊断训练集数据质量。"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("data/training_dataset.parquet")
print("=== Dataset Diagnostics ===")
print(f"Total: {len(df)}  Dates: {df['date'].nunique()}")
print()

pos = df[df["is_strong_pos"] == 1]["fwd_return_20d"]
neg = df[df["is_strong_pos"] == 0]["fwd_return_20d"]
print(f"Positive fwd_return_20d:  mean={pos.mean()*100:.2f}%  median={pos.median()*100:.2f}%  >10%={(pos>0.10).mean():.1%}")
print(f"Negative fwd_return_20d:  mean={neg.mean()*100:.2f}%  median={neg.median()*100:.2f}%  >10%={(neg>0.10).mean():.1%}")
print()

ct = pd.crosstab(df["is_strong_pos"], df["is_top20pct"], normalize="index")
print("is_strong_pos x is_top20pct (row-normalized):")
print(ct)
print()

# Per-feature single-feature AUC
feat_cols = [c for c in df.columns if c not in ["date", "instrument", "is_strong_pos", "fwd_return_20d", "is_top20pct"]]
feat_aucs = {}
for col in feat_cols:
    vals = df[col].fillna(df[col].median())
    try:
        auc = roc_auc_score(df["is_top20pct"], vals)
        feat_aucs[col] = max(auc, 1 - auc)
    except Exception:
        pass

feat_auc_series = pd.Series(feat_aucs).sort_values(ascending=False)
print("Per-feature AUC vs is_top20pct (top 15):")
print(feat_auc_series.head(15).to_string())
