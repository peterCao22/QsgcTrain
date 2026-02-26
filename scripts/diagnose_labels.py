"""标签诊断脚本，检查 is_strong_pos / fwd_return_30d 是否符合预期。"""
import pandas as pd
import sys

df = pd.read_parquet("data/training_dataset.parquet")
df["date"] = pd.to_datetime(df["date"])

print("=== 全局标签分布 ===")
print(df["is_strong_pos"].value_counts())
print(f"正样本率: {df['is_strong_pos'].mean():.2%}")
print()

# 看第一个截面
d0 = df[df["date"] == df["date"].min()].copy()
date0 = d0["date"].iloc[0].date()
n1 = (d0["is_strong_pos"] == 1).sum()
n0 = (d0["is_strong_pos"] == 0).sum()
print(f"截面 {date0}: is_1={n1}, is_0={n0}")
print("  is_1 样本:", d0[d0["is_strong_pos"] == 1]["instrument"].head(5).tolist())
print("  is_0 样本:", d0[d0["is_strong_pos"] == 0]["instrument"].head(5).tolist())
print()

# fwd_return_30d 对比
label_col = "fwd_return_30d"
if label_col in df.columns:
    print("=== fwd_return_30d: 正样本(1) vs 负样本(0) ===")
    print(df.groupby("is_strong_pos")[label_col].describe())
    print()
    print("正样本均值:", df[df["is_strong_pos"] == 1][label_col].mean())
    print("负样本均值:", df[df["is_strong_pos"] == 0][label_col].mean())
else:
    print("fwd_return_30d 列不存在，实际列名:", df.columns.tolist())
