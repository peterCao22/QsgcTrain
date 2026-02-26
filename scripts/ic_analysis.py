"""
单特征 IC 分析报告

读取训练数据集（data/training_dataset.parquet），对每个特征计算：
  - Spearman IC（与30日收益的截面相关性）
  - IC 均值、IC 标准差、ICIR（IC/std）
  - 正 IC 占比
  - 特征缺失率

并输出：
  results/ic_report.csv     完整报告（可在 Excel 打开）
  results/ic_report.png     IC 条形图（按均值排序）
  results/ic_timeseries.png IC 时序图（每月变化趋势）

运行方式：
    conda activate rqsdk
    python scripts/ic_analysis.py

    # 使用已有数据集
    python scripts/ic_analysis.py --dataset data/training_dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config import LABEL_COL, RESULTS_DIR
from features.precursor import ALL_FEATURE_COLS

DEFAULT_DATASET = ROOT_DIR / "data" / "training_dataset.parquet"
IC_REPORT_CSV   = RESULTS_DIR / "ic_report.csv"
IC_PLOT_BAR     = RESULTS_DIR / "ic_report.png"
IC_PLOT_TS      = RESULTS_DIR / "ic_timeseries.png"


# ─── 核心计算 ─────────────────────────────────────────────────────────────────

def compute_cross_sectional_ic(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str = LABEL_COL,
) -> pd.Series:
    """
    计算单个特征的截面 Spearman IC 时序（每个截面日期一个值）。

    Returns:
        Series，index=date，values=Spearman IC（-1~1）
    """
    ic_by_date = {}
    for date, grp in df.groupby("date"):
        sub = grp[[feature_col, label_col]].dropna()
        if len(sub) < 10:
            continue
        ic, _ = stats.spearmanr(sub[feature_col], sub[label_col])
        ic_by_date[date] = ic

    return pd.Series(ic_by_date, name=feature_col)


def compute_all_ic(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    对所有特征计算 IC 统计指标。

    Returns:
        DataFrame，index=feature_name，columns=
          [ic_mean, ic_std, icir, ic_positive_rate, missing_rate,
           valid_dates, signal_strength]
    """
    feature_cols = [c for c in ALL_FEATURE_COLS if c in dataset.columns]
    records = []

    for feat in feature_cols:
        ic_series = compute_cross_sectional_ic(dataset, feat)

        if ic_series.empty:
            records.append({
                "feature":         feat,
                "ic_mean":         np.nan,
                "ic_std":          np.nan,
                "icir":            np.nan,
                "ic_positive_rate": np.nan,
                "missing_rate":    float(dataset[feat].isna().mean()),
                "valid_dates":     0,
                "signal_strength": "无效",
            })
            continue

        ic_mean = float(ic_series.mean())
        ic_std  = float(ic_series.std())
        icir    = ic_mean / ic_std if ic_std > 0 else np.nan
        pos_rate = float((ic_series > 0).mean())
        n_dates  = len(ic_series)
        missing  = float(dataset[feat].isna().mean())

        # 信号强度分级
        abs_ic = abs(ic_mean)
        if abs_ic >= 0.05 and abs(icir) >= 0.5:
            strength = "强有效"
        elif abs_ic >= 0.02:
            strength = "弱有效"
        else:
            strength = "无效"

        records.append({
            "feature":          feat,
            "ic_mean":          ic_mean,
            "ic_std":           ic_std,
            "icir":             icir,
            "ic_positive_rate": pos_rate,
            "missing_rate":     missing,
            "valid_dates":      n_dates,
            "signal_strength":  strength,
        })

    report = pd.DataFrame(records).set_index("feature")
    report = report.sort_values("ic_mean", ascending=False, key=lambda x: x.abs())
    return report


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_ic_bar(report: pd.DataFrame, save_path: Path) -> None:
    """绘制特征 IC 均值条形图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))

        colors = [
            "steelblue" if v >= 0.05 else
            "skyblue"   if v >= 0.02 else
            "lightgray"
            for v in report["ic_mean"].abs()
        ]
        bars = ax.barh(report.index, report["ic_mean"], color=colors)

        ax.axvline(0,    color="black", linewidth=0.8)
        ax.axvline(0.05, color="green",  linewidth=1.0, linestyle="--", label="IC=0.05")
        ax.axvline(-0.05,color="green",  linewidth=1.0, linestyle="--")
        ax.axvline(0.02, color="orange", linewidth=0.8, linestyle=":",  label="IC=0.02")
        ax.axvline(-0.02,color="orange", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Spearman IC 均值（截面平均）")
        ax.set_title("蓄力期特征 IC 分析报告", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"IC bar chart saved: {save_path}")
    except Exception as e:
        logger.warning(f"Cannot save IC bar chart: {e}")


def plot_ic_timeseries(
    dataset: pd.DataFrame,
    report: pd.DataFrame,
    save_path: Path,
    top_n: int = 6,
) -> None:
    """绘制 top_n 有效特征的 IC 时序图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 取 |IC均值| 最大的 top_n 个特征
        top_feats = report.dropna(subset=["ic_mean"]).head(top_n).index.tolist()
        if not top_feats:
            return

        fig, axes = plt.subplots(
            (top_n + 1) // 2, 2,
            figsize=(14, 3 * ((top_n + 1) // 2)),
            sharex=True,
        )
        axes = axes.flatten()

        for i, feat in enumerate(top_feats):
            ic_series = compute_cross_sectional_ic(dataset, feat)
            ic_roll   = ic_series.rolling(3, min_periods=1).mean()

            ax = axes[i]
            ax.plot(ic_series.index, ic_series.values, alpha=0.4, color="gray", linewidth=0.8)
            ax.plot(ic_roll.index, ic_roll.values, color="steelblue", linewidth=1.5)
            ax.axhline(0,    color="black",  linewidth=0.6)
            ax.axhline(0.05, color="green",  linewidth=0.8, linestyle="--")
            ax.axhline(-0.05,color="green",  linewidth=0.8, linestyle="--")
            ax.set_title(
                f"{feat}  (IC={report.loc[feat,'ic_mean']:.3f}, "
                f"ICIR={report.loc[feat,'icir']:.2f})"
            )
            ax.tick_params(axis="x", rotation=30)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("TOP 特征 IC 时序（灰=原始，蓝=3期平滑）", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"IC timeseries saved: {save_path}")
    except Exception as e:
        logger.warning(f"Cannot save IC timeseries: {e}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_ic_analysis(dataset_path: Path) -> pd.DataFrame:
    """加载数据集并运行完整 IC 分析。"""
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please run: python scripts/build_training_data.py first")
        return pd.DataFrame()

    logger.info(f"Loading dataset: {dataset_path}")
    dataset = pd.read_parquet(dataset_path)
    dataset["date"] = pd.to_datetime(dataset["date"])

    logger.info(
        f"Dataset: {len(dataset):,} rows  "
        f"{dataset['date'].min().date()} ~ {dataset['date'].max().date()}  "
        f"positive={dataset['is_strong_pos'].mean():.1%}"
    )

    logger.info("Computing cross-sectional IC for all features...")
    report = compute_all_ic(dataset)

    # 输出到控制台
    _print_report(report)

    # 保存 CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(IC_REPORT_CSV, encoding="utf-8-sig")
    logger.info(f"IC report saved: {IC_REPORT_CSV}")

    # 保存图表
    plot_ic_bar(report, IC_PLOT_BAR)
    plot_ic_timeseries(dataset, report, IC_PLOT_TS)

    return report


def _print_report(report: pd.DataFrame) -> None:
    """在控制台打印格式化报告。"""
    strong = report[report["signal_strength"] == "强有效"]
    weak   = report[report["signal_strength"] == "弱有效"]
    inval  = report[report["signal_strength"] == "无效"]

    print(f"\n{'='*70}")
    print(f"  蓄力期特征 IC 分析报告（共 {len(report)} 个特征）")
    print(f"{'='*70}")

    sections = [
        ("[+] 强有效（|IC|>=0.05, |ICIR|>=0.5）", strong, True),
        ("[~] 弱有效（0.02<=|IC|<0.05）",          weak,   False),
        ("[-] 无效（|IC|<0.02）",                   inval,  False),
    ]
    for title, df, show_all in sections:
        if df.empty:
            continue
        print(f"\n{title}")
        print(f"{'特征':<25} {'IC均值':>8} {'IC_std':>8} {'ICIR':>8} {'正IC率':>8} {'缺失率':>8}")
        print("-" * 70)
        rows = df if show_all or len(df) <= 5 else df.head(3)
        for feat, row in rows.iterrows():
            print(
                f"{feat:<25} "
                f"{row['ic_mean']:>8.4f} "
                f"{row['ic_std']:>8.4f} "
                f"{row['icir']:>8.3f} "
                f"{row['ic_positive_rate']:>8.1%} "
                f"{row['missing_rate']:>8.1%}"
            )
        if not show_all and len(df) > 3:
            print(f"  ...（共 {len(df)} 个）")

    print(f"\n{'='*70}")
    print(f"  进入模型：{len(strong)} 个强有效特征")
    print(f"  观察期：  {len(weak)} 个弱有效特征")
    print(f"  淘汰：    {len(inval)} 个无效特征")
    print(f"{'='*70}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="特征 IC 分析报告")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="训练数据集 parquet 路径")
    args = parser.parse_args()

    run_ic_analysis(Path(args.dataset))


if __name__ == "__main__":
    main()
