"""
强势股蓄力期特征画像分析

从训练数据集（training_dataset.parquet）中，对正样本（强势股入池前约3个月）
与负样本（同期普通市场股票）做特征分布对比，识别共有特征并写出特征库。

输出：
  data/feature_library.json        特征库（显著特征列表 + 统计摘要）
  results/strong_stock_profile.csv  完整统计报告（Excel 可打开）
  results/strong_stock_boxplot.png  18 个特征箱线图
  results/strong_stock_radar.png    强势股 vs 普通股雷达图

运行方式：
    conda activate rqsdk
    python -X utf8 scripts/profile_strong_stocks.py
    python -X utf8 scripts/profile_strong_stocks.py --dataset data/training_dataset.parquet
    python -X utf8 scripts/profile_strong_stocks.py --p-threshold 0.05 --d-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mannwhitneyu

from config import DATA_DIR, RESULTS_DIR
from features.precursor import ALL_FEATURE_COLS

# ─── 常量 ─────────────────────────────────────────────────────────────────────

DATASET_PATH    = DATA_DIR / "training_dataset.parquet"
FEATURE_LIB_OUT = DATA_DIR / "feature_library.json"
PROFILE_CSV_OUT = RESULTS_DIR / "strong_stock_profile.csv"
BOXPLOT_OUT     = RESULTS_DIR / "strong_stock_boxplot.png"
RADAR_OUT       = RESULTS_DIR / "strong_stock_radar.png"

DEFAULT_P_THRESHOLD = 0.05   # 入库显著性阈值
DEFAULT_D_THRESHOLD = 0.05   # 入库效应量阈值（|Cohen's d|）
# 注：同板块对比时效应量天然偏小（板块Beta被控制掉），0.05 为合理下限

# 特征中文名称（用于图表标注）
FEATURE_CN: Dict[str, str] = {
    "vol_slope_60d":         "成交量60日斜率",
    "price_slope_60d":       "价格超额斜率",
    "mf_slope_30d":          "主力资金流斜率",
    "vol_accel":             "量能加速比",
    "range_compress":        "振幅收窄比",
    "momentum_accel":        "动量加速度",
    "excess_ret_change":     "超额收益变化",
    "days_above_ma20":       "站上MA20天数",
    "consec_higher_low":     "连续更高低点",
    "days_positive_mf":      "净流入连续天数",
    "dist_52w_high":         "距年高距离",
    "dist_60d_low_rebound":  "60日低点反弹",
    "win_percent":           "筹码盈利比例",
    "chip_concentration":    "筹码集中度",
    "price_to_avgcost":      "价格/均摊成本",
    "prev_tj_boards":        "历史最高连板",
    "prev_new_high_count":   "历史新高次数",
    "prev_pool_appearances": "历史入池次数",
}

# 特征类别（用于分组显示）
FEATURE_CATEGORY: Dict[str, str] = {
    "vol_slope_60d": "A趋势斜率", "price_slope_60d": "A趋势斜率", "mf_slope_30d": "A趋势斜率",
    "vol_accel": "B比率变化", "range_compress": "B比率变化",
    "momentum_accel": "B比率变化", "excess_ret_change": "B比率变化",
    "days_above_ma20": "C持续时间", "consec_higher_low": "C持续时间", "days_positive_mf": "C持续时间",
    "dist_52w_high": "D层级对比", "dist_60d_low_rebound": "D层级对比",
    "win_percent": "D层级对比", "chip_concentration": "D层级对比", "price_to_avgcost": "D层级对比",
    "prev_tj_boards": "E历史强势", "prev_new_high_count": "E历史强势", "prev_pool_appearances": "E历史强势",
}


# ─── 统计计算 ─────────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Cohen's d 效应量（正值=强势股均值更大）。"""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def analyze_features(
    strong: pd.DataFrame,
    market: pd.DataFrame,
    feature_cols: List[str],
    p_threshold: float,
    d_threshold: float,
) -> pd.DataFrame:
    """
    对每个特征做双样本对比分析。

    Returns:
        DataFrame，每行一个特征，含统计指标和入库标记。
    """
    rows = []
    for feat in feature_cols:
        s = strong[feat].dropna().values
        m = market[feat].dropna().values
        if len(s) < 10 or len(m) < 10:
            continue

        try:
            _, pval = mannwhitneyu(s, m, alternative="two-sided")
        except Exception:
            pval = np.nan

        d = cohens_d(s, m)

        rows.append({
            "feature":         feat,
            "feature_cn":      FEATURE_CN.get(feat, feat),
            "category":        FEATURE_CATEGORY.get(feat, "其他"),
            "strong_mean":     float(np.mean(s)),
            "strong_median":   float(np.median(s)),
            "strong_p25":      float(np.percentile(s, 25)),
            "strong_p75":      float(np.percentile(s, 75)),
            "market_mean":     float(np.mean(m)),
            "market_median":   float(np.median(m)),
            "market_p25":      float(np.percentile(m, 25)),
            "market_p75":      float(np.percentile(m, 75)),
            "cohens_d":        d,
            "abs_d":           abs(d) if not np.isnan(d) else np.nan,
            "pval":            float(pval) if not np.isnan(pval) else np.nan,
            "n_strong":        len(s),
            "n_market":        len(m),
            "selected":        (
                not np.isnan(pval) and pval < p_threshold and
                not np.isnan(d)    and abs(d) >= d_threshold
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_d", ascending=False).reset_index(drop=True)
    return df


# ─── 特征库写出 ───────────────────────────────────────────────────────────────

def save_feature_library(
    stats_df: pd.DataFrame,
    n_positive: int,
    n_total: int,
    lookback_days: int,
    p_threshold: float,
    d_threshold: float,
) -> None:
    """将筛选结果写出为 feature_library.json。"""
    selected = stats_df[stats_df["selected"]]["feature"].tolist()
    all_stats = {}
    for _, row in stats_df.iterrows():
        all_stats[row["feature"]] = {
            "feature_cn":    row["feature_cn"],
            "category":      row["category"],
            "cohens_d":      round(float(row["cohens_d"]), 4) if not np.isnan(row["cohens_d"]) else None,
            "pval":          float(row["pval"]) if not np.isnan(row["pval"]) else None,
            "strong_median": round(float(row["strong_median"]), 6),
            "market_median": round(float(row["market_median"]), 6),
            "selected":      bool(row["selected"]),
        }

    lib = {
        "version":           str(date.today()),
        "selected_features": selected,
        "feature_stats":     all_stats,
        "total_positive_samples": n_positive,
        "total_samples":     n_total,
        "lookback_days":     lookback_days,
        "selection_criteria": {
            "p_threshold": p_threshold,
            "d_threshold": d_threshold,
        },
    }

    FEATURE_LIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_LIB_OUT, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=2)
    logger.info(f"Feature library saved: {FEATURE_LIB_OUT}  ({len(selected)} features selected)")


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_boxplots(
    strong: pd.DataFrame,
    market: pd.DataFrame,
    stats_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    """绘制 18 个特征的箱线图（强势股 vs 普通股），高亮入库特征。"""
    ncols = 6
    nrows = int(np.ceil(len(feature_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    selected_set = set(stats_df[stats_df["selected"]]["feature"].tolist())

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        s_vals = strong[feat].dropna().values
        m_vals = market[feat].dropna().values

        # 截断异常值（1~99百分位）以便可视化
        if len(s_vals) > 0 and len(m_vals) > 0:
            lo = np.percentile(np.concatenate([s_vals, m_vals]), 1)
            hi = np.percentile(np.concatenate([s_vals, m_vals]), 99)
            s_plot = s_vals[(s_vals >= lo) & (s_vals <= hi)]
            m_plot = m_vals[(m_vals >= lo) & (m_vals <= hi)]
        else:
            s_plot, m_plot = s_vals, m_vals

        bp = ax.boxplot(
            [m_plot, s_plot],
            labels=["普通", "强势"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#DD8452")
        bp["boxes"][1].set_alpha(0.7)

        feat_cn = FEATURE_CN.get(feat, feat)
        row = stats_df[stats_df["feature"] == feat]
        if not row.empty:
            d_val = row.iloc[0]["cohens_d"]
            p_val = row.iloc[0]["pval"]
            p_str = f"p={p_val:.3f}" if not np.isnan(p_val) and p_val >= 0.001 else "p<0.001"
            d_str = f"d={d_val:.2f}" if not np.isnan(d_val) else ""
            subtitle = f"{p_str}  {d_str}"
        else:
            subtitle = ""

        is_selected = feat in selected_set
        title_color = "#c0392b" if is_selected else "#555555"
        title_prefix = "[*] " if is_selected else ""
        ax.set_title(f"{title_prefix}{feat_cn}\n{subtitle}",
                     fontsize=8, color=title_color, pad=3)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    strong_patch = mpatches.Patch(color="#DD8452", alpha=0.7, label="强势股")
    market_patch = mpatches.Patch(color="#4C72B0", alpha=0.7, label="普通股")
    fig.legend(handles=[strong_patch, market_patch],
               loc="lower right", fontsize=10, framealpha=0.9)
    fig.suptitle("强势股蓄力期特征画像  [*]=入特征库", fontsize=13, y=1.01)
    plt.tight_layout()

    BOXPLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOXPLOT_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplot saved: {BOXPLOT_OUT}")


def plot_radar(
    strong: pd.DataFrame,
    market: pd.DataFrame,
    selected_features: List[str],
) -> None:
    """
    绘制强势股 vs 普通股雷达图（选用入库特征，各自归一化到 [0,1]）。
    """
    feats = [f for f in selected_features if f in strong.columns and f in market.columns]
    if len(feats) < 3:
        logger.warning("选入特征不足 3 个，跳过雷达图生成")
        return

    s_medians = strong[feats].median().values
    m_medians = market[feats].median().values

    # 逐特征归一化到 [0,1]（相对于两组合并的范围）
    combined = np.vstack([s_medians, m_medians])
    cmin = combined.min(axis=0)
    cmax = combined.max(axis=0)
    rng_val = cmax - cmin
    rng_val[rng_val < 1e-10] = 1.0
    s_norm = (s_medians - cmin) / rng_val
    m_norm = (m_medians - cmin) / rng_val

    N = len(feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    s_vals = s_norm.tolist() + s_norm[:1].tolist()
    m_vals = m_norm.tolist() + m_norm[:1].tolist()

    labels = [FEATURE_CN.get(f, f) for f in feats]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, s_vals, "o-", linewidth=2, color="#DD8452", label="强势股")
    ax.fill(angles, s_vals, alpha=0.20, color="#DD8452")
    ax.plot(angles, m_vals, "s--", linewidth=2, color="#4C72B0", label="普通股")
    ax.fill(angles, m_vals, alpha=0.10, color="#4C72B0")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_yticklabels([])
    ax.set_title("强势股蓄力期指纹  vs  普通股", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    RADAR_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RADAR_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Radar chart saved: {RADAR_OUT}")


# ─── 控制台报告 ───────────────────────────────────────────────────────────────

def print_report(stats_df: pd.DataFrame, n_pos: int, n_total: int) -> None:
    """打印可读的特征画像摘要。"""
    selected = stats_df[stats_df["selected"]]
    not_selected = stats_df[~stats_df["selected"]]

    print("\n" + "=" * 70)
    print(f"  强势股蓄力期特征画像")
    print(f"  正样本（新入池）: {n_pos:,}   总样本: {n_total:,}")
    print(f"  特征总数: {len(stats_df)}   入库数: {len(selected)}")
    print("=" * 70)

    if not selected.empty:
        print(f"\n  [+] 入库特征（p<{DEFAULT_P_THRESHOLD}, |d|>={DEFAULT_D_THRESHOLD}），按效应量排序：\n")
        header = f"  {'排名':<4} {'特征':<26} {'中文名':<16} {'强势中位数':>12} {'市场中位数':>12} {'Cohen d':>9} {'p值':>10}"
        print(header)
        print("  " + "-" * 90)
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            d_str  = f"{row['cohens_d']:+.3f}" if not np.isnan(row['cohens_d']) else "  N/A "
            pv_str = f"{row['pval']:.4f}" if not np.isnan(row['pval']) and row['pval'] >= 0.0001 else "<0.0001"
            direction = "^强势更高" if row['cohens_d'] > 0 else "v强势更低"
            print(f"  {rank:<4} {row['feature']:<26} {row['feature_cn']:<16} "
                  f"{row['strong_median']:>12.4f} {row['market_median']:>12.4f} "
                  f"{d_str:>9} {pv_str:>10}  {direction}")

    if not not_selected.empty:
        print(f"\n  [-] 未入库特征（{len(not_selected)} 个）：")
        names = [f"{r['feature']}({r['feature_cn']})" for _, r in not_selected.iterrows()]
        print("  " + ", ".join(names))

    print("\n" + "=" * 70 + "\n")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_profile(
    dataset_path: Path,
    p_threshold: float,
    d_threshold: float,
) -> None:
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("请先运行: python scripts/build_training_data.py")
        sys.exit(1)

    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    n_total = len(df)
    strong  = df[df["is_strong_pos"] == 1]
    market  = df[df["is_strong_pos"] == 0]
    n_pos   = len(strong)
    n_neg   = len(market)

    logger.info(f"正样本: {n_pos:,}  负样本: {n_neg:,}  总计: {n_total:,}")
    logger.info(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    if n_pos < 20:
        logger.error("正样本数量不足（<20），无法进行可靠分析。请先扩充训练数据。")
        sys.exit(1)

    # 可用特征列
    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    logger.info(f"分析特征数: {len(feat_cols)}")

    # 统计分析
    logger.info("计算特征分布统计...")
    stats_df = analyze_features(strong, market, feat_cols, p_threshold, d_threshold)

    # 保存 CSV
    PROFILE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(PROFILE_CSV_OUT, index=False, encoding="utf-8-sig")
    logger.info(f"Profile CSV saved: {PROFILE_CSV_OUT}")

    # 写出特征库
    lookback = n_pos  # 默认用样本数作为参考，实际 lookback 来自模型配置
    save_feature_library(
        stats_df     = stats_df,
        n_positive   = n_pos,
        n_total      = n_total,
        lookback_days = 65,
        p_threshold  = p_threshold,
        d_threshold  = d_threshold,
    )

    # 箱线图
    logger.info("生成箱线图...")
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    plot_boxplots(strong, market, stats_df, feat_cols)

    # 雷达图（仅使用入库特征）
    selected_features = stats_df[stats_df["selected"]]["feature"].tolist()
    if len(selected_features) >= 3:
        logger.info(f"生成雷达图（{len(selected_features)} 个入库特征）...")
        plot_radar(strong, market, selected_features)
    else:
        logger.warning(f"入库特征只有 {len(selected_features)} 个（<3），跳过雷达图")

    # 控制台报告
    print_report(stats_df, n_pos, n_total)

    print(f"输出文件：")
    print(f"  特征库：   {FEATURE_LIB_OUT}")
    print(f"  统计报告：{PROFILE_CSV_OUT}")
    print(f"  箱线图：  {BOXPLOT_OUT}")
    if len(selected_features) >= 3:
        print(f"  雷达图：  {RADAR_OUT}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="强势股蓄力期特征画像分析")
    parser.add_argument("--dataset",     default=str(DATASET_PATH),
                        help=f"训练数据集路径（默认 {DATASET_PATH}）")
    parser.add_argument("--p-threshold", type=float, default=DEFAULT_P_THRESHOLD,
                        help=f"入库显著性阈值（默认 {DEFAULT_P_THRESHOLD}）")
    parser.add_argument("--d-threshold", type=float, default=DEFAULT_D_THRESHOLD,
                        help=f"入库效应量阈值 |d| >= 此值（默认 {DEFAULT_D_THRESHOLD}）")
    args = parser.parse_args()

    run_profile(
        dataset_path = Path(args.dataset),
        p_threshold  = args.p_threshold,
        d_threshold  = args.d_threshold,
    )


if __name__ == "__main__":
    main()
