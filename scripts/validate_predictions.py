"""
预测结果验证脚本（Step 6）

读取历史扫描结果（results/scan_YYYYMMDD.csv），
计算从扫描日起约 1 个月（22 个交易日）的实际涨幅，
将实际表现与模型预测得分对比，输出验证报告。

指标：
  - Top-K 平均实际涨幅 vs 全市场均值
  - Precision@K（实际涨幅超过全市场前 20% 的比率）
  - 得分-涨幅 Spearman 相关系数
  - 涨幅分布（直方图）

输出：
  results/validation_YYYYMMDD.csv  详细验证数据
  results/validation_YYYYMMDD.png  可视化图表（涨幅分布 + 得分散点图）
  控制台打印摘要报告

运行方式：
    conda activate rqsdk

    # 验证某次历史扫描
    python -X utf8 scripts/validate_predictions.py --scan-date 2025-06-01

    # 验证指定文件
    python -X utf8 scripts/validate_predictions.py --scan-file results/scan_20250601.csv

    # 批量验证所有历史扫描
    python -X utf8 scripts/validate_predictions.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from config import RESULTS_DIR
from data.db_loader import offset_trading_day
from data.market_loader import load_kline

# ─── 常量 ─────────────────────────────────────────────────────────────────────

FORWARD_DAYS   = 22   # 约1个月（22个交易日）
TOPK_VALUES    = [10, 20, 50]
TOP_PCT_THRESHOLD = 0.20   # Precision@K 的正样本定义（涨幅前20%）


# ─── 收益率计算 ───────────────────────────────────────────────────────────────

def compute_actual_returns(
    instruments: List[str],
    scan_date: str,
    forward_days: int = FORWARD_DAYS,
) -> Dict[str, float]:
    """
    计算从 scan_date 起 forward_days 个交易日的实际涨幅。

    Returns:
        {instrument: return_ratio}，无法计算的返回 NaN
    """
    scan_ts = pd.Timestamp(scan_date)
    try:
        end_ts = offset_trading_day(scan_ts, forward_days)
    except IndexError:
        logger.warning(f"Cannot compute end date ({forward_days} days from {scan_date}); "
                       f"the date may be too recent.")
        return {inst: np.nan for inst in instruments}

    end_str   = str(end_ts.date()) if hasattr(end_ts, "date") else str(end_ts)[:10]
    start_str = str((scan_ts - pd.Timedelta(days=5)).date())

    logger.info(f"Loading kline for validation: {scan_date} ~ {end_str}")
    kline = load_kline(start_str, end_str, instruments=instruments)
    if kline.empty:
        return {inst: np.nan for inst in instruments}

    kline["date"] = pd.to_datetime(kline["date"])

    close_t  = kline[kline["date"] == scan_ts].set_index("instrument")["close"]
    close_tn = kline[kline["date"] == end_ts].set_index("instrument")["close"]

    merged = close_t.rename("t").to_frame().join(close_tn.rename("tn"), how="inner")
    merged = merged[(merged["t"] > 0) & (merged["tn"] > 0)]
    merged["ret"] = merged["tn"] / merged["t"] - 1
    merged.loc[merged["ret"].abs() > 3.0, "ret"] = np.nan  # 过滤极端值

    result = merged["ret"].to_dict()
    for inst in instruments:
        if inst not in result:
            result[inst] = np.nan
    return result


# ─── 评估指标 ─────────────────────────────────────────────────────────────────

def evaluate(
    df: pd.DataFrame,
    topk_values: List[int] = TOPK_VALUES,
    top_pct: float = TOP_PCT_THRESHOLD,
) -> Dict:
    """
    计算验证指标。

    Args:
        df: 含 score、actual_return 列，按 score 降序排列
    """
    valid = df.dropna(subset=["actual_return"])
    if len(valid) < 5:
        return {}

    market_ret = valid["actual_return"].mean()
    market_top_threshold = valid["actual_return"].quantile(1 - top_pct)

    metrics = {
        "scan_date":         df["scan_date"].iloc[0] if "scan_date" in df.columns else "—",
        "n_total":           len(df),
        "n_valid":           len(valid),
        "market_avg_return": float(market_ret),
        "market_top_threshold": float(market_top_threshold),
    }

    # Spearman 相关系数（得分 vs 实际涨幅）
    spearman_corr, spearman_pval = spearmanr(
        valid["score"].values, valid["actual_return"].values
    )
    metrics["spearman_corr"] = float(spearman_corr)
    metrics["spearman_pval"] = float(spearman_pval)

    # Top-K 指标
    for k in topk_values:
        top_k = valid.nlargest(min(k, len(valid)), "score")
        avg_ret  = float(top_k["actual_return"].mean())
        prec_k   = float((top_k["actual_return"] >= market_top_threshold).mean())
        beat_mkt = float((top_k["actual_return"] > market_ret).mean())
        metrics[f"top{k}_avg_return"]  = avg_ret
        metrics[f"top{k}_precision"]   = prec_k
        metrics[f"top{k}_beat_market"] = beat_mkt

    return metrics


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_validation(df: pd.DataFrame, metrics: Dict, out_path: Path) -> None:
    """绘制验证图表：涨幅分布直方图 + 得分-涨幅散点图。"""
    valid = df.dropna(subset=["actual_return"])
    if len(valid) < 5:
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    scan_date = metrics.get("scan_date", "")
    fig.suptitle(f"预测验证报告  扫描日: {scan_date}  (持有 {FORWARD_DAYS} 交易日)",
                 fontsize=12)

    # -- 左图：涨幅分布对比 --
    top20 = valid.nlargest(min(20, len(valid)), "score")
    all_ret = valid["actual_return"].values * 100
    top_ret = top20["actual_return"].values * 100
    bins = np.linspace(
        np.percentile(all_ret[~np.isnan(all_ret)], 1),
        np.percentile(all_ret[~np.isnan(all_ret)], 99),
        30
    )
    ax1.hist(all_ret, bins=bins, alpha=0.5, color="#4C72B0", label=f"全部候选({len(valid)}只)")
    ax1.hist(top_ret, bins=bins, alpha=0.7, color="#DD8452", label=f"Top20({len(top20)}只)")
    ax1.axvline(metrics.get("market_avg_return", 0) * 100, color="red",
                linestyle="--", linewidth=1.5, label="市场均值")
    ax1.set_xlabel("实际涨幅 (%)")
    ax1.set_ylabel("频次")
    ax1.set_title("涨幅分布对比")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # -- 右图：得分 vs 实际涨幅散点 --
    sc = valid.nlargest(min(200, len(valid)), "score")  # 只展示 Top-200 避免过度密集
    ax2.scatter(sc["score"].values, sc["actual_return"].values * 100,
                alpha=0.5, s=20, color="#4C72B0")
    corr = metrics.get("spearman_corr", np.nan)
    ax2.set_xlabel("模型得分")
    ax2.set_ylabel("实际涨幅 (%)")
    ax2.set_title(f"得分 vs 实际涨幅  (Spearman r={corr:.3f})")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Validation chart saved: {out_path}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def validate_scan(scan_file: Path, forward_days: int = FORWARD_DAYS) -> Optional[Dict]:
    """验证单次历史扫描结果。"""
    if not scan_file.exists():
        logger.error(f"Scan file not found: {scan_file}")
        return None

    logger.info(f"Validating: {scan_file}")
    scan_df = pd.read_csv(scan_file)
    if "instrument" not in scan_df.columns or "score" not in scan_df.columns:
        logger.error(f"Invalid scan file format: needs 'instrument' and 'score' columns")
        return None

    # 从文件名推断扫描日期
    stem = scan_file.stem  # e.g. scan_20250601
    if "scan_" in stem:
        date_part = stem.replace("scan_", "")
        if len(date_part) == 8:
            scan_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
        else:
            scan_date = date_part
    else:
        scan_date = str(scan_df.get("date", ["unknown"]).iloc[0])

    instruments = scan_df["instrument"].tolist()

    # 计算实际涨幅
    actual_rets = compute_actual_returns(instruments, scan_date, forward_days=forward_days)
    scan_df["actual_return"] = scan_df["instrument"].map(actual_rets)
    scan_df["scan_date"]     = scan_date

    n_valid = scan_df["actual_return"].notna().sum()
    if n_valid < 5:
        logger.warning(
            f"Only {n_valid} instruments have actual returns. "
            f"The scan date may be too recent (need {forward_days} trading days of history)."
        )

    # 计算指标
    metrics = evaluate(scan_df)
    metrics["scan_date"]    = scan_date
    metrics["forward_days"] = forward_days

    # 保存详细数据
    date_str = scan_date.replace("-", "")
    out_csv  = RESULTS_DIR / f"validation_{date_str}.csv"
    out_png  = RESULTS_DIR / f"validation_{date_str}.png"
    scan_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Validation CSV saved: {out_csv}")

    # 可视化
    if n_valid >= 5:
        plot_validation(scan_df, metrics, out_png)

    return metrics


def print_metrics(metrics: Dict) -> None:
    """控制台打印验证指标。"""
    fd = metrics.get("forward_days", FORWARD_DAYS)
    print("\n" + "=" * 60)
    print(f"  验证报告  扫描日期: {metrics.get('scan_date')}  "
          f"持有 {fd} 交易日")
    print("=" * 60)
    print(f"  样本数:   总计 {metrics.get('n_total',0)}  有效 {metrics.get('n_valid',0)}")
    print(f"  市场均值: {metrics.get('market_avg_return', np.nan)*100:.2f}%")
    print(f"  Spearman: r={metrics.get('spearman_corr', np.nan):.4f}  "
          f"p={metrics.get('spearman_pval', np.nan):.4f}")
    print()
    for k in TOPK_VALUES:
        avg_r  = metrics.get(f"top{k}_avg_return",  np.nan) * 100
        prec   = metrics.get(f"top{k}_precision",   np.nan) * 100
        beat   = metrics.get(f"top{k}_beat_market", np.nan) * 100
        excess = avg_r - metrics.get("market_avg_return", 0) * 100
        print(f"  Top-{k:<3}: 均涨幅={avg_r:+.2f}%  超额={excess:+.2f}%  "
              f"精确率={prec:.1f}%  跑赢大市={beat:.1f}%")
    print("=" * 60 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="验证历史扫描预测结果的实际涨幅")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scan-date", default=None,
                       help="扫描日期（自动找 results/scan_YYYYMMDD.csv）")
    group.add_argument("--scan-file", default=None,
                       help="直接指定扫描结果文件路径")
    group.add_argument("--all",  action="store_true", default=False,
                       help="批量验证 results/ 下所有 scan_*.csv 文件")
    parser.add_argument("--forward-days", type=int, default=FORWARD_DAYS,
                        help=f"持有天数（默认 {FORWARD_DAYS} 交易日）")
    args = parser.parse_args()

    if args.all:
        scan_files = sorted(RESULTS_DIR.glob("scan_*.csv"))
        if not scan_files:
            print(f"没有找到扫描结果文件（{RESULTS_DIR}/scan_*.csv）")
            sys.exit(0)
        print(f"找到 {len(scan_files)} 个扫描结果，开始批量验证...")
        all_metrics = []
        for sf in scan_files:
            m = validate_scan(sf, forward_days=args.forward_days)
            if m:
                all_metrics.append(m)
                print_metrics(m)

        if all_metrics:
            summary = pd.DataFrame(all_metrics)
            summary_path = RESULTS_DIR / "validation_summary.csv"
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"\n汇总已保存至: {summary_path}")

    elif args.scan_file:
        m = validate_scan(Path(args.scan_file), forward_days=args.forward_days)
        if m:
            print_metrics(m)

    elif args.scan_date:
        date_str = args.scan_date.replace("-", "")
        scan_file = RESULTS_DIR / f"scan_{date_str}.csv"
        m = validate_scan(scan_file, forward_days=args.forward_days)
        if m:
            print_metrics(m)

    else:
        # 默认：验证最新一次扫描
        scan_files = sorted(RESULTS_DIR.glob("scan_*.csv"))
        if not scan_files:
            print(f"没有找到扫描结果文件（{RESULTS_DIR}/scan_*.csv）")
            print("请先运行: python -X utf8 scripts/daily_scan.py")
            sys.exit(0)
        latest = scan_files[-1]
        print(f"验证最新扫描结果: {latest}")
        m = validate_scan(latest)
        if m:
            print_metrics(m)


if __name__ == "__main__":
    main()
