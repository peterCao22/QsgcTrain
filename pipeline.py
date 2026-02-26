"""
强势相似选股系统 — 主管线

用法：
    # 完整训练 + Walk-forward 验证（2023-2025年数据）
    python pipeline.py

    # 指定时间范围
    python pipeline.py --start_date 2023-01-01 --end_date 2025-06-30

    # 仅验证"强势相似度假设"（路线B核心假设验证，不需要训练）
    python pipeline.py --mode hypothesis_test

    # 在指定日期预测 TopK（实盘使用）
    python pipeline.py --mode predict --date 2025-01-20 --topk 20

    # 生成完整报告（含图表）
    python pipeline.py --mode report --result_file results/walkforward_20250120_1000.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

# 确保项目根目录在 sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LABEL_COL,
    LOGS_DIR,
    RESULTS_DIR,
    TOPK_VALUES,
    WALKFORWARD_STEP_DAYS,
    WALKFORWARD_TRAIN_DAYS,
)
from data.db_loader import get_trading_days
from data.strong_pool import StrongPoolLoader
from features.builder import FeatureBuilder
from models.similarity import SimilarityScorer, batch_evaluate_hypothesis
from validation.labels import LabelGenerator
from validation.walkforward import WalkForwardEvaluator


# ─── 日志配置 ────────────────────────────────────────────────────────────────

def _setup_logging(mode: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
    log_file = LOGS_DIR / f"pipeline_{mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, level="DEBUG", rotation="50 MB")
    logger.info(f"Pipeline started. Mode: {mode}  Log: {log_file}")


# ─── 辅助：获取周频截面日期 ──────────────────────────────────────────────────

def get_weekly_dates(start: str, end: str, weekday: int = 4) -> List[str]:
    """
    返回 [start, end] 区间内每周固定星期几（weekday=4 = 周五）的交易日列表。
    若该星期几非交易日，取该周最后一个交易日。
    """
    cal = get_trading_days(start, end)
    cal_df = pd.DataFrame({"date": cal})
    cal_df["year_week"] = cal_df["date"].dt.isocalendar().week.astype(str) + "_" + \
                          cal_df["date"].dt.isocalendar().year.astype(str)

    # 每周内选 weekday（0=周一, 4=周五）最近的交易日
    weekly = cal_df.groupby("year_week")["date"].apply(
        lambda g: g[g.dt.weekday <= weekday].max() if (g.dt.weekday <= weekday).any() else g.max()
    ).dropna().sort_values()

    dates = weekly.dt.strftime("%Y-%m-%d").tolist()
    logger.info(f"Weekly dates [{start} ~ {end}]: {len(dates)} dates")
    return dates


# ─── Mode 1：Walk-forward 验证 ────────────────────────────────────────────────

def run_walkforward(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    similarity_weight: float = 0.3,
    topk_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    完整的 Walk-forward 验证流程：
    1. 确定周频截面日期
    2. 生成标签（30日收盘收益）
    3. 构建特征面板
    4. Walk-forward 训练 + 评估

    Returns:
        验证结果 DataFrame
    """
    topk_values = topk_values or TOPK_VALUES
    logger.info(f"Run walkforward: {start_date} ~ {end_date}")

    # ── 1. 获取截面日期 ──────────────────────────────────────────────────────
    dates = get_weekly_dates(start_date, end_date)
    if len(dates) < WALKFORWARD_TRAIN_DAYS + 5:
        raise ValueError(
            f"Not enough dates ({len(dates)}) for walk-forward "
            f"(train_window={WALKFORWARD_TRAIN_DAYS})"
        )

    # ── 2. 生成标签 ──────────────────────────────────────────────────────────
    logger.info("Step 1/4: Generating labels...")
    label_gen = LabelGenerator(apply_buyability_filter=True)
    # 标签需要覆盖到 dates[-1] + 30个交易日
    label_df = label_gen.generate(dates)

    if label_df.empty:
        raise RuntimeError("Label generation failed. Check database connection.")

    logger.info(f"Labels: {len(label_df):,} rows  "
                f"mean_ret={label_df[LABEL_COL].mean():.3f}")

    # ── 3. 构建特征面板 ──────────────────────────────────────────────────────
    logger.info("Step 2/4: Building feature panel...")
    # 数据加载起始需要比第一个截面日期早 252 交易日（特征窗口）
    feature_start = _offset_cal(start_date, -300)
    builder = FeatureBuilder(
        start=feature_start,
        end=end_date,
        normalize="rank",
    )
    builder.preload()

    panel = builder.build_panel(dates, label_df, filter_features=True)
    if panel.empty:
        raise RuntimeError("Feature panel is empty. Check database.")

    logger.info(
        f"Panel: {len(panel):,} rows  "
        f"features={len([c for c in panel.columns if c not in {'date','instrument',LABEL_COL,'rank_label'}])}"
    )

    # ── 4. Walk-forward 评估 ─────────────────────────────────────────────────
    logger.info("Step 3/4: Running walk-forward evaluation...")
    strong_pool = StrongPoolLoader()
    evaluator = WalkForwardEvaluator(
        topk_values=topk_values,
        train_window=WALKFORWARD_TRAIN_DAYS,
        step=WALKFORWARD_STEP_DAYS,
        similarity_weight=similarity_weight,
    )
    result = evaluator.run(
        panel=panel,
        label_df=label_df,
        strong_pool_loader=strong_pool if similarity_weight > 0 else None,
        feature_builder=builder if similarity_weight > 0 else None,
    )

    # ── 5. 可视化报告 ────────────────────────────────────────────────────────
    logger.info("Step 4/4: Generating report...")
    evaluator.plot_report(result)

    return result


# ─── Mode 2：假设验证 ─────────────────────────────────────────────────────────

def run_hypothesis_test(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    验证核心假设：
    "在截面日期 t，与强势股池相似的股票，30日后收益是否更高？"

    Returns:
        各日期的 Spearman IC / L/S Spread 汇总 DataFrame
    """
    logger.info("Running hypothesis test...")

    dates = get_weekly_dates(start_date, end_date)

    # 生成标签
    label_gen = LabelGenerator(apply_buyability_filter=False)
    label_df = label_gen.generate(dates)
    if label_df.empty:
        raise RuntimeError("No labels generated")

    # 构建特征
    builder = FeatureBuilder(start=_offset_cal(start_date, -300), end=end_date)
    builder.preload()

    # 强势股池
    strong_pool = StrongPoolLoader()
    available_dates = strong_pool.get_snapshot_dates()
    if not available_dates:
        logger.warning(
            "No strong pool snapshots found. "
            "Place CSV files in data/strong_pool_snapshots/ and retry."
        )
        return pd.DataFrame()

    test_dates = [d for d in dates if d in available_dates]
    if not test_dates:
        logger.warning(
            f"None of the weekly dates match available snapshots. "
            f"Available: {available_dates[:5]}..."
        )
        return pd.DataFrame()

    summary = batch_evaluate_hypothesis(
        feature_builder=builder,
        strong_pool_loader=strong_pool,
        label_df=label_df,
        dates=test_dates,
        method="mean",
    )

    if not summary.empty:
        out_path = RESULTS_DIR / "hypothesis_test.csv"
        summary.to_csv(out_path, index=False)
        logger.info(f"Hypothesis test results saved: {out_path}")

    return summary


# ─── Mode 3：单日预测（实盘使用）─────────────────────────────────────────────

def run_predict(
    date: str,
    topk: int = 20,
    model_path: Optional[Path] = None,
    strong_pool_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    在指定日期预测全市场股票排序，输出 TopK 候选。

    Args:
        date:             预测截面日期（用该日收盘后的特征）
        topk:             输出 Top 多少只
        model_path:       已训练模型路径；None 时提示用户先训练
        strong_pool_date: 强势股池日期（默认等于 date）

    Returns:
        TopK 候选 DataFrame：[rank, instrument, score, similarity_score]
    """
    from models.baseline import BaselineRankModel

    logger.info(f"Predicting @ {date}  TopK={topk}")

    # 加载模型
    if model_path is None or not model_path.exists():
        logger.warning(
            "No trained model found. Run pipeline first:\n"
            "  python pipeline.py --start_date 2023-01-01 --end_date 2025-12-31"
        )
        return pd.DataFrame()

    model = BaselineRankModel.load(model_path)

    # 构建特征
    start = _offset_cal(date, -300)
    builder = FeatureBuilder(start=start, end=date)
    builder.preload()
    feat_df = builder.build(date)

    if feat_df.empty:
        logger.error(f"No features built for {date}")
        return pd.DataFrame()

    # 路线A分数
    feat_cols = model._feature_cols or [c for c in feat_df.columns]
    scores_a = model.predict(feat_df[feat_cols].fillna(0.0))

    # 路线B相似度分数
    strong_pool = StrongPoolLoader()
    sdate = strong_pool_date or date
    strong_insts = strong_pool.get(sdate)

    if strong_insts:
        strong_feat = feat_df.loc[feat_df.index.isin(strong_insts)]
        if not strong_feat.empty:
            scorer = SimilarityScorer(method="mean")
            scorer.fit_prototype(strong_feat)
            sim_scores = scorer.score(feat_df)
        else:
            sim_scores = pd.Series(0.0, index=feat_df.index)
    else:
        sim_scores = pd.Series(0.0, index=feat_df.index)

    # 融合
    from validation.walkforward import _rank_scale
    import numpy as np
    a_norm = _rank_scale(scores_a)
    b_norm = _rank_scale(sim_scores.values)
    final = 0.7 * a_norm + 0.3 * b_norm

    result = pd.DataFrame({
        "instrument": feat_df.index,
        "score": final,
        "score_a": a_norm,
        "similarity_score": b_norm,
    })
    result = result.nlargest(topk, "score").reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    print(f"\n{'='*60}")
    print(f"Top {topk} Stock Picks @ {date}")
    print(f"{'='*60}")
    print(result[["rank", "instrument", "score", "similarity_score"]].to_string(index=False))

    out_path = RESULTS_DIR / f"predict_{date}_top{topk}.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Prediction saved: {out_path}")
    return result


# ─── Mode 4：从已有结果文件生成报告 ─────────────────────────────────────────

def run_report(result_file: str) -> None:
    """从已保存的 CSV 结果文件重新生成可视化报告。"""
    path = Path(result_file)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return
    result = pd.read_csv(path)
    result["date"] = pd.to_datetime(result["date"])
    evaluator = WalkForwardEvaluator()
    evaluator.print_summary(result)
    evaluator.plot_report(result, save_path=path.with_suffix(".png"))


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

def _offset_cal(date: str, days: int) -> str:
    ts = pd.Timestamp(date) + pd.Timedelta(days=max(days, days * 1.5))
    return ts.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="强势相似选股系统 Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["walkforward", "hypothesis_test", "predict", "report"],
        default="walkforward",
        help=(
            "walkforward:      完整训练 + Walk-forward 验证（默认）\n"
            "hypothesis_test:  仅验证强势相似度假设\n"
            "predict:          单日预测 TopK（需要已训练模型）\n"
            "report:           从已有结果文件生成报告"
        ),
    )
    parser.add_argument("--start_date", default="2023-01-01", help="数据起始日期")
    parser.add_argument("--end_date",   default="2025-12-31", help="数据截止日期")
    parser.add_argument("--date",       default=None,         help="[predict 模式] 预测日期")
    parser.add_argument("--topk",       type=int, default=20, help="[predict 模式] 输出 TopK 数量")
    parser.add_argument(
        "--model_path", default=None,
        help="[predict 模式] 已训练模型路径（.lgb 文件）"
    )
    parser.add_argument(
        "--result_file", default=None,
        help="[report 模式] 已有结果 CSV 文件路径"
    )
    parser.add_argument(
        "--similarity_weight", type=float, default=0.3,
        help="[walkforward 模式] 相似度分数融合权重（0=仅路线A，默认0.3）"
    )
    parser.add_argument(
        "--no_similarity", action="store_true",
        help="[walkforward 模式] 禁用路线B相似度融合（仅运行路线A基线）"
    )

    args = parser.parse_args()
    _setup_logging(args.mode)

    if args.mode == "walkforward":
        sw = 0.0 if args.no_similarity else args.similarity_weight
        run_walkforward(
            start_date=args.start_date,
            end_date=args.end_date,
            similarity_weight=sw,
        )

    elif args.mode == "hypothesis_test":
        run_hypothesis_test(
            start_date=args.start_date,
            end_date=args.end_date,
        )

    elif args.mode == "predict":
        if args.date is None:
            parser.error("--date is required for predict mode")
        mp = Path(args.model_path) if args.model_path else None
        run_predict(date=args.date, topk=args.topk, model_path=mp)

    elif args.mode == "report":
        if args.result_file is None:
            parser.error("--result_file is required for report mode")
        run_report(args.result_file)


if __name__ == "__main__":
    main()
