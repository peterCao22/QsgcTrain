"""
全市场每日蓄力期扫描

对全市场可交易股票提取与训练时完全相同的蓄力期特征，
用已训练好的 LightGBM 模型打出"pre-strong 概率分"，
按分数从高到低排列，输出 Top-K 候选股名单。

输出：
  results/scan_YYYYMMDD.csv   当日候选股详情（含各特征值 + 得分）
  控制台打印 Top-N 名单

运行方式：
    conda activate rqsdk

    # 扫描今天
    python -X utf8 scripts/daily_scan.py

    # 扫描指定日期（历史回测验证）
    python -X utf8 scripts/daily_scan.py --date 2025-06-01

    # 只输出 Top-30，不含特征列
    python -X utf8 scripts/daily_scan.py --topk 30 --no-features
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from loguru import logger

from config import DATA_DIR, EXCLUDE_PREFIXES, RESULTS_DIR
from data.db_loader import get_tradable_universe, offset_trading_day
from data.market_loader import (
    load_chips,
    load_concept_bar,
    load_concept_component_range,
    load_index_kline,
    load_kline,
    load_moneyflow,
    load_valuation,
)
from features.precursor import ALL_FEATURE_COLS, PrecursorFeatureExtractor
from models.lgb_classifier import FEATURE_LIB_PATH, MODELS_DIR, LGBClassifier, load_feature_library

# ─── 常量 ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = MODELS_DIR / "lgb_classifier.pkl"
DEFAULT_TOPK       = 50
DEFAULT_LOOKBACK   = 280   # 特征计算所需最大回看天数（v2.0 延长至280支持52周周线特征）

# T1: 大市值/稳定行业过滤阈值（流通市值，单位：元）
# 500亿 = 5e10 元。覆盖格力(2106亿)、中国移动(2万亿)等高市值稳定股
FLOAT_CAP_MAX = 5e10  # 500亿元

# T1: 稳定行业的股票前缀或特殊代码（补充兜底规则）
# 主要依赖市值过滤，这里作为额外保障过滤已知的公用事业/高速/大行
_STABLE_INDUSTRY_CODES = {
    "601328",  # 交通银行
    "601398",  # 工商银行
    "601288",  # 农业银行
    "601939",  # 建设银行
    "600036",  # 招商银行
    "601988",  # 中国银行
    "601816",  # 京沪高铁
    "601377",  # 兴业银行
    "600377",  # 宁沪高速
    "600350",  # 山东高速
    "600012",  # 皖通高速
}

# ─── 辅助 ─────────────────────────────────────────────────────────────────────

def _load_strong_pool_hist(lookback_start: str, scan_date: str) -> pd.DataFrame:
    """加载历史强势股池（用于 E 类特征）。"""
    from data.db_loader import read_sql
    try:
        sql = """
            SELECT date, instrument, tj_boards, new_high
            FROM strong_pool
            WHERE date >= :start AND date <= :end
            ORDER BY date, instrument
        """
        df = read_sql(sql, {"start": lookback_start, "end": scan_date})
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"strong_pool hist loaded: {len(df):,} rows")
        return df
    except Exception as exc:
        logger.warning(f"strong_pool unavailable ({exc}); E-type features will be 0")
        return pd.DataFrame(columns=["date", "instrument", "tj_boards", "new_high"])


def _get_universe(scan_date: str) -> List[str]:
    """获取扫描日的全市场可交易股票，过滤科创/北交所。"""
    try:
        instruments = get_tradable_universe(scan_date)
    except Exception as exc:
        logger.warning(f"get_tradable_universe failed ({exc}), falling back to kline universe")
        instruments = []

    if not instruments:
        from data.db_loader import read_sql
        sql = """
            SELECT DISTINCT instrument FROM kline_all
            WHERE date = :d
        """
        df = read_sql(sql, {"d": scan_date})
        instruments = df["instrument"].tolist()

    instruments = [
        i for i in instruments
        if not any(i.startswith(p) for p in EXCLUDE_PREFIXES)
    ]
    logger.info(f"Universe on {scan_date}: {len(instruments)} stocks")
    return instruments


def _top_feature_summary(row: pd.Series, feat_cols: List[str], top_n: int = 3,
                          global_stats: Optional[dict] = None) -> str:
    """
    从特征值中提取最显著的 top_n 个特征描述（用于控制台展示）。
    用 z-score（与全局中位数的标准差倍数）衡量偏离程度，避免大量纲特征主导排序。
    """
    FEAT_DESC = {
        # A: 趋势斜率
        "vol_slope_60d":         ("量能趋升",    "量能趋降"),
        "price_slope_60d":       ("相对走强60",   "相对走弱60"),
        "price_slope_20d":       ("相对走强20",   "相对走弱20"),
        "mf_slope_30d":          ("资金流入30",   "资金流出30"),
        "mf_slope_10d":          ("资金加速",     "资金撤出"),
        # B: 动量/比率
        "range_compress":        ("振幅收窄",     "振幅扩大"),
        "bb_width":              ("布林收窄",     "布林扩张"),
        "vol_accel":             ("量能加速",     "量能萎缩"),
        "vol_ratio_5_20":        ("短期量爆发",   "短期缩量"),
        "momentum_accel":        ("动量加速",     "动量减速"),
        "ret_5d":                ("5日强势",      "5日弱势"),
        "ret_10d":               ("10日强势",     "10日弱势"),
        "ret_20d":               ("20日强势",     "20日弱势"),
        "ret_60d":               ("60日强势",     "60日弱势"),
        "excess_ret_change":     ("超额提速",     "超额减速"),
        # C: 形态
        "high_close_ratio":      ("阳线强势",     "阴线弱势"),
        "consec_green":          ("连续上涨",     "连续下跌"),
        "vol_spike_count":       ("放量频繁",     "缩量频繁"),
        "days_above_ma20":       ("站上均线久",   "均线下方"),
        "consec_higher_low":     ("不断抬底",     "不断创低"),
        # D: 层级
        "dist_52w_high":         ("接近年高",     "距年高远"),
        "dist_60d_low_rebound":  ("60日反弹大",   "低位粘底"),
        "price_vs_ma5":          ("站上MA5",      "跌破MA5"),
        "price_vs_ma60":         ("站上MA60",     "跌破MA60"),
        "win_percent":           ("筹码盈利",     "筹码亏损"),
        "chip_concentration":    ("筹码集中",     "筹码分散"),
        "price_to_avgcost":      ("高于均成本",   "低于均成本"),
        # E: 历史强势
        "prev_tj_boards":        ("历史连板",     "无连板记录"),
        "prev_new_high_count":   ("历史多次新高", "无新高记录"),
        "prev_pool_appearances": ("历史多次入池", "首次关注"),
        # F: 估值
        "log_float_cap":         ("市值偏小",     "市值偏大"),
        "pb_hist_rank":          ("PB历史低位",   "PB历史高位"),
        "pe_ttm":                ("PE偏低",       "PE偏高"),
        "ps_ttm":                ("PS偏低",       "PS偏高"),
        # D扩展: 筹码优化
        "chip_vs_avg_cost":      ("略高于成本",   "大幅偏离成本"),
        # G: 周K线
        "weekly_vol_ratio":      ("周量加速",     "周量萎缩"),
        "weekly_vol_spike":      ("本周放量",     "本周缩量"),
        "weekly_price_pct_26w":  ("26周低位",     "26周高位"),
        "weekly_price_pct_52w":  ("52周低位",     "52周高位"),
        "weekly_ma_bull":        ("周线多头",     "周线空头"),
        "weekly_w_bottom":       ("周线W底",      "无W底形态"),
        "weekly_ma5_slope":      ("周均线上扬",   "周均线下压"),
        # H: 概念热度
        "concept_rank_20d":      ("概念强势20",   "概念偏弱20"),
        "concept_rank_60d":      ("概念强势60",   "概念偏弱60"),
        "concept_mf_trend":      ("概念资金加速", "概念资金放缓"),
    }
    desc_parts = []
    for feat in feat_cols:
        val = row.get(feat, np.nan)
        if pd.isna(val) or feat not in FEAT_DESC:
            continue
        # 用 z-score 衡量偏离程度，消除量纲差异
        if global_stats and feat in global_stats:
            med, std = global_stats[feat]
            z = abs(float(val) - med) / (std + 1e-10)
        else:
            z = abs(float(val))
        pos_desc, neg_desc = FEAT_DESC[feat]
        # 方向判断：有中位数时与中位数比较（避免对数特征永为正导致方向错误）
        if global_stats and feat in global_stats:
            med, _ = global_stats[feat]
            is_positive = float(val) >= med
        else:
            is_positive = float(val) >= 0
        desc_parts.append((z, pos_desc if is_positive else neg_desc))
    desc_parts.sort(key=lambda x: x[0], reverse=True)
    return " / ".join(d for _, d in desc_parts[:top_n]) or "—"


# ─── 主扫描流程 ───────────────────────────────────────────────────────────────

def _apply_cap_filter(
    feat_df: pd.DataFrame,
    valuation: pd.DataFrame,
    scan_date: str,
    cap_max: float = FLOAT_CAP_MAX,
) -> pd.DataFrame:
    """
    T1: 过滤掉流通市值 > cap_max 或属于稳定行业的股票。

    使用已加载的 valuation 表取扫描日前最新一条估值快照，
    过滤 float_market_cap > cap_max（默认500亿元）的股票，
    并补充过滤特定稳定行业代码。
    """
    if valuation.empty or "float_market_cap" not in valuation.columns:
        logger.warning("T1: valuation 数据不可用，跳过市值过滤")
        return feat_df

    scan_ts = pd.Timestamp(scan_date)
    # 取每只股票 <= scan_date 的最新估值快照
    val_snap = (
        valuation[valuation["date"] <= scan_ts]
        .sort_values("date")
        .groupby("instrument")
        .last()[["float_market_cap"]]
        .reset_index()
    )
    val_snap["float_market_cap"] = pd.to_numeric(
        val_snap["float_market_cap"], errors="coerce"
    )

    before_n = len(feat_df)
    # 过滤大市值
    large_cap = set(
        val_snap[val_snap["float_market_cap"] > cap_max]["instrument"].tolist()
    )
    # 过滤稳定行业代码
    stable = {
        i for i in feat_df["instrument"].tolist()
        if any(i.startswith(c) for c in _STABLE_INDUSTRY_CODES)
    }
    remove = large_cap | stable
    feat_df = feat_df[~feat_df["instrument"].isin(remove)].copy()
    removed_n = before_n - len(feat_df)
    if removed_n > 0:
        logger.info(
            f"T1 市值/行业过滤: 移除 {removed_n} 只（大市值>{cap_max/1e8:.0f}亿 {len(large_cap)}只，"
            f"稳定行业 {len(stable & remove)}只），剩余 {len(feat_df)} 只"
        )
    return feat_df


def run_scan(
    scan_date: str,
    topk: int = DEFAULT_TOPK,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_features: bool = True,
    apply_cap_filter: bool = True,
) -> pd.DataFrame:
    """
    对指定日期的全市场进行蓄力期扫描。

    Args:
        scan_date:        扫描日期（YYYY-MM-DD）
        topk:             输出 Top-K 候选股
        model_path:       模型 pkl 文件路径
        output_features:  True = 输出文件含各特征值列
        apply_cap_filter: True = 启用 T1 大市值/稳定行业后置过滤（默认开启）

    Returns:
        DataFrame，按 score 降序排列，含 instrument / score / 各特征列
    """
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("请先运行: python -m models.lgb_classifier")
        sys.exit(1)

    logger.info(f"=== 全市场蓄力期扫描  日期: {scan_date} ===")

    # 计算数据加载范围
    scan_ts = pd.Timestamp(scan_date)
    try:
        data_start_ts = offset_trading_day(scan_ts, -(DEFAULT_LOOKBACK + 10))
    except IndexError:
        data_start_ts = scan_ts - pd.Timedelta(days=280)
    data_start = str(data_start_ts.date()) if hasattr(data_start_ts, "date") else str(data_start_ts)[:10]

    # 获取全市场股票
    universe = _get_universe(scan_date)
    if not universe:
        logger.error("Unable to get universe, aborting.")
        sys.exit(1)

    # 加载市场数据
    logger.info(f"Loading market data: {data_start} ~ {scan_date}")
    kline       = load_kline(data_start, scan_date)
    chips       = load_chips(data_start, scan_date)
    moneyflow   = load_moneyflow(data_start, scan_date)
    index_kline = load_index_kline(data_start, scan_date, instruments=["000001.SH"])
    sp_hist     = _load_strong_pool_hist(data_start, scan_date)
    valuation   = load_valuation(data_start, scan_date)
    # 概念数据（H类特征，concept_bar1d 从2023年开始，往前取90天快照即可）
    concept_bar  = load_concept_bar(data_start, scan_date)
    concept_comp = load_concept_component_range(data_start, scan_date)

    if kline.empty:
        logger.error("No kline data loaded. Check database connection and date range.")
        sys.exit(1)

    # 初始化特征提取器
    extractor = PrecursorFeatureExtractor(
        kline               = kline,
        chips               = chips,
        moneyflow           = moneyflow,
        strong_pool_hist    = sp_hist,
        index_kline         = index_kline,
        valuation           = valuation,
        concept_bar         = concept_bar,
        concept_comp_range  = concept_comp,
    )

    # 提取特征（扫描日当天即为 T_feat）
    logger.info(f"Extracting features for {len(universe)} stocks on {scan_date}...")
    feat_df = extractor.compute(scan_date, universe)
    feat_df = feat_df.reset_index()
    if "index" in feat_df.columns:
        feat_df = feat_df.rename(columns={"index": "instrument"})
    feat_df["date"] = scan_date

    n_valid = feat_df[ALL_FEATURE_COLS].notna().any(axis=1).sum()
    logger.info(f"Feature extraction complete: {n_valid}/{len(feat_df)} stocks have valid features")

    # 加载模型
    logger.info(f"Loading model: {model_path}")
    model = LGBClassifier.load(model_path)

    # 读取特征列表（与训练时对齐）
    feat_cols = model.feature_cols_
    if feat_cols is None:
        feat_cols = load_feature_library() or ALL_FEATURE_COLS
    missing = [c for c in feat_cols if c not in feat_df.columns]
    if missing:
        logger.warning(f"Missing features (will be NaN): {missing}")
        for c in missing:
            feat_df[c] = np.nan

    # 模型打分
    logger.info("Scoring...")
    feat_df["score"] = model.predict_proba(feat_df)
    feat_df = feat_df.sort_values("score", ascending=False).reset_index(drop=True)

    # 计算全局特征统计（用于 z-score 标准化显示，打分后、过滤前计算，保证统计的代表性）
    global_stats = {}
    for fc in feat_cols:
        if fc in feat_df.columns:
            col = feat_df[fc].dropna()
            if len(col) > 10:
                global_stats[fc] = (float(col.median()), float(col.std() + 1e-10))

    # T1: 大市值 / 稳定行业过滤（在全市场统计后过滤，不影响 global_stats 代表性）
    if apply_cap_filter:
        feat_df = _apply_cap_filter(feat_df, valuation, scan_date)
        feat_df = feat_df.reset_index(drop=True)

    # 取 Top-K
    top_df = feat_df.head(topk).copy()
    top_df.insert(0, "rank", range(1, len(top_df) + 1))

    # 添加股票名称（若 kline 有 name 列）
    if "name" in kline.columns:
        name_map = (
            kline[kline["date"] == scan_ts][["instrument", "name"]]
            .drop_duplicates("instrument")
            .set_index("instrument")["name"]
            .to_dict()
        )
        top_df.insert(2, "name", top_df["instrument"].map(name_map).fillna("—"))

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = scan_date.replace("-", "")
    out_path = RESULTS_DIR / f"scan_{date_str}.csv"

    save_cols = (
        [c for c in ["rank", "instrument", "name", "score", "date"] if c in top_df.columns]
        + (feat_cols if output_features else [])
    )
    save_cols = [c for c in save_cols if c in top_df.columns]
    top_df[save_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Scan results saved: {out_path}")

    return top_df, global_stats


def print_scan_results(top_df: pd.DataFrame, feat_cols: List[str],
                        global_stats: Optional[dict] = None) -> None:
    """控制台打印扫描结果。"""
    print("\n" + "=" * 70)
    has_name = "name" in top_df.columns
    header = (
        f"  全市场蓄力期扫描结果  |  扫描日期: {top_df['date'].iloc[0]}  |  "
        f"展示 Top-{len(top_df)}"
    )
    print(header)
    print("=" * 70)

    col_score = "score"
    name_w = 10 if has_name else 0

    header_line = (
        f"  {'排名':<4} {'代码':<14}"
        + (f"{'名称':<{name_w}}" if has_name else "")
        + f"{'得分':>8}   主要匹配特征"
    )
    print(header_line)
    print("  " + "-" * 65)

    for _, row in top_df.iterrows():
        name_str = str(row.get("name", ""))[:name_w].ljust(name_w) if has_name else ""
        summary  = _top_feature_summary(row, feat_cols, global_stats=global_stats)
        print(
            f"  {int(row['rank']):<4} {row['instrument']:<14}"
            + name_str
            + f"{row[col_score]:>8.4f}   {summary}"
        )

    print("=" * 70 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    today = str(date.today())
    parser = argparse.ArgumentParser(description="全市场蓄力期扫描，输出 Top-K 候选股")
    parser.add_argument("--date",    default=today,
                        help=f"扫描日期（默认今天 {today}）")
    parser.add_argument("--topk",    type=int, default=DEFAULT_TOPK,
                        help=f"输出候选股数量（默认 {DEFAULT_TOPK}）")
    parser.add_argument("--model",   default=str(DEFAULT_MODEL_PATH),
                        help=f"模型文件路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--no-features", action="store_true", default=False,
                        help="输出 CSV 不含各特征列（文件更小）")
    parser.add_argument("--no-cap-filter", action="store_true", default=False,
                        help="禁用大市值/稳定行业过滤（用于对比分析）")
    args = parser.parse_args()

    top_df, global_stats = run_scan(
        scan_date        = args.date,
        topk             = args.topk,
        model_path       = Path(args.model),
        output_features  = not args.no_features,
        apply_cap_filter = not args.no_cap_filter,
    )

    model = LGBClassifier.load(Path(args.model))
    feat_cols = model.feature_cols_ or ALL_FEATURE_COLS
    print_scan_results(top_df, feat_cols, global_stats=global_stats)

    date_str = args.date.replace("-", "")
    print(f"结果已保存至: {RESULTS_DIR / f'scan_{date_str}.csv'}")


if __name__ == "__main__":
    main()
