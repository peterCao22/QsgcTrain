"""
生成训练数据集

从 strong_pool 历史记录生成带标签的训练样本。

核心逻辑（画像匹配，非前向预测）：
  目标：学习强势股"入池前 N 个交易日"的特征画像，以便在全市场扫描中
        找出当前状态与此画像最相似的股票。

  对每个历史日期 T（强势股池快照日）：
    T_feat = T - PRECURSOR_N 个交易日（特征提取时间点，约为入池前1~3个月）

  正样本：T 日强势池中的所有股票（不限"新入池"），提取 T_feat 时的特征。
  负样本：分两层
    - 硬负样本（60%）：与正样本同板块（concept_component）、但近20日涨幅
                       排在板块内后 30% 的股票（真正的"同类弱势对比"）
    - 软负样本（40%）：从全市场可交易股中随机补充（排除强势池）
  标签：
    is_strong_pos   1 = 正样本（在强势池），0 = 负样本（主训练目标）
    fwd_return_30d  从 T_feat 起 30 日收盘收益率（连续标签，辅助分析）
    is_top20pct     同截面 fwd_return_30d 前 20% 分位 = 1（辅助标签）

输出：
  data/training_dataset.parquet   完整训练集
  data/training_stats.json        统计摘要

运行方式：
    conda activate rqsdk
    python scripts/build_training_data.py

    # 指定日期范围
    python scripts/build_training_data.py --start 2025-02-01 --end 2025-12-01

    # 调整参数
    python scripts/build_training_data.py --precursor-n 30 --neg-ratio 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config import (
    DATA_DIR,
    EXCLUDE_PREFIXES,
    LABEL_COL,
    LABEL_FORWARD_DAYS,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from data.db_loader import get_trading_days, offset_trading_day, read_sql, read_sql_chunked
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

OUTPUT_PARQUET = DATA_DIR / "training_dataset.parquet"
OUTPUT_STATS   = DATA_DIR / "training_stats.json"

# ─── 参数默认值 ────────────────────────────────────────────────────────────────

DEFAULT_PRECURSOR_N    = 20    # T_feat = T_entry - 20 交易日（蓄力期）
DEFAULT_NEG_RATIO      = 5.0   # 负样本:正样本 = 5:1
DEFAULT_KLINE_LOOKBACK = 160   # 特征计算所需的最大回看天数
DEFAULT_MONTH_SAMPLES  = 100   # 每月最多采样正样本数（新鲜入池股票）
FRESH_ENTRY_LOOKBACK   = 30    # 股票在过去 N 交易日内未出现在强势池才算"新鲜入池"

# 负样本过滤：在 T_feat 前必须至少有 N 个交易日的 kline 记录
MIN_HIST_DAYS = 60

# Top-K% 分位二分类标签阈值（同截面 fwd_return_20d 前 20% = 1，主训练目标）
TOPK_QUANTILE = 0.80

# 硬负样本：同板块弱势股抽样参数
HARD_NEG_FRAC         = 0.5   # 总负样本中同板块硬负样本目标占比
HARD_NEG_BOTTOM_PCT   = 0.30  # 取板块内近20日涨幅后 30% 作为硬负样本候选
HARD_NEG_MAX_CONCEPTS = 3     # 每只正样本最多参考前 N 个概念板块
HARD_NEG_LOOKBACK     = 20    # 计算"弱势"所用的近 N 个交易日涨幅

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_strong_pool(start_date: str, end_date: str) -> pd.DataFrame:
    """加载指定日期范围内的强势股池数据。"""
    sql = """
        SELECT date, instrument, tj_boards, new_high
        FROM strong_pool
        WHERE date >= :start AND date <= :end
        ORDER BY date, instrument
    """
    df = read_sql(sql, {"start": start_date, "end": end_date})
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"strong_pool loaded: {len(df):,} rows [{start_date} ~ {end_date}]")
    return df


def load_universe(date: str) -> list:
    """加载指定日期的可交易股票（排除科创/北交所/ST/停牌）。"""
    from data.db_loader import get_tradable_universe
    instruments = get_tradable_universe(date)
    # 排除科创/北交所前缀
    instruments = [
        i for i in instruments
        if not any(i.startswith(p) for p in EXCLUDE_PREFIXES)
    ]
    return instruments


def compute_forward_returns_batch(
    instruments: list,
    feat_date: pd.Timestamp,
    kline: pd.DataFrame,
    forward_days: int = LABEL_FORWARD_DAYS,
) -> dict:
    """
    向量化批量计算一组股票的前向收益率（避免逐只股票循环）。

    Args:
        instruments: 股票列表
        feat_date:   特征提取日期（T_feat）
        kline:       全量日线数据（宽窗口）
        forward_days: 前向天数

    Returns:
        {instrument: fwd_return} 字典，无法计算的股票值为 NaN
    """
    try:
        t_plus_n = offset_trading_day(feat_date, forward_days)
    except IndexError:
        return {inst: np.nan for inst in instruments}

    inst_set = set(instruments)
    sub = kline[kline["instrument"].isin(inst_set)]

    # T 日收盘
    close_t = (
        sub[sub["date"] == feat_date]
        .set_index("instrument")["close"]
    )
    # T+N 日收盘
    close_tn = (
        sub[sub["date"] == t_plus_n]
        .set_index("instrument")["close"]
    )

    merged = close_t.rename("close_t").to_frame().join(
        close_tn.rename("close_tn"), how="inner"
    )
    merged = merged[(merged["close_t"] > 0) & (merged["close_tn"] > 0)]
    merged["ret"] = merged["close_tn"] / merged["close_t"] - 1
    # 过滤异常值（超过 ±200%）
    merged.loc[merged["ret"].abs() > 2.0, "ret"] = np.nan

    result = merged["ret"].to_dict()
    # 补全没有数据的股票
    for inst in instruments:
        if inst not in result:
            result[inst] = np.nan
    return result


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def _build_concept_cache(concept_df: pd.DataFrame):
    """
    将 concept_component 宽表构建为两个快速查找结构。

    Returns:
        cache      : {snapshot_date: (code_to_insts, inst_to_codes)}
                     code_to_insts : {concept_code: set(instruments)}
                     inst_to_codes : {instrument: [concept_codes]}（按权重无序）
        snap_dates : sorted list of pd.Timestamp，所有快照日期
    """
    if concept_df.empty:
        return {}, []

    cache = {}
    snap_dates = sorted(concept_df["date"].unique())
    for snap_date, grp in concept_df.groupby("date"):
        code_to_insts = grp.groupby("concept_code")["instrument"].apply(set).to_dict()
        inst_to_codes = grp.groupby("instrument")["concept_code"].apply(list).to_dict()
        cache[snap_date] = (code_to_insts, inst_to_codes)
    return cache, snap_dates


def _get_latest_snapshot(snap_dates: list, t_feat_ts: pd.Timestamp):
    """返回 snap_dates 中最近一个 <= t_feat_ts 的快照日期，不存在则返回 None。"""
    import bisect
    idx = bisect.bisect_right(snap_dates, t_feat_ts) - 1
    return snap_dates[idx] if idx >= 0 else None


def _get_hard_neg_candidates(
    pos_instruments: list,
    t_feat_ts: pd.Timestamp,
    returns_20d: pd.Series,
    concept_cache: dict,
    snap_dates: list,
    strong_pool_set: set,
    valid_at_feat: set,
    exclude_prefixes: list,
) -> list:
    """
    从同板块中筛选近 20 日涨幅排名后 30% 的弱势股作为硬负样本候选。

    Args:
        pos_instruments : 当前截面的正样本股票列表
        t_feat_ts       : 特征提取日期
        returns_20d     : 截面日 t_feat_ts 的全市场近20日涨幅 Series（index=instrument）
        concept_cache   : _build_concept_cache 返回的 cache
        snap_dates      : _build_concept_cache 返回的 snap_dates
        strong_pool_set : 当天强势池所有股票（正样本+老成员），排除用
        valid_at_feat   : 在 t_feat 时有足够 kline 历史的股票集合
        exclude_prefixes: 排除前缀列表（科创/北交所）

    Returns:
        硬负样本候选股票列表（未经数量抽样）
    """
    snap_date = _get_latest_snapshot(snap_dates, t_feat_ts)
    if snap_date is None or returns_20d.empty:
        return []

    code_to_insts, inst_to_codes = concept_cache[snap_date]

    # 收集所有正样本所属板块的成员
    sector_pool: set = set()
    for p in pos_instruments:
        codes = inst_to_codes.get(p, [])[:HARD_NEG_MAX_CONCEPTS]
        for code in codes:
            sector_pool |= code_to_insts.get(code, set())

    # 过滤：排除强势池、科创/北交所、历史不足、正样本自身
    pos_set = set(pos_instruments)
    sector_pool = {
        i for i in sector_pool
        if i not in strong_pool_set
        and i not in pos_set
        and not any(i.startswith(p) for p in exclude_prefixes)
        and i in valid_at_feat
    }

    if len(sector_pool) < 10:
        return list(sector_pool)

    # 按近20日涨幅排名，取后 HARD_NEG_BOTTOM_PCT
    sector_rets = returns_20d.reindex(list(sector_pool)).dropna()
    if len(sector_rets) < 5:
        return list(sector_pool)

    threshold = sector_rets.quantile(HARD_NEG_BOTTOM_PCT)
    return sector_rets[sector_rets <= threshold].index.tolist()


def _select_dates_monthly(
    sp_df: pd.DataFrame,
    month_samples: int,
    rng: np.random.Generator,
) -> list:
    """
    按月分批选取强势股池日期。

    每个自然月内，从该月所有有强势股池数据的交易日中随机采样，
    每月最多保留 month_samples 个正样本对应的日期。
    若当月新入池股票总数 <= month_samples，则保留该月全部日期。

    Returns:
        筛选后的 T 日期列表（pd.Timestamp）
    """
    sp_df = sp_df.copy()
    sp_df["ym"] = sp_df["date"].dt.to_period("M")
    selected: list = []
    for ym, grp in sp_df.groupby("ym"):
        month_dates = sorted(grp["date"].unique())
        if len(month_dates) <= month_samples:
            selected.extend(month_dates)
        else:
            chosen = rng.choice(month_dates, size=month_samples, replace=False)
            selected.extend(sorted(chosen))
    return sorted(set(selected))


def _get_fresh_entries(
    sp_df: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    all_trading_days: list,
    lookback_days: int = FRESH_ENTRY_LOOKBACK,
) -> pd.DataFrame:
    """
    找出在 [month_start, month_end] 内"新鲜入池"的股票及其首次出现日期。

    "新鲜" = 在 month_start 前 lookback_days 个交易日内未在强势池中出现过，
    这样可以过滤掉长期滞留池内的股票，只保留真正的新启动信号。

    Returns:
        DataFrame，列：instrument, t_entry（首次入池日期）
    """
    # 确定回看窗口起点
    try:
        lookback_start = offset_trading_day(month_start, -lookback_days)
    except IndexError:
        lookback_start = month_start - pd.Timedelta(days=lookback_days * 2)

    # 回看窗口内出现过的股票（已在池，不算新鲜）
    prev_pool = set(
        sp_df[(sp_df["date"] >= lookback_start) & (sp_df["date"] < month_start)]["instrument"]
    )

    # 本月强势池
    month_df = sp_df[(sp_df["date"] >= month_start) & (sp_df["date"] <= month_end)]
    if month_df.empty:
        return pd.DataFrame(columns=["instrument", "t_entry"])

    # 每只股票本月首次出现日期
    first_dates = (
        month_df.groupby("instrument")["date"].min()
        .reset_index()
        .rename(columns={"date": "t_entry"})
    )

    # 只保留"新鲜"入池的股票
    fresh = first_dates[~first_dates["instrument"].isin(prev_pool)].copy()
    return fresh


def build_training_data(
    start_date: str,
    end_date: str,
    precursor_n: int = DEFAULT_PRECURSOR_N,
    neg_ratio: float = DEFAULT_NEG_RATIO,
    month_samples: int = DEFAULT_MONTH_SAMPLES,
) -> pd.DataFrame:
    """
    构建训练数据集（新架构：首次入池正样本 + fwd_return_20d 标签）。

    核心逻辑：
      1. 按月分批，找出每月"新鲜入池"股票（T_entry = 首次入池日）
      2. T_feat = T_entry - precursor_n 个交易日（蓄力期特征提取点）
      3. 正样本：该股票在 T_feat 时的特征
      4. 负样本：同 T_feat 截面，随机+同板块弱势股（不含强势池）
      5. 标签：fwd_return_20d = close[T_entry] / close[T_feat] - 1
              is_top20pct = 同截面前 20% = 1（主训练目标，二分类）
              is_strong_pos = 1（正样本），0（负样本）（辅助标签）

    Returns:
        DataFrame 含列：date, instrument, is_strong_pos, fwd_return_20d,
                        is_top20pct, <feature_cols>
    """
    logger.info(f"Building training data (新架构): {start_date} ~ {end_date}  "
                f"precursor_n={precursor_n}  neg_ratio={neg_ratio}  "
                f"month_samples={month_samples}")

    cal = get_trading_days(start_date, end_date)
    if len(cal) == 0:
        logger.error("No trading days found in given range")
        return pd.DataFrame()

    sp_hist_start = str((pd.Timestamp(start_date) - pd.Timedelta(days=60)).date())
    sp_df_full = load_strong_pool(sp_hist_start, end_date)
    sp_df = sp_df_full[sp_df_full["date"] >= pd.Timestamp(start_date)].copy()
    if sp_df.empty:
        logger.error("No strong_pool data found.")
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    months = sp_df["date"].dt.to_period("M").unique()
    logger.info(f"Processing {len(months)} months, up to {month_samples} pos samples/month")

    pos_records = []
    for ym in months:
        month_start = ym.to_timestamp()
        month_end   = (ym + 1).to_timestamp() - pd.Timedelta(days=1)
        fresh = _get_fresh_entries(sp_df_full, month_start, month_end, cal)
        if fresh.empty:
            continue
        if len(fresh) > month_samples:
            fresh = fresh.sample(n=month_samples, random_state=42)
        for _, row in fresh.iterrows():
            inst    = row["instrument"]
            t_entry = pd.Timestamp(row["t_entry"])
            if any(inst.startswith(p) for p in EXCLUDE_PREFIXES):
                continue
            try:
                t_feat_ts = offset_trading_day(t_entry, -precursor_n)
            except IndexError:
                continue
            pos_records.append((t_feat_ts, inst, t_entry))

    if not pos_records:
        logger.error("No fresh entries found.")
        return pd.DataFrame()
    logger.info(f"Collected {len(pos_records)} positive candidates")

    all_t_feats   = [r[0] for r in pos_records]
    all_t_entries = [r[2] for r in pos_records]
    try:
        earliest_feat = offset_trading_day(min(all_t_feats), -DEFAULT_KLINE_LOOKBACK)
        latest_label  = max(all_t_entries) + pd.Timedelta(days=5)
    except Exception:
        earliest_feat = min(all_t_feats) - pd.Timedelta(days=250)
        latest_label  = max(all_t_entries) + pd.Timedelta(days=10)

    data_start = str(earliest_feat.date()) if hasattr(earliest_feat, "date") else str(earliest_feat)[:10]
    data_end   = str(latest_label.date())  if hasattr(latest_label,  "date") else str(latest_label)[:10]
    logger.info(f"Loading market data: {data_start} ~ {data_end}")

    kline       = load_kline(data_start, data_end)
    chips       = load_chips(data_start, data_end)
    moneyflow   = load_moneyflow(data_start, data_end)
    index_kline = load_index_kline(data_start, data_end, instruments=["000001.SH"])
    valuation   = load_valuation(data_start, data_end)

    concept_df = load_concept_component_range(data_start, data_end)
    concept_cache, snap_dates_cc = _build_concept_cache(concept_df)
    if not snap_dates_cc:
        logger.warning("concept_component unavailable, negatives will be random")

    # 加载概念K线，供 H 类特征计算（T4 概念热度）
    concept_bar = load_concept_bar(data_start, data_end)

    logger.info("Precomputing 20-day return matrix...")
    try:
        kline_close = kline.pivot_table(index="date", columns="instrument", values="close").sort_index()
        returns_20d_matrix = kline_close.pct_change(HARD_NEG_LOOKBACK)
    except Exception as e:
        logger.warning(f"Return matrix failed ({e})")
        returns_20d_matrix = pd.DataFrame()

    extractor = PrecursorFeatureExtractor(
        kline=kline, chips=chips, moneyflow=moneyflow,
        strong_pool_hist=sp_df_full, index_kline=index_kline, valuation=valuation,
        concept_bar=concept_bar, concept_comp_range=concept_df,
    )

    logger.info(f"Precomputing kline validity (>= {MIN_HIST_DAYS} days)...")
    valid_since = {}
    for inst, grp in kline.sort_values(["instrument", "date"]).groupby("instrument"):
        dates = grp["date"].tolist()
        if len(dates) >= MIN_HIST_DAYS:
            valid_since[inst] = dates[MIN_HIST_DAYS - 1]

    def get_valid_insts_at(t_ts):
        return {i for i, s in valid_since.items() if s <= t_ts}

    from collections import defaultdict
    feat_groups = defaultdict(list)
    for t_feat_ts, inst, t_entry in pos_records:
        if inst in valid_since and valid_since[inst] <= t_feat_ts:
            feat_groups[t_feat_ts].append((inst, t_entry))

    logger.info("Precomputing universe cache...")
    universe_cache = {}
    for t_feat_ts in feat_groups:
        t_feat_str = str(t_feat_ts.date())
        if t_feat_str not in universe_cache:
            try:
                u = load_universe(t_feat_str)
            except Exception:
                u = kline[kline["date"] == t_feat_ts]["instrument"].tolist()
                u = [i for i in u if not any(i.startswith(p) for p in EXCLUDE_PREFIXES)]
            valid_at = get_valid_insts_at(t_feat_ts)
            universe_cache[t_feat_str] = [i for i in u if i in valid_at]
    logger.info(f"Universe cache: {len(universe_cache)} dates")

    all_samples = []
    for t_feat_ts, pos_list in tqdm(feat_groups.items(), desc="Extracting precursor features"):
        t_feat_str      = str(t_feat_ts.date())
        pos_instruments = [item[0] for item in pos_list]
        pos_entry_map   = {item[0]: item[1] for item in pos_list}
        pos_set         = set(pos_instruments)

        pool_today_set = set(sp_df[sp_df["date"] == t_feat_ts]["instrument"].tolist())
        try:
            t_feat_plus20   = offset_trading_day(t_feat_ts, precursor_n)
            pool_future_set = set(
                sp_df_full[
                    (sp_df_full["date"] > t_feat_ts) &
                    (sp_df_full["date"] <= t_feat_plus20)
                ]["instrument"].tolist()
            )
        except Exception:
            pool_future_set = set()
        exclude_set = pool_today_set | pool_future_set | pos_set

        n_neg_total   = int(len(pos_instruments) * neg_ratio)
        n_hard_target = int(n_neg_total * HARD_NEG_FRAC)
        valid_at_feat = get_valid_insts_at(t_feat_ts)

        ret_cross = (
            returns_20d_matrix.loc[t_feat_ts].dropna()
            if not returns_20d_matrix.empty and t_feat_ts in returns_20d_matrix.index
            else pd.Series(dtype=float)
        )
        hard_neg_pool = _get_hard_neg_candidates(
            pos_instruments=pos_instruments, t_feat_ts=t_feat_ts, returns_20d=ret_cross,
            concept_cache=concept_cache, snap_dates=snap_dates_cc,
            strong_pool_set=exclude_set, valid_at_feat=valid_at_feat,
            exclude_prefixes=EXCLUDE_PREFIXES,
        )
        universe      = universe_cache.get(t_feat_str, [])
        universe_set  = set(universe)
        hard_neg_pool = [i for i in hard_neg_pool if i in universe_set]
        n_hard        = min(len(hard_neg_pool), n_hard_target)
        hard_neg      = rng.choice(hard_neg_pool, size=n_hard, replace=False).tolist() if n_hard > 0 else []

        hard_neg_set = set(hard_neg)
        soft_pool    = [i for i in universe if i not in exclude_set and i not in hard_neg_set]
        n_soft       = min(len(soft_pool), max(0, n_neg_total - len(hard_neg)))
        soft_neg     = rng.choice(soft_pool, size=n_soft, replace=False).tolist() if n_soft > 0 else []

        neg_instruments = hard_neg + soft_neg
        all_instruments = pos_instruments + neg_instruments

        feat_df = extractor.compute(t_feat_str, all_instruments)
        if feat_df.empty:
            continue
        feat_df = feat_df.reset_index().rename(columns={"index": "instrument"})
        feat_df["date"]          = t_feat_ts
        feat_df["is_strong_pos"] = feat_df["instrument"].isin(pos_set).astype(int)

        close_sub   = kline[kline["instrument"].isin(set(all_instruments))]
        close_pivot = close_sub.pivot_table(index="date", columns="instrument", values="close").sort_index()

        def _get_close(inst, date):
            try:
                v = close_pivot.loc[date, inst]
                return float(v) if not pd.isna(v) else np.nan
            except KeyError:
                return np.nan

        try:
            t_label_generic = offset_trading_day(t_feat_ts, precursor_n)
        except Exception:
            t_label_generic = None

        fwd_rets = {}
        for inst in all_instruments:
            c_t = _get_close(inst, t_feat_ts)
            if pd.isna(c_t) or c_t <= 0:
                fwd_rets[inst] = np.nan
                continue
            c_end = (
                _get_close(inst, pos_entry_map[inst]) if inst in pos_entry_map
                else (_get_close(inst, t_label_generic) if t_label_generic is not None else np.nan)
            )
            fwd_rets[inst] = float(c_end / c_t - 1) if (not pd.isna(c_end) and c_end > 0) else np.nan

        feat_df["fwd_return_20d"] = feat_df["instrument"].map(fwd_rets)
        feat_df = feat_df.dropna(subset=["fwd_return_20d"])
        if not feat_df.empty:
            all_samples.append(feat_df)

    if not all_samples:
        logger.error("No samples generated.")
        return pd.DataFrame()

    dataset = pd.concat(all_samples, ignore_index=True)
    dataset["is_top20pct"] = (
        dataset.groupby("date")["fwd_return_20d"]
        .transform(lambda x: (x >= x.quantile(TOPK_QUANTILE)).astype(int))
    )

    meta_cols  = ["date", "instrument", "is_strong_pos", "fwd_return_20d", "is_top20pct"]
    final_cols = meta_cols + [c for c in ALL_FEATURE_COLS if c in dataset.columns]
    dataset    = dataset[[c for c in final_cols if c in dataset.columns]]

    n_pos     = int(dataset["is_strong_pos"].sum())
    n_top20   = int(dataset["is_top20pct"].sum())
    pos_top20 = int(((dataset["is_strong_pos"] == 1) & (dataset["is_top20pct"] == 1)).sum())
    logger.info(
        f"\n{'='*60}\n"
        f"Training data complete (new arch):\n"
        f"  Total:         {len(dataset):,}\n"
        f"  is_strong=1:   {n_pos:,} ({n_pos/len(dataset):.1%})\n"
        f"  is_top20pct=1: {n_top20:,} ({n_top20/len(dataset):.1%})\n"
        f"  pos->top20%:   {pos_top20}/{n_pos} ({pos_top20/max(n_pos,1):.1%})\n"
        f"  fwd_ret mean:  {dataset['fwd_return_20d'].mean()*100:+.2f}%\n"
        f"{'='*60}"
    )

    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:
    """保存训练集到 parquet，并写入统计摘要 JSON。"""
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"Dataset saved: {OUTPUT_PARQUET}")

    stats = {
        "total_samples": len(dataset),
        "is_strong_pos_1": int(dataset["is_strong_pos"].sum()),
        "is_strong_pos_0": int((dataset["is_strong_pos"] == 0).sum()),
        "is_top20pct_1": int(dataset["is_top20pct"].sum()),
        "positive_rate_strong": float(dataset["is_strong_pos"].mean()),
        "positive_rate_top20": float(dataset["is_top20pct"].mean()),
        "n_dates": int(dataset["date"].nunique()),
        "date_range": [
            str(dataset["date"].min().date()),
            str(dataset["date"].max().date()),
        ],
        "fwd_return_20d_mean":   float(dataset["fwd_return_20d"].mean()),
        "fwd_return_20d_median": float(dataset["fwd_return_20d"].median()),
        "fwd_return_20d_std":    float(dataset["fwd_return_20d"].std()),
        "feature_missing_rate": {
            c: float(dataset[c].isna().mean())
            for c in ALL_FEATURE_COLS
            if c in dataset.columns
        },
    }
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Stats saved: {OUTPUT_STATS}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成蓄力期特征训练数据集（新架构：首次入池正样本）")
    from datetime import datetime, timedelta
    one_year_ago     = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    three_months_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    parser.add_argument("--start", default=one_year_ago,
                        help=f"强势池起始日期（默认1年前：{one_year_ago}）")
    parser.add_argument("--end",   default=three_months_ago,
                        help=f"强势池结束日期（默认3个月前：{three_months_ago}）")
    parser.add_argument("--precursor-n", type=int, default=DEFAULT_PRECURSOR_N,
                        help=f"T_feat = T_entry - N 交易日（默认 {DEFAULT_PRECURSOR_N}）")
    parser.add_argument("--neg-ratio",   type=float, default=DEFAULT_NEG_RATIO,
                        help=f"负:正样本比例（默认 {DEFAULT_NEG_RATIO}）")
    parser.add_argument("--month-samples", type=int, default=DEFAULT_MONTH_SAMPLES,
                        help=f"每月最多采样正样本数（默认 {DEFAULT_MONTH_SAMPLES}）")
    args = parser.parse_args()

    logger.info(f"参数：start={args.start}  end={args.end}  "
                f"precursor_n={args.precursor_n}  neg_ratio={args.neg_ratio}  "
                f"month_samples={args.month_samples}")

    dataset = build_training_data(
        start_date    = args.start,
        end_date      = args.end,
        precursor_n   = args.precursor_n,
        neg_ratio     = args.neg_ratio,
        month_samples = args.month_samples,
    )

    if not dataset.empty:
        save_dataset(dataset)
        print(f"\n训练集已保存至：{OUTPUT_PARQUET}")
        print(f"统计摘要已保存至：{OUTPUT_STATS}")
    else:
        print("训练集生成失败，请检查数据。")


if __name__ == "__main__":
    main()
