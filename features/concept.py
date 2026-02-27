"""
概念热度特征模块 (H 类特征，3个)

核心思路：
  每只股票可能属于多个概念（300629.SZ 属于14个概念，最多的股票属于61个）。
  在 T_feat 时刻，找出该股所属概念中"最热"的那个，用其热度作为特征。

  "最热"定义：近 N 日累计涨幅在全部435个概念中的百分位排名最高。
  取最热而非平均，是因为大牛股的驱动力通常来自某一个主题概念的爆发，
  用平均值会稀释这个强信号。

特征清单：
  H1: concept_rank_20d  — 该股最热概念的20日涨幅在全概念中的百分位（0~1）
  H2: concept_rank_60d  — 该股最热概念的60日涨幅百分位（捕捉持续热点）
  H3: concept_mf_trend  — 最热概念的成交额加速度（近5日均值 / 近20日均值，代理资金关注度）

数据来源：
  - concept_bar1d：每个概念的每日 pct_change（涨幅）和 net_amount（资金净流入）
    已验证：435个概念，2023-01-03 ~ 2026-01-30
  - concept_component：股票→概念映射（97个周期性快照，约每周一次）
    使用策略：对任意 T_feat，取 <= T_feat 的最新快照

实证验证（2026-02-27）：
  好行情(2025-12-12)：涨幅>25%的股票中，4只属于"商业航天"（Top排名概念）
  差行情(2024-12-10)：逆势上涨的股票中，3只属于"人形机器人"（当期最强主题）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ─── 特征名常量 ────────────────────────────────────────────────────────────────

CONCEPT_RANK_20D  = "concept_rank_20d"   # H1: 最热概念20日涨幅百分位
CONCEPT_RANK_60D  = "concept_rank_60d"   # H2: 最热概念60日涨幅百分位
CONCEPT_MF_TREND  = "concept_mf_trend"   # H3: 最热概念资金流趋势(5d/20d均值比)

CONCEPT_FEATURE_COLS: List[str] = [
    CONCEPT_RANK_20D,
    CONCEPT_RANK_60D,
    CONCEPT_MF_TREND,
]

# 过滤掉成员数过多的泛概念（如"融资融券"、"沪深300"等，无主题意义）
# 成员数 > MAX_CONCEPT_SIZE 的概念跳过
MAX_CONCEPT_SIZE = 300


# ─── 主接口 ────────────────────────────────────────────────────────────────────

def compute_concept_features(
    concept_bar: pd.DataFrame,
    concept_comp_range: pd.DataFrame,
    feat_date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    计算 H 类概念热度特征。

    Args:
        concept_bar:        concept_bar1d 数据，列：date/concept_code/pct_change/net_amount
        concept_comp_range: concept_component 全量快照，列：date/concept_code/instrument
                            调用方应加载覆盖足够历史的快照（建议 feat_date 前 90 天到 feat_date）
        feat_date:          特征提取日期（严格使用 <= feat_date 数据，防泄漏）
        instruments:        目标股票列表

    Returns:
        DataFrame，index=instrument，columns=CONCEPT_FEATURE_COLS
        数据不足时返回 NaN（不阻断流程）
    """
    feat_ts = pd.Timestamp(feat_date)
    result = pd.DataFrame(
        index=instruments, columns=CONCEPT_FEATURE_COLS, dtype=float
    )

    if concept_bar.empty or concept_comp_range.empty:
        logger.debug("concept_bar 或 concept_comp_range 为空，H类特征全部置NaN")
        return result

    # ── Step 1: 取 <= feat_ts 的最新概念成分快照 ─────────────────────────────
    comp = _get_latest_snapshot(concept_comp_range, feat_ts)
    if comp.empty:
        logger.warning(f"H类: 无 <= {feat_date} 的 concept_component 快照")
        return result

    # ── Step 2: 过滤泛概念（成员数过多的概念无主题意义）────────────────────────
    concept_size = comp.groupby("concept_code")["instrument"].count()
    valid_concepts = concept_size[concept_size <= MAX_CONCEPT_SIZE].index
    comp = comp[comp["concept_code"].isin(valid_concepts)]

    # ── Step 3: 构建股票→概念列表的映射（只保留目标股票）──────────────────────
    inst_set = set(instruments)
    inst_to_concepts: Dict[str, List[str]] = (
        comp[comp["instrument"].isin(inst_set)]
        .groupby("instrument")["concept_code"]
        .apply(list)
        .to_dict()
    )
    if not inst_to_concepts:
        logger.warning("H类: 无目标股票的概念映射数据")
        return result

    all_concept_codes = set(comp["concept_code"].unique())

    # ── Step 4: 计算每个概念的 20日/60日 累计涨幅 + 资金流趋势 ─────────────────
    concept_bar = concept_bar.copy()
    concept_bar["date"] = pd.to_datetime(concept_bar["date"])
    hist = concept_bar[
        (concept_bar["date"] <= feat_ts) &
        (concept_bar["concept_code"].isin(all_concept_codes))
    ].copy()

    if hist.empty:
        return result

    concept_stats = _calc_concept_stats(hist, feat_ts)

    if not concept_stats:
        return result

    # ── Step 5: 计算全概念横截面百分位排名 ──────────────────────────────────
    ret20_vals = {c: s["ret_20d"] for c, s in concept_stats.items()
                  if not np.isnan(s["ret_20d"])}
    ret60_vals = {c: s["ret_60d"] for c, s in concept_stats.items()
                  if not np.isnan(s["ret_60d"])}

    rank20 = _cross_section_rank(ret20_vals)
    rank60 = _cross_section_rank(ret60_vals)

    # ── Step 6: 每只股票取所属概念中的最高排名 ───────────────────────────────
    for inst in instruments:
        concepts = inst_to_concepts.get(inst, [])
        if not concepts:
            continue

        best_rank20 = np.nan
        best_rank60 = np.nan
        best_mf     = np.nan

        for code in concepts:
            r20 = rank20.get(code, np.nan)
            r60 = rank60.get(code, np.nan)

            # 用 20日排名作为选取"最热概念"的主依据
            if not np.isnan(r20) and (np.isnan(best_rank20) or r20 > best_rank20):
                best_rank20 = r20
                # 同时取这个概念的 60日排名 和 资金流趋势
                best_rank60 = rank60.get(code, np.nan)
                mf = concept_stats.get(code, {})
                best_mf = mf.get("mf_trend", np.nan)

        result.loc[inst, CONCEPT_RANK_20D] = best_rank20
        result.loc[inst, CONCEPT_RANK_60D] = best_rank60
        result.loc[inst, CONCEPT_MF_TREND] = best_mf

    n_valid = result[CONCEPT_RANK_20D].notna().sum()
    logger.debug(
        f"H类概念特征: {n_valid}/{len(instruments)} 只有效  feat_date={feat_date}"
    )
    return result


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _get_latest_snapshot(
    comp_range: pd.DataFrame,
    feat_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    从周期性快照中取 <= feat_ts 的最新一条。
    concept_component 约每周更新一次，这里找最近的那个快照日期。
    """
    comp_range = comp_range.copy()
    comp_range["date"] = pd.to_datetime(comp_range["date"])
    valid = comp_range[comp_range["date"] <= feat_ts]
    if valid.empty:
        return pd.DataFrame()
    latest_snap = valid["date"].max()
    return valid[valid["date"] == latest_snap][["instrument", "concept_code"]].copy()


def _calc_concept_stats(
    hist: pd.DataFrame,
    feat_ts: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    """
    对每个概念，计算：
    - ret_20d: 近20日 pct_change 累加（百分比形式，如 5.2 表示 +5.2%）
    - ret_60d: 近60日 pct_change 累加
    - mf_trend: 近5日 net_amount 均值 / 近20日 net_amount 均值（资金流加速度）

    注：pct_change 字段在 concept_bar1d 中存储为小数（如 0.015 = 1.5%），
    乘以100转为百分比，方便理解。
    """
    result: Dict[str, Dict[str, float]] = {}

    for code, gdf in hist.groupby("concept_code", sort=False):
        gdf = gdf.sort_values("date")
        last20 = gdf[gdf["date"] <= feat_ts].tail(20)
        last60 = gdf[gdf["date"] <= feat_ts].tail(60)

        stats: Dict[str, float] = {}

        # 累计涨幅（近似：逐日叠加，转为百分比）
        if "pct_change" in gdf.columns:
            if len(last20) >= 5:
                stats["ret_20d"] = float(last20["pct_change"].sum() * 100)
            else:
                stats["ret_20d"] = np.nan

            if len(last60) >= 20:
                stats["ret_60d"] = float(last60["pct_change"].sum() * 100)
            else:
                stats["ret_60d"] = np.nan
        else:
            stats["ret_20d"] = np.nan
            stats["ret_60d"] = np.nan

        # 成交额趋势：近5日均值 / 近20日均值（>1 表示资金关注度加速）
        # 注：concept_bar1d 无主力净流入字段，用总成交额(trade_amount)作为概念热度代理
        amt_col = "trade_amount" if "trade_amount" in gdf.columns else None
        if amt_col:
            last5_mf  = gdf[gdf["date"] <= feat_ts].tail(5)[amt_col]
            last20_mf = gdf[gdf["date"] <= feat_ts].tail(20)[amt_col]
            avg5  = float(last5_mf.mean())  if len(last5_mf)  >= 3 else np.nan
            avg20 = float(last20_mf.mean()) if len(last20_mf) >= 10 else np.nan
            if not any(np.isnan([avg5, avg20])) and avg20 > 1e8:
                stats["mf_trend"] = avg5 / avg20
            else:
                stats["mf_trend"] = np.nan
        else:
            stats["mf_trend"] = np.nan

        result[code] = stats

    return result


def _cross_section_rank(val_dict: Dict[str, float]) -> Dict[str, float]:
    """
    将概念→数值的字典转为 0~1 的横截面百分位排名。
    值越大排名越高（1.0=最强）。
    """
    if not val_dict:
        return {}
    codes = list(val_dict.keys())
    vals  = np.array([val_dict[c] for c in codes], dtype=float)
    # 百分位排名（0~1，越大越热）
    ranks = (vals.argsort().argsort() + 1) / len(vals)
    return dict(zip(codes, ranks.tolist()))
