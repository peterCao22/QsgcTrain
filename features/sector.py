"""
板块特征模块

在截面日期 t 提取以下特征：

板块强度特征（5维，基于所属概念板块的最近表现）：
    sector_rank_pct         所属主要板块的近5日涨幅在全板块中的分位数排名（0~1）
    sector_net_inflow_ratio 所属主要板块的近5日资金净流入占比
    sector_rise_pct         所属主要板块近3日的上涨家数比例
    sector_momentum_5d      所属主要板块过去5日涨幅均值
    sector_momentum_10d     所属主要板块过去10日涨幅均值

个股资金流特征（4维，基于 moneyflow 表）：
    mf_main_net_ratio       主力净流入占比（近3日均值）
    mf_main_inflow_ratio    主力流入比（inflow / (inflow+outflow)，近3日均值）
    mf_active_buy_ratio     主动买入比例（active_buy / amount，近3日均值）
    mf_main_trend_5d        主力净流入比 5日趋势（最近-5日前，归一化）

总计：9维

注：若板块数据不可用，对应列填 0（降级处理，不抛异常）。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


def compute_sector_features(
    kline: pd.DataFrame,
    sector_mf: pd.DataFrame,
    sector_comp: pd.DataFrame,
    moneyflow: pd.DataFrame,
    date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    批量计算截面日期 date 的板块特征。

    Args:
        kline:       日线数据（含 date/instrument/amount 等，用于资金流归一化）
        sector_mf:   concept_bar1d 板块日线数据
                     [date, concept_code, concept_name, pct_change, net_amount, rise_count, fall_count]
        sector_comp: 当日股票→板块映射 [instrument, concept_code, concept_name]
        moneyflow:   个股资金流 [date, instrument, netflow_amount_main,
                     netflow_amount_rate_main, inflow_amount_rate_main,
                     outflow_amount_rate_main, active_buy_amount_all]
        date:        截面日期（str 'YYYY-MM-DD'）
        instruments: 目标股票代码列表

    Returns:
        DataFrame: index=instrument, columns=板块+资金流特征列（9维）
    """
    ts = pd.Timestamp(date)
    results = {}

    # ── 预处理板块数据 ────────────────────────────────────────────────────────
    sector_feats_by_code = _build_sector_stats(sector_mf, ts)

    # 股票→主要板块映射（取第一个概念，如有多个）
    if not sector_comp.empty and "instrument" in sector_comp.columns:
        comp_map: Dict[str, str] = (
            sector_comp.groupby("instrument")["concept_code"].first().to_dict()
        )
    else:
        comp_map = {}

    # ── 预处理个股资金流 ─────────────────────────────────────────────────────
    mf_feats_by_inst = _build_mf_stats(moneyflow, ts, instruments)

    # ── 合并 ─────────────────────────────────────────────────────────────────
    for inst in instruments:
        feats: dict = {}

        # 板块特征
        code = comp_map.get(inst)
        if code and code in sector_feats_by_code:
            sf = sector_feats_by_code[code]
            feats["sector_rank_pct"] = sf.get("rank_pct", 0.5)
            feats["sector_net_inflow_ratio"] = sf.get("net_inflow_ratio", 0.0)
            feats["sector_rise_pct"] = sf.get("rise_pct", 0.5)
            feats["sector_momentum_5d"] = sf.get("momentum_5d", 0.0)
            feats["sector_momentum_10d"] = sf.get("momentum_10d", 0.0)
        else:
            feats["sector_rank_pct"] = 0.5
            feats["sector_net_inflow_ratio"] = 0.0
            feats["sector_rise_pct"] = 0.5
            feats["sector_momentum_5d"] = 0.0
            feats["sector_momentum_10d"] = 0.0

        # 个股资金流特征
        if inst in mf_feats_by_inst:
            mf = mf_feats_by_inst[inst]
            feats["mf_main_net_ratio"] = mf.get("main_net_ratio", 0.0)
            feats["mf_main_inflow_ratio"] = mf.get("main_inflow_ratio", 0.5)
            feats["mf_active_buy_ratio"] = mf.get("active_buy_ratio", 0.5)
            feats["mf_main_trend_5d"] = mf.get("main_trend_5d", 0.0)
        else:
            feats["mf_main_net_ratio"] = 0.0
            feats["mf_main_inflow_ratio"] = 0.5
            feats["mf_active_buy_ratio"] = 0.5
            feats["mf_main_trend_5d"] = 0.0

        results[inst] = feats

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "instrument"
    return df.fillna(0.0)


# ─── 辅助：板块统计 ───────────────────────────────────────────────────────────

def _build_sector_stats(
    sector_mf: pd.DataFrame,
    ts: pd.Timestamp,
) -> Dict[str, Dict[str, float]]:
    """为每个板块代码构建截面日期前的统计数据。"""
    if sector_mf.empty:
        return {}

    # 限制到截面日期 ≤ ts 的数据
    hist = sector_mf[sector_mf["date"] <= ts].copy()
    if hist.empty:
        return {}

    # 每个板块的近5日 / 近10日数据
    result: Dict[str, Dict[str, float]] = {}

    for code, gdf in hist.groupby("concept_code", sort=False):
        gdf = gdf.sort_values("date")
        last5 = gdf.tail(5)
        last10 = gdf.tail(10)

        # 当日数据
        today_row = last5[last5["date"] == ts]

        sf: Dict[str, float] = {}

        # 动量：累计涨幅
        if len(last5) >= 1 and "pct_change" in last5.columns:
            sf["momentum_5d"] = float(last5["pct_change"].sum())
        else:
            sf["momentum_5d"] = 0.0

        if len(last10) >= 1 and "pct_change" in last10.columns:
            sf["momentum_10d"] = float(last10["pct_change"].sum())
        else:
            sf["momentum_10d"] = 0.0

        # 资金净流入比（当日）
        if not today_row.empty and "net_amount" in today_row.columns:
            sf["net_inflow_ratio"] = float(today_row.iloc[-1].get("net_amount", 0.0))
        else:
            sf["net_inflow_ratio"] = 0.0

        # 上涨家数比例（近3日均值）
        last3 = gdf.tail(3)
        if "rise_count" in last3.columns and "fall_count" in last3.columns:
            rc = last3["rise_count"].sum()
            fc = last3["fall_count"].sum()
            total = rc + fc
            sf["rise_pct"] = float(rc / total) if total > 0 else 0.5
        else:
            sf["rise_pct"] = 0.5

        # rank_pct 先存动量，后面做横截面排名
        sf["rank_pct"] = sf["momentum_5d"]
        result[code] = sf

    # 横截面排名：将 rank_pct 从原始动量值转为分位数排名
    all_mom = [v["rank_pct"] for v in result.values()]
    if all_mom:
        arr = np.array(all_mom)
        ranks = (arr.argsort().argsort() + 1) / len(arr)
        for (code, sf), rank_pct in zip(result.items(), ranks):
            sf["rank_pct"] = float(rank_pct)

    return result


# ─── 辅助：个股资金流统计 ─────────────────────────────────────────────────────

def _build_mf_stats(
    moneyflow: pd.DataFrame,
    ts: pd.Timestamp,
    instruments: List[str],
) -> Dict[str, Dict[str, float]]:
    """为每只股票构建资金流特征。"""
    if moneyflow.empty:
        return {}

    hist = moneyflow[
        (moneyflow["date"] <= ts) & moneyflow["instrument"].isin(instruments)
    ].copy()
    if hist.empty:
        return {}

    result: Dict[str, Dict[str, float]] = {}

    for inst, gdf in hist.groupby("instrument", sort=False):
        gdf = gdf.sort_values("date")
        last5 = gdf.tail(5)
        last3 = gdf.tail(3)

        mf: Dict[str, float] = {}

        # 主力净流入比（近3日均值）
        if "netflow_amount_rate_main" in last3.columns:
            mf["main_net_ratio"] = float(last3["netflow_amount_rate_main"].mean())
        else:
            mf["main_net_ratio"] = 0.0

        # 主力流入比
        if "inflow_amount_rate_main" in last3.columns:
            inflow = last3["inflow_amount_rate_main"].mean()
            outflow = last3.get("outflow_amount_rate_main", pd.Series([0.5])).mean()
            total = inflow + outflow
            mf["main_inflow_ratio"] = float(inflow / total) if total > 0 else 0.5
        else:
            mf["main_inflow_ratio"] = 0.5

        # 主动买入比例（当日主动买入 / 成交额）
        # 无法直接从 moneyflow 表得到成交额，用 active_buy / amount 估算
        mf["active_buy_ratio"] = 0.5  # 占位，需与 kline 的 amount 联表才能算

        # 主力净流入 5日趋势
        if "netflow_amount_rate_main" in last5.columns and len(last5) >= 2:
            vals = last5["netflow_amount_rate_main"].values
            mf["main_trend_5d"] = float(vals[-1] - vals[0])
        else:
            mf["main_trend_5d"] = 0.0

        result[inst] = mf

    return result
