"""
筹码特征模块

基于 chips_all 表（avg_cost / win_percent / concentration）
在截面日期 t 提取以下特征：

静态筹码特征（3维）：
    chip_avg_cost_ratio     当前价 / 平均持仓成本
    chip_win_percent        当前盈利筹码比例（0~1）
    chip_concentration      筹码集中度（0~1）

动态趋势特征（3维）：
    chip_cost_trend_5d      过去5日平均成本变化斜率（归一化）
    chip_win_change_5d      过去5日盈利比例变化量
    chip_conc_change_5d     过去5日集中度变化量

综合质量分（1维）：
    chip_quality_score      加权综合：win*0.4 + conc*0.3 + (1-|cost_ratio-1|)*0.3

总计：7维
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_chip_features(
    chips: pd.DataFrame,
    kline: pd.DataFrame,
    date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    批量计算截面日期 date 的筹码特征。

    Args:
        chips:       chips_all 数据，含 [date, instrument, avg_cost,
                     win_percent, concentration]，时间范围覆盖 date 前 ~10日。
        kline:       日线数据（用于获取当日收盘价），含 [date, instrument, close]。
        date:        截面日期（str 'YYYY-MM-DD'）。
        instruments: 目标股票代码列表。

    Returns:
        DataFrame: index=instrument, columns=筹码特征列（7维）
    """
    ts = pd.Timestamp(date)
    chips = chips[chips["instrument"].isin(instruments)].copy()
    kline_t = kline[
        (kline["date"] == ts) & kline["instrument"].isin(instruments)
    ][["instrument", "close"]].set_index("instrument")

    results = {}

    for inst, gdf in chips.groupby("instrument", sort=False):
        gdf = gdf.sort_values("date").reset_index(drop=True)

        # 当日筹码
        today_rows = gdf[gdf["date"] == ts]
        if today_rows.empty:
            continue
        today = today_rows.iloc[-1]

        avg_cost = today["avg_cost"]
        win_pct = today["win_percent"]
        conc = today["concentration"]

        # 收盘价
        close = kline_t.loc[inst, "close"] if inst in kline_t.index else np.nan
        if np.isnan(close) or close <= 0:
            continue

        feats: dict = {}

        # ── 静态特征 ─────────────────────────────────────────────────────
        feats["chip_avg_cost_ratio"] = float(close / avg_cost) if avg_cost > 0 else 1.0
        feats["chip_win_percent"] = float(win_pct) if not np.isnan(win_pct) else 0.5
        feats["chip_concentration"] = float(conc) if not np.isnan(conc) else 0.5

        # ── 动态趋势（过去5日）────────────────────────────────────────────
        past5 = gdf[gdf["date"] <= ts].tail(5)

        if len(past5) >= 2:
            cost_arr = past5["avg_cost"].values
            x = np.arange(len(cost_arr), dtype=float)
            slope = np.polyfit(x, cost_arr, 1)[0]
            feats["chip_cost_trend_5d"] = float(slope / (cost_arr.mean() + 1e-9))

            win_arr = past5["win_percent"].values
            feats["chip_win_change_5d"] = float(win_arr[-1] - win_arr[0])

            conc_arr = past5["concentration"].values
            feats["chip_conc_change_5d"] = float(conc_arr[-1] - conc_arr[0])
        else:
            feats["chip_cost_trend_5d"] = 0.0
            feats["chip_win_change_5d"] = 0.0
            feats["chip_conc_change_5d"] = 0.0

        # ── 综合质量分 ────────────────────────────────────────────────────
        cost_ratio_score = max(0.0, 1.0 - abs(feats["chip_avg_cost_ratio"] - 1.0))
        feats["chip_quality_score"] = (
            feats["chip_win_percent"] * 0.4
            + feats["chip_concentration"] * 0.3
            + cost_ratio_score * 0.3
        )

        results[inst] = feats

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "instrument"
    return df.fillna(0.0)
