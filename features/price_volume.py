"""
价量特征模块

对每只股票在截面日期 t 提取以下特征（全部基于 t 日收盘后可得的历史数据）：

动量特征（4窗口 × 3指标 = 12维）：
    pct_5d / pct_10d / pct_20d / pct_60d        收益率
    vol_ratio_5d / vol_ratio_10d / vol_ratio_20d / vol_ratio_60d   量比
    turn_ratio_5d / turn_ratio_10d / turn_ratio_20d / turn_ratio_60d 换手率比

波动特征（3维）：
    volatility_20d      过去20日收益率标准差（年化）
    max_drawdown_20d    过去20日最大回撤
    amplitude_avg_5d    过去5日平均振幅

均线偏离（4维）：
    close_vs_ma5 / close_vs_ma10 / close_vs_ma20 / close_vs_ma60

趋势形态（4维）：
    ma5_slope_5d        MA5斜率（过去5日归一化）
    ma_alignment        多头排列得分（MA5>MA10>MA20 各+1）
    consec_up_days      连续上涨天数（过去5日中）
    body_ratio_5d       过去5日实体比均值（(close-open)/振幅）

高位预警（2维）：
    near_52w_high       收盘价 / 近252日最高价
    below_52w_high_pct  距252日最高价回撤百分比

总计：25维
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config import MOMENTUM_WINDOWS


def compute_price_volume_features(
    kline: pd.DataFrame,
    date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    批量计算截面日期 date 的价量特征。

    Args:
        kline:  包含 [date, instrument, open, high, low, close, volume,
                       amount, turn, pctChg, ma5, ma10, ma20, ma60] 的日线 DataFrame。
                时间范围须覆盖 date 前 ~252 交易日。
        date:   截面日期（str 'YYYY-MM-DD'），结果仅包含该日期的特征。
        instruments: 目标股票代码列表（过滤范围）。

    Returns:
        DataFrame: index=instrument, columns=价量特征列（25维）
    """
    ts = pd.Timestamp(date)
    kline = kline[kline["instrument"].isin(instruments)].copy()
    kline = kline.sort_values(["instrument", "date"])

    results = {}

    for inst, gdf in kline.groupby("instrument", sort=False):
        gdf = gdf.reset_index(drop=True)
        # 找到截面日期在该股历史中的位置
        idx_list = gdf.index[gdf["date"] == ts].tolist()
        if not idx_list:
            continue
        t_idx = idx_list[0]

        row = gdf.iloc[t_idx]
        feats: dict = {}

        close_t = row["close"]
        if close_t <= 0:
            continue

        # ── 动量 & 量比 & 换手率比 ────────────────────────────────────────
        for w in MOMENTUM_WINDOWS:
            start_idx = max(0, t_idx - w + 1)
            window = gdf.iloc[start_idx : t_idx + 1]

            # 收益率（相对 w 日前收盘）
            if len(window) >= 2:
                close_start = window.iloc[0]["close"]
                feats[f"pct_{w}d"] = (close_t / close_start - 1) if close_start > 0 else 0.0
            else:
                feats[f"pct_{w}d"] = 0.0

            # 量比：近w日均量 / 近60日均量
            vol_arr = window["volume"].values
            feats[f"vol_ratio_{w}d"] = _safe_ratio(
                vol_arr.mean() if len(vol_arr) > 0 else np.nan,
                _rolling_mean(gdf["volume"].values, t_idx, 60),
            )

            # 换手率比
            turn_arr = window["turn"].values if "turn" in gdf.columns else np.array([])
            feats[f"turn_ratio_{w}d"] = _safe_ratio(
                turn_arr.mean() if len(turn_arr) > 0 else np.nan,
                _rolling_mean(gdf["turn"].values if "turn" in gdf.columns else np.zeros(len(gdf)), t_idx, 60),
            )

        # ── 波动特征 ─────────────────────────────────────────────────────
        ret20 = _pct_change_arr(gdf["close"].values, t_idx, 20)
        feats["volatility_20d"] = float(np.std(ret20) * np.sqrt(252)) if len(ret20) > 1 else 0.0

        # 最大回撤（过去20日）
        close20 = gdf["close"].values[max(0, t_idx - 19) : t_idx + 1]
        feats["max_drawdown_20d"] = _max_drawdown(close20)

        # 平均振幅（过去5日）
        amp5 = gdf.iloc[max(0, t_idx - 4) : t_idx + 1]
        if len(amp5) > 0 and "amplitude" in gdf.columns:
            feats["amplitude_avg_5d"] = float(amp5["amplitude"].mean())
        elif len(amp5) > 0:
            feats["amplitude_avg_5d"] = float(
                ((amp5["high"] - amp5["low"]) / amp5["close"].replace(0, np.nan)).mean()
            )
        else:
            feats["amplitude_avg_5d"] = 0.0

        # ── 均线偏离 ─────────────────────────────────────────────────────
        for ma_col in ["ma5", "ma10", "ma20", "ma60"]:
            ma_val = row.get(ma_col, 0.0)
            if ma_val and ma_val > 0:
                feats[f"close_vs_{ma_col}"] = close_t / ma_val - 1
            else:
                feats[f"close_vs_{ma_col}"] = 0.0

        # ── 趋势形态 ─────────────────────────────────────────────────────
        ma5_arr = gdf["ma5"].values
        ma5_5d = ma5_arr[max(0, t_idx - 4) : t_idx + 1]
        if len(ma5_5d) >= 2:
            x = np.arange(len(ma5_5d), dtype=float)
            slope = np.polyfit(x, ma5_5d, 1)[0]
            feats["ma5_slope_5d"] = float(slope / (ma5_5d.mean() + 1e-9))
        else:
            feats["ma5_slope_5d"] = 0.0

        # 多头排列得分
        ma5_v = row.get("ma5", 0.0)
        ma10_v = row.get("ma10", 0.0)
        ma20_v = row.get("ma20", 0.0)
        alignment = int(ma5_v > ma10_v) + int(ma10_v > ma20_v)
        feats["ma_alignment"] = float(alignment)

        # 连续上涨天数（过去5日）
        close5 = gdf["close"].values[max(0, t_idx - 4) : t_idx + 1]
        feats["consec_up_days"] = float(_consec_up(close5))

        # 实体比均值（过去5日）
        w5 = gdf.iloc[max(0, t_idx - 4) : t_idx + 1]
        if len(w5) > 0:
            amp = (w5["high"] - w5["low"]).replace(0, np.nan)
            body = (w5["close"] - w5["open"]).abs()
            feats["body_ratio_5d"] = float((body / amp).mean())
        else:
            feats["body_ratio_5d"] = 0.0

        # ── 高位预警 ─────────────────────────────────────────────────────
        close_252 = gdf["close"].values[max(0, t_idx - 251) : t_idx + 1]
        high_252 = close_252.max() if len(close_252) > 0 else close_t
        feats["near_52w_high"] = float(close_t / high_252) if high_252 > 0 else 1.0
        feats["below_52w_high_pct"] = float((high_252 - close_t) / high_252) if high_252 > 0 else 0.0

        results[inst] = feats

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "instrument"
    return df.fillna(0.0)


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _rolling_mean(arr: np.ndarray, t_idx: int, window: int) -> float:
    start = max(0, t_idx - window + 1)
    sub = arr[start : t_idx + 1]
    return float(sub.mean()) if len(sub) > 0 else np.nan


def _safe_ratio(num: float, denom: float) -> float:
    if denom and not np.isnan(denom) and denom > 0:
        return float(num / denom)
    return 1.0


def _pct_change_arr(close: np.ndarray, t_idx: int, window: int) -> np.ndarray:
    start = max(0, t_idx - window)
    sub = close[start : t_idx + 1]
    if len(sub) < 2:
        return np.array([])
    return np.diff(sub) / (sub[:-1] + 1e-9)


def _max_drawdown(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.0
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / (peak + 1e-9)
    return float(dd.min())


def _consec_up(prices: np.ndarray) -> int:
    """从最新一日往回数，连续收涨天数。"""
    if len(prices) < 2:
        return 0
    count = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] > prices[i - 1]:
            count += 1
        else:
            break
    return count
