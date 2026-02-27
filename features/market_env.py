"""
市场环境特征模块 (I 类特征，3个)

核心思路：
  这3个特征对所有股票在同一天取相同的值（市场级别特征），但提供了重要的
  "环境上下文"，让模型能自动学习：好行情和差行情下，各信号的权重应该不同。

  实证依据（2026-02-27 验证）：
  - 好行情(2025-12-12, 上证+5.74%)：多头排列/量能特征更有效
  - 差行情(2024-12-10, 上证-7.45%)：相对走弱60成噪音，需市场状态来区分

特征清单：
  I1: market_trend_20d   — 大盘指数近20日涨幅（正=牛市，负=熊市）
  I2: market_vol_20d     — 市场近20日日收益率标准差（高=高波动，低=低波动）
  I3: market_breadth_5d  — 近5日全市场上涨股票占比（0~1，越高越健康）

数据来源：
  - I1/I2：复用 PrecursorFeatureExtractor._market_ret（来自 index_bar1d 或全市场均值）
  - I3：从 kline 全量数据计算每日上涨股票比例

重要说明：
  这3个特征在同一天对所有股票取相同值，不会导致"泄漏"（因为特征时间点
  就是 feat_date 当天，而预测目标是 feat_date 之后），符合防泄漏原则。
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


# ─── 特征名常量 ────────────────────────────────────────────────────────────────

MARKET_TREND_20D   = "market_trend_20d"    # I1: 指数近20日涨幅
MARKET_VOL_20D     = "market_vol_20d"      # I2: 近20日波动率（日收益标准差）
MARKET_BREADTH_5D  = "market_breadth_5d"   # I3: 近5日上涨股票占比

MARKET_ENV_COLS: List[str] = [
    MARKET_TREND_20D,
    MARKET_VOL_20D,
    MARKET_BREADTH_5D,
]


# ─── 主接口 ────────────────────────────────────────────────────────────────────

def compute_market_env_features(
    market_ret: pd.Series,
    kline: pd.DataFrame,
    feat_date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    计算 I 类市场环境特征。

    Args:
        market_ret:   日涨幅序列，index=date（pd.Timestamp），value=float
                      来自 index_bar1d 或全市场等权均值，已在 PrecursorFeatureExtractor 中计算好
        kline:        日线全量数据，含 date/instrument/close 列（用于计算市场宽度）
        feat_date:    特征提取日期（严格使用 <= feat_date 数据，防泄漏）
        instruments:  目标股票列表（所有股票取同一市场值）

    Returns:
        DataFrame，index=instrument，columns=MARKET_ENV_COLS
        三列对所有股票取相同值（市场级别特征）。
    """
    feat_ts = pd.Timestamp(feat_date)
    result = pd.DataFrame(
        index=instruments, columns=MARKET_ENV_COLS, dtype=float
    )

    # ── I1 & I2：基于市场收益率序列 ────────────────────────────────────────────
    if market_ret is not None and not market_ret.empty:
        mret = market_ret[market_ret.index <= feat_ts].dropna()

        # I1: 近20日累计涨幅（乘积形式：(1+r1)*(1+r2)*...-1）
        last20 = mret.tail(20)
        if len(last20) >= 10:
            trend_20d = float((1 + last20).prod() - 1)
            result[MARKET_TREND_20D] = trend_20d
        else:
            result[MARKET_TREND_20D] = np.nan

        # I2: 近20日波动率（日收益标准差）
        if len(last20) >= 5:
            vol_20d = float(last20.std())
            result[MARKET_VOL_20D] = vol_20d
        else:
            result[MARKET_VOL_20D] = np.nan
    else:
        logger.debug("I类: market_ret 为空，I1/I2 置 NaN")

    # ── I3：近5日全市场上涨股票占比（市场宽度）──────────────────────────────────
    if kline is not None and not kline.empty:
        breadth = _calc_market_breadth(kline, feat_ts, window=5)
        result[MARKET_BREADTH_5D] = breadth
    else:
        result[MARKET_BREADTH_5D] = np.nan

    logger.debug(
        f"I类市场特征: trend={result[MARKET_TREND_20D].iloc[0]:.3f}  "
        f"vol={result[MARKET_VOL_20D].iloc[0]:.4f}  "
        f"breadth={result[MARKET_BREADTH_5D].iloc[0]:.3f}  "
        f"feat_date={feat_date}"
    )
    return result


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _calc_market_breadth(
    kline: pd.DataFrame,
    feat_ts: pd.Timestamp,
    window: int = 5,
) -> float:
    """
    计算近 window 个交易日的全市场上涨股票平均占比。

    方法：
      1. 取 <= feat_ts 的最近 window+1 个交易日的收盘价矩阵
      2. 计算日涨跌（pct_change），统计每日上涨股票数 / 总股票数
      3. 返回近 window 日的平均宽度

    Returns:
        0~1 之间的浮点数，越高表示上涨股票越多（市场越健康）
        数据不足时返回 NaN
    """
    kl = kline[kline["date"] <= feat_ts][["date", "instrument", "close"]].copy()
    if kl.empty:
        return np.nan

    # 取最近 window+1 天（多取1天用于计算涨跌）
    recent_dates = sorted(kl["date"].unique())
    if len(recent_dates) < window + 1:
        return np.nan
    cutoff = recent_dates[-(window + 1)]
    kl = kl[kl["date"] >= cutoff]

    # 透视为宽表（日期×股票）
    try:
        pivot = kl.pivot_table(index="date", columns="instrument", values="close")
    except Exception:
        return np.nan

    # 计算日涨跌
    daily_ret = pivot.pct_change(fill_method=None).dropna(how="all")
    if len(daily_ret) < 1:
        return np.nan

    # 每日上涨股票占比
    daily_up = (daily_ret > 0).sum(axis=1) / daily_ret.notna().sum(axis=1)
    breadth = float(daily_up.tail(window).mean())
    return breadth
