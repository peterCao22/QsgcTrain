"""
交互特征模块 (J 类特征，2个)

核心思路：
  LightGBM 能学习特征重要性，但无法自动学习"同一特征在不同市场环境下方向相反"
  这类高阶交互。显式构造交互特征，让模型直接获得"有条件的信号"。

问题背景（T8：相对走弱60 特征重审）：
  - 差行情（2024-12-10 扫描）Top50 中约 60% 出现"相对走弱60"
  - price_slope_60d < 0 的股票在熊市中平均涨幅 -5.24%，是最大噪音源
  - 根本原因：
    * 好行情：60日动量信号有效（强者恒强）
    * 差行情：60日动量信号反转或失效（超卖弱势股不代表机会）
  - 解决方案：构造"市场环境×动量"交互项，让模型感知信号的方向依赖

特征清单：
  J1: slope60_mkt_adj  — 60日超额斜率 × (1 + 市场20日涨幅)
                          牛市放大动量信号，熊市压缩甚至反转
  J2: pct52w_mkt_risk  — 52周价格分位 × (1 − 市场宽度5日)
                          熊市中高位股的风险复合分（越高越危险）

实证逻辑：
  差行情 (market_trend_20d ≈ -7.45%):
    - J1: price_slope_60d = -0.05 → J1 = -0.05 * (1-0.074) = -0.046（较小）
          price_slope_60d = +0.05 → J1 = 0.05 * 0.926 = 0.046（缩小信号）
  好行情 (market_trend_20d ≈ +5%):
    - J1: price_slope_60d = -0.05 → J1 = -0.05 * 1.05 = -0.053（更负，更危险）
          price_slope_60d = +0.05 → J1 = 0.05 * 1.05 = 0.053（放大信号）
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


# ─── 特征名常量 ────────────────────────────────────────────────────────────────

SLOPE60_MKT_ADJ  = "slope60_mkt_adj"   # J1: 60日超额斜率 × 市场环境修正
PCT52W_MKT_RISK  = "pct52w_mkt_risk"   # J2: 52周分位 × 熊市风险乘数

CROSS_FEATURE_COLS: List[str] = [
    SLOPE60_MKT_ADJ,
    PCT52W_MKT_RISK,
]


# ─── 主接口 ────────────────────────────────────────────────────────────────────

def compute_cross_features(
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    在已有特征 DataFrame 上构造 J 类交互特征。

    Args:
        feat_df:  已包含所有基础特征的 DataFrame，index=instrument
                  需要包含以下列（缺失时对应 J 类特征置 NaN）：
                  - price_slope_60d (A类)
                  - market_trend_20d (I类)
                  - weekly_price_pct_52w (G类)
                  - market_breadth_5d (I类)

    Returns:
        DataFrame，index=instrument，columns=CROSS_FEATURE_COLS
    """
    result = pd.DataFrame(
        index=feat_df.index, columns=CROSS_FEATURE_COLS, dtype=float
    )

    # J1: slope60_mkt_adj = price_slope_60d × (1 + market_trend_20d)
    if "price_slope_60d" in feat_df.columns and "market_trend_20d" in feat_df.columns:
        s60  = pd.to_numeric(feat_df["price_slope_60d"],  errors="coerce")
        mkt  = pd.to_numeric(feat_df["market_trend_20d"], errors="coerce")
        result[SLOPE60_MKT_ADJ] = s60 * (1 + mkt)
        n_valid = result[SLOPE60_MKT_ADJ].notna().sum()
        logger.debug(f"J1 slope60_mkt_adj: {n_valid}/{len(feat_df)} 只有效")
    else:
        logger.debug("J1: 缺少 price_slope_60d 或 market_trend_20d，置 NaN")

    # J2: pct52w_mkt_risk = weekly_price_pct_52w × (1 − market_breadth_5d)
    # 熊市（breadth 低）× 高价格分位 = 高风险；牛市（breadth 高）× 高分位 = 较低风险
    if "weekly_price_pct_52w" in feat_df.columns and "market_breadth_5d" in feat_df.columns:
        pct52  = pd.to_numeric(feat_df["weekly_price_pct_52w"],  errors="coerce")
        bread  = pd.to_numeric(feat_df["market_breadth_5d"],     errors="coerce")
        result[PCT52W_MKT_RISK] = pct52 * (1 - bread)
        n_valid = result[PCT52W_MKT_RISK].notna().sum()
        logger.debug(f"J2 pct52w_mkt_risk: {n_valid}/{len(feat_df)} 只有效")
    else:
        logger.debug("J2: 缺少 weekly_price_pct_52w 或 market_breadth_5d，置 NaN")

    return result
