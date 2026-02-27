"""
周K线特征模块 (G 类特征，9个)

核心思路：
  将日线数据聚合为周线（W-FRI），计算反映中期趋势、量能异动和高/低位判断的特征。
  在 T_feat 时刻（蓄力期特征提取点）生成，与日线特征共同作为模型输入。

特征清单：
  G1: weekly_vol_ratio     — 近4周均量 / 近13周均量（量能中期趋势）
  G2: weekly_vol_spike     — 最近1周量 / 近8周均量（近期量能异动）
  G3: weekly_price_pct_26w — 收盘价在近26周价格区间的分位（0=低点, 1=高点）
  G4: weekly_price_pct_52w — 收盘价在近52周价格区间的分位
  G5: weekly_ma_bull       — 多头排列得分（MA5>MA10>MA20 且均线向上）
  G6: weekly_w_bottom      — W底形态得分（过去24周内存在双底结构）
  G7: weekly_ma5_slope     — 周线MA5的3周斜率（近期趋势方向）
  G8: weekly_v_bottom      — V底形态得分（近期单次大跌后快速反弹）
  G9: weekly_box_break     — 箱体突破得分（横盘后向上有效突破）

数据依据（2026-02-27 验证）：
  - weekly_vol_ratio 与好行情涨幅相关系数 +0.35（最强信号）
  - weekly_price_pct_26w > 85% 与差行情大跌强相关（大跌组均值 90.4%）
  - 大牛股在扫描日附近普遍出现本周放量 1.5~2x
  - 多只高涨幅股票（301257、300629）呈 V 形或箱体突破启动形态
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# ─── 特征名常量 ────────────────────────────────────────────────────────────────

WEEKLY_VOL_RATIO      = "weekly_vol_ratio"     # G1: 4周/13周量比
WEEKLY_VOL_SPIKE      = "weekly_vol_spike"     # G2: 本周量/近8周均量
WEEKLY_PRICE_PCT_26W  = "weekly_price_pct_26w" # G3: 26周价格分位
WEEKLY_PRICE_PCT_52W  = "weekly_price_pct_52w" # G4: 52周价格分位
WEEKLY_MA_BULL        = "weekly_ma_bull"       # G5: 均线多头排列得分 (0~1)
WEEKLY_W_BOTTOM       = "weekly_w_bottom"      # G6: W底形态得分 (0~1)
WEEKLY_MA5_SLOPE      = "weekly_ma5_slope"     # G7: 周线MA5的3周斜率
WEEKLY_V_BOTTOM       = "weekly_v_bottom"      # G8: V底形态得分 (0~1)
WEEKLY_BOX_BREAK      = "weekly_box_break"     # G9: 箱体突破得分 (0~1)

WEEKLY_FEATURE_COLS: List[str] = [
    WEEKLY_VOL_RATIO,
    WEEKLY_VOL_SPIKE,
    WEEKLY_PRICE_PCT_26W,
    WEEKLY_PRICE_PCT_52W,
    WEEKLY_MA_BULL,
    WEEKLY_W_BOTTOM,
    WEEKLY_MA5_SLOPE,
    WEEKLY_V_BOTTOM,
    WEEKLY_BOX_BREAK,
]


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _sma(arr: np.ndarray, w: int) -> float:
    """简单移动平均（取末尾 w 个值）。"""
    if len(arr) < w:
        return np.nan
    return float(np.mean(arr[-w:]))


def _detect_w_bottom(closes: np.ndarray, window: int = 24) -> float:
    """
    W底形态识别。
    在过去 window 根周K中寻找双底结构，返回得分：
      1.0 = 存在W底且当前价已突破颈线（右侧启动）
      0.5 = 存在W底但尚未突破颈线（颈线附近）
      0.0 = 未发现W底

    识别条件（放宽版，基于验证数据）：
      - 两低点价差 < 25%（原为 15%，扩大覆盖范围）
      - 中间颈线反弹 > 6%
      - 两低点之间最少 3 根周K（避免假双底）
    """
    w = closes[-min(len(closes), window):]
    n = len(w)
    if n < 10:
        return 0.0

    cur = w[-1]
    best_score = 0.0

    # 滑动搜索所有可能的双底组合
    for ll_idx in range(0, n // 2):
        for rl_idx in range(ll_idx + 3, n - 1):
            ll = w[ll_idx]
            rl = w[rl_idx]
            if ll <= 0 or rl <= 0:
                continue
            # 两低点价差 < 25%
            low_diff = abs(ll - rl) / max(ll, rl)
            if low_diff >= 0.25:
                continue
            # 中间颈线高点
            neck = float(np.max(w[ll_idx:rl_idx + 1]))
            higher_low = max(ll, rl)
            rally = (neck - higher_low) / higher_low
            if rally < 0.06:
                continue
            # 找到有效W底
            if cur > neck * 0.97:
                best_score = 1.0  # 已突破颈线
                break
            elif cur > neck * 0.90:
                best_score = max(best_score, 0.5)  # 颈线附近
            else:
                best_score = max(best_score, 0.3)  # 还在底部
        if best_score == 1.0:
            break

    return best_score


def _detect_v_bottom(closes: np.ndarray, window: int = 16) -> float:
    """
    V底形态识别：近期经历单次大幅下跌后快速强力反弹，当前价已回到或超过跌前水平。

    识别条件：
      - 在最近 window 根周K中，找到一个低点（最低价）
      - 低点之前的高点（低点前1~8周内）与低点之间跌幅 >= 15%（下跌幅度要有意义）
      - 低点之后当前价反弹幅度 >= 12%（已经有力反弹）
      - 整个过程在最近 window 周内（近期事件，非远古历史）

    返回值：
      1.0 = V底反弹且当前价已超过跌前高点（完全收复失地）
      0.7 = V底反弹中，已反弹超过60%但未完全收复
      0.4 = V底反弹中，已反弹超过30%
      0.0 = 未发现V底
    """
    w = closes[-min(len(closes), window):]
    n = len(w)
    if n < 8:
        return 0.0

    cur = w[-1]

    # 在后半段（近8周）找最低点
    search_range = w[max(0, n - 8):]
    low_idx_local = int(np.argmin(search_range))
    low_idx = max(0, n - 8) + low_idx_local
    low_val = w[low_idx]

    # 低点不能是最后一根（需要已经开始反弹）
    if low_idx >= n - 1:
        return 0.0

    # 低点之前找高点（低点前1~8周）
    pre_range = w[max(0, low_idx - 8): low_idx]
    if len(pre_range) < 1:
        return 0.0
    pre_high = float(np.max(pre_range))

    # 下跌幅度需 >= 15%
    if pre_high <= 0 or low_val <= 0:
        return 0.0
    drop = (pre_high - low_val) / pre_high
    if drop < 0.15:
        return 0.0

    # 从低点到当前的反弹幅度
    rebound = (cur - low_val) / low_val
    if rebound < 0.12:
        return 0.0

    # 评分
    if cur >= pre_high * 0.98:
        return 1.0          # 完全收复
    elif rebound >= drop * 0.6:
        return 0.7          # 反弹超过下跌60%
    elif rebound >= drop * 0.3:
        return 0.4          # 反弹超过下跌30%
    else:
        return 0.0


def _detect_box_break(closes: np.ndarray, window: int = 20) -> float:
    """
    箱体突破形态识别：近N周横盘震荡后向上有效突破。

    识别条件：
      - 在最近 window 周中，前半段（window//2 周）处于横盘状态：
        价格区间 (max-min)/min < 15%（振幅不超过15%）
      - 横盘至少持续 5 周（避免一两根K线的短暂盘整）
      - 当前价相对横盘上轨突破幅度 >= 2%（有效向上突破）
      - 突破发生在最近 4 周内（近期事件）

    返回值：
      1.0 = 当前正处于突破状态（近2周内突破，突破幅度 > 5%）
      0.7 = 近4周内突破，幅度2~5%
      0.3 = 仍在箱体内但已接近上轨（距上轨 < 3%）
      0.0 = 未发现箱体或未突破
    """
    w = closes[-min(len(closes), window):]
    n = len(w)
    if n < 10:
        return 0.0

    cur = w[-1]
    half = max(5, n // 2)

    # 用前半段判断是否横盘
    consolidation = w[: n - 4]   # 留最近4周给突破判断
    if len(consolidation) < 5:
        return 0.0

    box_hi = float(np.max(consolidation))
    box_lo = float(np.min(consolidation))
    if box_lo <= 0:
        return 0.0

    amplitude = (box_hi - box_lo) / box_lo
    if amplitude >= 0.18:        # 振幅超过18%，不算横盘
        return 0.0

    # 平稳性检验：前半段与后半段均价差异 < 5%，排除单边趋势行情
    mid = len(consolidation) // 2
    mean_first = float(np.mean(consolidation[:mid]))
    mean_second = float(np.mean(consolidation[mid:]))
    if mean_first > 0 and abs(mean_second - mean_first) / mean_first > 0.05:
        return 0.0              # 价格单方向趋势，不是横盘

    # 检查近4周是否发生突破
    recent = w[n - 4:]
    break_weeks = [i for i, p in enumerate(recent) if p > box_hi * 1.02]

    if not break_weeks:
        # 尚未突破，检查是否接近上轨
        if cur > box_hi * 0.97:
            return 0.3
        return 0.0

    first_break = break_weeks[0]  # 第一次突破是第几周（0-based，0=最早）
    break_pct = (cur - box_hi) / box_hi

    if first_break <= 1 and break_pct >= 0.05:
        return 1.0           # 近2周内突破且突破幅度 > 5%
    elif break_pct >= 0.02:
        return 0.7           # 已突破，幅度2~5%
    else:
        return 0.3           # 突破幅度不足但已越过箱体


def _weekly_metrics_single(
    closes: np.ndarray,
    volumes: np.ndarray,
) -> dict:
    """
    对单只股票的周线序列（时间升序）计算所有 G 类指标。

    Args:
        closes:  周K收盘价序列（升序）
        volumes: 周K成交量序列（升序）

    Returns:
        dict，键为特征名，值为 float（不足数据时为 NaN）
    """
    result = {k: np.nan for k in WEEKLY_FEATURE_COLS}
    n = len(closes)
    if n < 8:
        return result

    c = closes
    v = volumes
    cur = c[-1]

    # G1: 近4周均量 / 近13周均量
    vol4 = _sma(v, 4)
    vol13 = _sma(v, 13)
    if not np.isnan(vol4) and not np.isnan(vol13) and vol13 > 0:
        result[WEEKLY_VOL_RATIO] = vol4 / vol13

    # G2: 最近1周量 / 近8周均量（排除最后1周自身）
    if n >= 9:
        vol8_excl = float(np.mean(v[-9:-1]))
        if vol8_excl > 0:
            result[WEEKLY_VOL_SPIKE] = float(v[-1]) / vol8_excl

    # G3: 26周价格分位
    win26 = min(n, 26)
    hi26, lo26 = float(np.max(c[-win26:])), float(np.min(c[-win26:]))
    if hi26 > lo26:
        result[WEEKLY_PRICE_PCT_26W] = (cur - lo26) / (hi26 - lo26)

    # G4: 52周价格分位
    win52 = min(n, 52)
    hi52, lo52 = float(np.max(c[-win52:])), float(np.min(c[-win52:]))
    if hi52 > lo52:
        result[WEEKLY_PRICE_PCT_52W] = (cur - lo52) / (hi52 - lo52)

    # G5: 均线多头排列得分
    ma5  = _sma(c, 5)
    ma10 = _sma(c, 10)
    ma20 = _sma(c, min(20, n))
    if not any(np.isnan([ma5, ma10, ma20])):
        ma5p  = _sma(c[:-1], 5)   if n > 5  else np.nan
        ma10p = _sma(c[:-1], 10)  if n > 10 else np.nan
        score = 0.0
        if ma5 > ma10:   score += 0.3
        if ma10 > ma20:  score += 0.3
        if not np.isnan(ma5p)  and ma5  > ma5p:  score += 0.2
        if not np.isnan(ma10p) and ma10 > ma10p: score += 0.2
        result[WEEKLY_MA_BULL] = score

    # G6: W底形态得分
    result[WEEKLY_W_BOTTOM] = _detect_w_bottom(c)

    # G7: 周线MA5的3周斜率
    if n > 8:
        ma5_cur  = _sma(c, 5)
        ma5_3ago = _sma(c[:-3], 5)
        if not any(np.isnan([ma5_cur, ma5_3ago])) and ma5_3ago > 0:
            result[WEEKLY_MA5_SLOPE] = (ma5_cur - ma5_3ago) / ma5_3ago

    # G8: V底形态得分
    result[WEEKLY_V_BOTTOM] = _detect_v_bottom(c)

    # G9: 箱体突破得分
    result[WEEKLY_BOX_BREAK] = _detect_box_break(c)

    return result


# ─── 主接口 ────────────────────────────────────────────────────────────────────

def compute_weekly_features(
    kline: pd.DataFrame,
    feat_date: str,
    instruments: List[str],
) -> pd.DataFrame:
    """
    计算 G 类周K线特征。

    Args:
        kline:       日线 DataFrame，须含 date/instrument/close/volume 列
        feat_date:   特征提取日期（只用 <= feat_date 的数据，严格防泄漏）
        instruments: 目标股票列表

    Returns:
        DataFrame，index=instrument，columns=WEEKLY_FEATURE_COLS
        数据不足时对应特征返回 NaN（不阻断整体流程）
    """
    feat_ts = pd.Timestamp(feat_date)
    result = pd.DataFrame(index=instruments, columns=WEEKLY_FEATURE_COLS, dtype=float)

    if kline.empty:
        return result

    # 过滤到 <= feat_date 的数据，只保留目标股票
    inst_set = set(instruments)
    kl = kline[
        (kline["date"] <= feat_ts) &
        (kline["instrument"].isin(inst_set))
    ][["date", "instrument", "close", "volume"]].copy()

    if kl.empty:
        return result

    # 聚合成周线（每周五收盘）
    kl = kl.sort_values(["instrument", "date"])
    kl_idx = kl.set_index("date")

    # 按股票分组，resample 成周线
    n_processed = 0
    for inst, grp in kl_idx.groupby("instrument"):
        if inst not in inst_set:
            continue
        wk = grp[["close", "volume"]].resample("W-FRI").agg(
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"])
        wk = wk[wk.index <= feat_ts]

        if len(wk) < 8:
            continue

        c = wk["close"].values.astype(float)
        v = wk["volume"].values.astype(float)

        metrics = _weekly_metrics_single(c, v)
        for col, val in metrics.items():
            result.loc[inst, col] = val
        n_processed += 1

    logger.debug(
        f"Weekly features computed: {n_processed}/{len(instruments)} stocks "
        f"on {feat_date}"
    )
    return result
