"""
蓄力期特征提取模块

核心设计：
  T_entry = 股票首次进入强势股池的日期（已经涨起来了）
  T_feat  = T_entry - 20 个交易日（蓄力期特征提取点）

  在 T_feat 时刻提取"前驱特征"——描述股票在大涨前的"蓄力状态"，
  目标是预测该股未来 20 日的涨幅是否位于截面前 20%（is_top20pct）。

6 类共 50+ 个特征（模型自动发现重要性，不预先筛选）：
  A. 趋势斜率（方向在变好？）         5 个
  B. 多周期动量/比率变化              10 个
  C. 持续时间 / 形态确认              6 个
  D. 层级对比（短期 vs 长期位置）     7 个
  E. 历史强势信号（来自 strong_pool） 3 个
  F. 估值特征（来自 valuation_all）   6 个

使用方式：
    extractor = PrecursorFeatureExtractor(kline, chips, moneyflow, strong_pool_hist)
    df = extractor.compute("2025-03-01", instruments)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── 特征名常量 ────────────────────────────────────────────────────────────────

# A类：趋势斜率（5个）
VOL_SLOPE_60D        = "vol_slope_60d"   # 成交量 60 日归一化斜率
PRICE_SLOPE_60D      = "price_slope_60d" # 价格超额 60 日斜率（vs 大盘）
MF_SLOPE_30D         = "mf_slope_30d"   # 主力资金流 30 日斜率
MF_SLOPE_10D         = "mf_slope_10d"   # 主力资金流 10 日斜率（短期加速）
PRICE_SLOPE_20D      = "price_slope_20d" # 价格超额 20 日斜率（短周期趋势）

# B类：多周期动量 / 比率变化（10个）
VOL_ACCEL            = "vol_accel"         # 量能加速比（近20日均量 / 近60日均量）
VOL_RATIO_5_20       = "vol_ratio_5_20"    # 近5日均量 / 近20日均量（短期量能爆发）
RANGE_COMPRESS       = "range_compress"    # 振幅收窄（近20日ATR / 近60日ATR）
MOMENTUM_ACCEL       = "momentum_accel"    # 动量加速度（近20日收益 - 近60日收益/3）
EXCESS_RET_CHANGE    = "excess_ret_change" # 超额收益变化（近20日超额 - 近60日超额）
RET_5D               = "ret_5d"            # 近5日涨幅
RET_10D              = "ret_10d"           # 近10日涨幅
RET_20D              = "ret_20d"           # 近20日涨幅
RET_60D              = "ret_60d"           # 近60日涨幅
BB_WIDTH             = "bb_width"          # 布林带宽度（近20日std*4 / MA20，越窄=蓄力）

# C类：持续时间 / 形态确认（6个）
DAYS_ABOVE_MA20      = "days_above_ma20"   # 连续站上 MA20 的天数
CONSEC_HIGHER_LOW    = "consec_higher_low" # 连续创更高低点
DAYS_POSITIVE_MF     = "days_positive_mf" # 主力净流入连续天数
HIGH_CLOSE_RATIO     = "high_close_ratio"  # 近5日收盘位置比（收盘/最高，越高越强势）
CONSEC_GREEN         = "consec_green"      # 末端连续上涨天数
VOL_SPIKE_COUNT      = "vol_spike_count"   # 近20日放量（>均量1.5倍）次数

# D类：层级对比（7个）
DIST_52W_HIGH        = "dist_52w_high"       # 距52周高点距离（越小=越接近年高）
DIST_60D_LOW_REBOUND = "dist_60d_low_rebound"# 从60日低点反弹幅度
PRICE_VS_MA5         = "price_vs_ma5"        # 现价相对MA5偏离度
PRICE_VS_MA60        = "price_vs_ma60"       # 现价相对MA60偏离度
WIN_PERCENT          = "win_percent"         # 筹码盈利比例
CHIP_CONCENTRATION   = "chip_concentration" # 筹码集中度
PRICE_TO_AVGCOST     = "price_to_avgcost"   # 价格与平均成本比率

# E类：历史强势信号（3个）
PREV_TJ_BOARDS        = "prev_tj_boards"       # 近N月内最大涨停板数
PREV_NEW_HIGH_COUNT   = "prev_new_high_count"  # 近N月内出现新高次数
PREV_POOL_APPEARANCES = "prev_pool_appearances"# 历史入强势池天数

# F类：估值特征（6个）
LOG_FLOAT_CAP  = "log_float_cap"  # log(流通市值)
PE_TTM         = "pe_ttm"         # 市盈率TTM（负值=亏损）
PB             = "pb"             # 市净率
PS_TTM         = "ps_ttm"         # 市销率TTM
PE_SECT_RANK   = "pe_sect_rank"   # PE在截面全市场的百分位（0~1）
PB_HIST_RANK   = "pb_hist_rank"   # PB在自身过去252日历史百分位（0~1）

# H类：概念热度特征（3个，来自 features/concept.py）
CONCEPT_RANK_20D  = "concept_rank_20d"   # H1: 最热概念20日涨幅百分位
CONCEPT_RANK_60D  = "concept_rank_60d"   # H2: 最热概念60日涨幅百分位
CONCEPT_MF_TREND  = "concept_mf_trend"   # H3: 最热概念资金流趋势(5d/20d均值比)

# I类：市场环境特征（3个，来自 features/market_env.py）
MARKET_TREND_20D  = "market_trend_20d"   # I1: 指数近20日涨幅（同日所有股票相同）
MARKET_VOL_20D    = "market_vol_20d"     # I2: 近20日波动率（日收益标准差）
MARKET_BREADTH_5D = "market_breadth_5d"  # I3: 近5日全市场上涨股票占比

# J类：交互特征（2个，来自 features/cross.py）— T8 相对走弱60 重审
SLOPE60_MKT_ADJ  = "slope60_mkt_adj"    # J1: 60日超额斜率 × (1+市场20日涨幅)
PCT52W_MKT_RISK  = "pct52w_mkt_risk"    # J2: 52周价格分位 × (1−市场宽度5日)

# G类：周K线特征（9个，来自 features/weekly.py）
WEEKLY_VOL_RATIO     = "weekly_vol_ratio"     # G1: 4周/13周量比（量能中期趋势）
WEEKLY_VOL_SPIKE     = "weekly_vol_spike"     # G2: 本周量/近8周均量（量能异动）
WEEKLY_PRICE_PCT_26W = "weekly_price_pct_26w" # G3: 26周价格分位（高/低位判断）
WEEKLY_PRICE_PCT_52W = "weekly_price_pct_52w" # G4: 52周价格分位
WEEKLY_MA_BULL       = "weekly_ma_bull"       # G5: 均线多头排列得分(0~1)
WEEKLY_W_BOTTOM      = "weekly_w_bottom"      # G6: W底形态得分(0~1)
WEEKLY_MA5_SLOPE     = "weekly_ma5_slope"     # G7: 周线MA5三周斜率
WEEKLY_V_BOTTOM      = "weekly_v_bottom"      # G8: V底形态得分(0~1)
WEEKLY_BOX_BREAK     = "weekly_box_break"     # G9: 箱体突破得分(0~1)

# D类扩展：筹码优化（1个）
CHIP_VS_AVG_COST = "chip_vs_avg_cost"  # (close - avg_cost) / avg_cost，价格在成本区间的相对位置

ALL_FEATURE_COLS: List[str] = [
    # A: 趋势斜率 (5)
    VOL_SLOPE_60D, PRICE_SLOPE_60D, MF_SLOPE_30D, MF_SLOPE_10D, PRICE_SLOPE_20D,
    # B: 多周期动量/比率 (10)
    VOL_ACCEL, VOL_RATIO_5_20, RANGE_COMPRESS, MOMENTUM_ACCEL, EXCESS_RET_CHANGE,
    RET_5D, RET_10D, RET_20D, RET_60D, BB_WIDTH,
    # C: 持续时间/形态 (6)
    DAYS_ABOVE_MA20, CONSEC_HIGHER_LOW, DAYS_POSITIVE_MF,
    HIGH_CLOSE_RATIO, CONSEC_GREEN, VOL_SPIKE_COUNT,
    # D: 层级对比 (8，含新增 chip_vs_avg_cost)
    DIST_52W_HIGH, DIST_60D_LOW_REBOUND, PRICE_VS_MA5, PRICE_VS_MA60,
    WIN_PERCENT, CHIP_CONCENTRATION, PRICE_TO_AVGCOST, CHIP_VS_AVG_COST,
    # E: 历史强势 (3)
    PREV_TJ_BOARDS, PREV_NEW_HIGH_COUNT, PREV_POOL_APPEARANCES,
    # F: 估值 (6)
    LOG_FLOAT_CAP, PE_TTM, PB, PS_TTM, PE_SECT_RANK, PB_HIST_RANK,
    # G: 周K线特征 (9)
    WEEKLY_VOL_RATIO, WEEKLY_VOL_SPIKE,
    WEEKLY_PRICE_PCT_26W, WEEKLY_PRICE_PCT_52W,
    WEEKLY_MA_BULL, WEEKLY_W_BOTTOM, WEEKLY_MA5_SLOPE,
    WEEKLY_V_BOTTOM, WEEKLY_BOX_BREAK,
    # H: 概念热度特征 (3)
    CONCEPT_RANK_20D, CONCEPT_RANK_60D, CONCEPT_MF_TREND,
    # I: 市场环境特征 (3)
    MARKET_TREND_20D, MARKET_VOL_20D, MARKET_BREADTH_5D,
    # J: 交互特征 (2)  — T8 相对走弱60重审
    SLOPE60_MKT_ADJ, PCT52W_MKT_RISK,
]
# 共 55 个特征（A5 + B10 + C6 + D8 + E3 + F6 + G9 + H3 + I3 + J2）
# v2.0 新增：G9(周K线) + D1(chip_vs_avg_cost) + H3(概念热度) + I3(市场环境) + J2(交互)


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _normalized_slope(series: np.ndarray) -> float:
    """
    计算线性回归斜率，除以均值绝对值归一化。
    序列太短（<5）或均值接近0时返回 NaN。
    """
    n = len(series)
    if n < 5:
        return np.nan
    x = np.arange(n, dtype=float)
    # polyfit: [slope, intercept]
    slope = np.polyfit(x, series, 1)[0]
    mean_val = np.abs(series).mean()
    if mean_val < 1e-10:
        return np.nan
    return float(slope / mean_val)


def _consec_true_from_end(arr: np.ndarray) -> int:
    """返回数组末尾连续 True 的天数。"""
    count = 0
    for v in reversed(arr):
        if v:
            count += 1
        else:
            break
    return count


# ─── 主类 ────────────────────────────────────────────────────────────────────

class PrecursorFeatureExtractor:
    """
    蓄力期特征提取器。

    初始化时传入预加载好的 DataFrame（宽窗口，覆盖 T_feat-130d ~ T_feat）。
    调用 compute(feat_date, instruments) 返回截面特征矩阵。
    """

    def __init__(
        self,
        kline: pd.DataFrame,
        chips: pd.DataFrame,
        moneyflow: pd.DataFrame,
        strong_pool_hist: pd.DataFrame,
        index_kline: Optional[pd.DataFrame] = None,
        valuation: Optional[pd.DataFrame] = None,
        concept_bar: Optional[pd.DataFrame] = None,
        concept_comp_range: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            kline:              日线数据，必须包含 date, instrument, close, high, low,
                                volume, ma20（若无 ma20 则内部计算）
            chips:              筹码数据，date, instrument, win_percent, concentration, avg_cost
            moneyflow:          资金流数据，date, instrument, netflow_amount_main
            strong_pool_hist:   历史强势股池，date, instrument, tj_boards, new_high
            index_kline:        大盘指数日K线（来自 index_bar1d 表），
                                含 date, instrument, close 列。
                                传入时优先用真实指数收益率计算超额收益；
                                不传时降级为全市场等权均值（精度略低）。
            valuation:          个股估值数据（valuation_all 表），
                                含 date, instrument, pe_ttm, pb, ps_ttm, float_market_cap。
                                不传时 F 类估值特征全部置 NaN。
            concept_bar:        概念日K线（concept_bar1d 表），
                                含 date, concept_code, pct_change, net_amount。
                                不传时 H 类概念特征全部置 NaN。
            concept_comp_range: 概念成分快照（concept_component 表），
                                含 date, concept_code, instrument。
                                不传时 H 类概念特征全部置 NaN。
        """
        kline["date"] = pd.to_datetime(kline["date"])
        self._kline = kline.sort_values(["instrument", "date"]) # 全市场宽表，一次加载，多次复用

        # 市场基准收益率：优先用真实指数（000001.SH 上证指数），
        # 无指数数据时退化为全市场股票等权均值（粗略但不阻断流程）
        if index_kline is not None and not index_kline.empty:
            index_kline = index_kline.copy()
            index_kline["date"] = pd.to_datetime(index_kline["date"])
            # 若有多个指数，取第一个（默认上证指数）
            first_inst = index_kline["instrument"].iloc[0]
            idx = index_kline[index_kline["instrument"] == first_inst].sort_values("date")
            # 指数涨幅
            self._market_ret = (
                idx.set_index("date")["close"]
                .pct_change()
                .rename("mkt_ret")
            )
            logger.info(f"市场基准：真实指数 {first_inst}，{len(self._market_ret)} 个交易日")
        else:
            # 降级：全市场等权均值，受个股异常值影响，但无需额外数据
            self._market_ret = (
                kline.groupby("date")["close"]
                .mean()
                .pct_change()
                .rename("mkt_ret")
            )
            logger.warning("index_kline 未传入，市场基准降级为全市场等权均值")

        # 筹码/资金流预先按股票分组为字典，避免每次计算时 O(n) 全表扫描
        # 对 5000+ 只股票 × 35 个截面日期，这将查询从 O(n*k) 降到 O(1) 字典查找
        if not chips.empty:
            chips["date"] = pd.to_datetime(chips["date"])
            self._chips_by_inst = {
                inst: grp.sort_values("date")
                for inst, grp in chips.groupby("instrument")
            }
        else:
            self._chips_by_inst = {}
        self._chips = chips  # 保留原表备用

        if not moneyflow.empty:
            moneyflow["date"] = pd.to_datetime(moneyflow["date"])
            self._mf_by_inst = {
                inst: grp.sort_values("date")
                for inst, grp in moneyflow.groupby("instrument")
            }
        else:
            self._mf_by_inst = {}
        self._mf = moneyflow  # 保留原表备用

        strong_pool_hist["date"] = pd.to_datetime(strong_pool_hist["date"])
        self._sp = strong_pool_hist

        # 估值数据：按股票预分组，O(1) 查找
        if valuation is not None and not valuation.empty:
            valuation = valuation.copy()
            valuation["date"] = pd.to_datetime(valuation["date"])
            self._val_by_inst = {
                inst: grp.sort_values("date")
                for inst, grp in valuation.groupby("instrument")
            }
            # 全市场截面，用于 pe_sect_rank（板块内 PE 分位）
            self._val_all = valuation
        else:
            self._val_by_inst = {}
            self._val_all = pd.DataFrame()

        # 概念热度数据（H 类特征，可选）
        if concept_bar is not None and not concept_bar.empty:
            cb = concept_bar.copy()
            cb["date"] = pd.to_datetime(cb["date"])
            self._concept_bar: Optional[pd.DataFrame] = cb
        else:
            self._concept_bar = None

        if concept_comp_range is not None and not concept_comp_range.empty:
            cc = concept_comp_range.copy()
            cc["date"] = pd.to_datetime(cc["date"])
            self._concept_comp_range: Optional[pd.DataFrame] = cc
        else:
            self._concept_comp_range = None

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def compute(
        self,
        feat_date: str,
        instruments: List[str],
    ) -> pd.DataFrame:
        """
        提取指定截面日期所有股票的蓄力期特征。

        Args:
            feat_date:   特征提取日期（T_feat），即强势股入池前 N 天
            instruments: 目标股票列表

        Returns:
            DataFrame，index=instrument，columns=ALL_FEATURE_COLS
            缺失数据以 NaN 填充，调用方负责后续插补。
        """
        feat_ts = pd.Timestamp(feat_date)

        # 取覆盖窗口 160 个交易日的 kline 子集（约 8 个月日历日）
        lookback_start = feat_ts - pd.Timedelta(days=250)
        kline_w = self._kline[
            (self._kline["date"] >= lookback_start) &
            (self._kline["date"] <= feat_ts) &
            (self._kline["instrument"].isin(instruments))
        ].copy()

        if kline_w.empty:
            logger.warning(f"[precursor] No kline data on/before {feat_date}")
            return pd.DataFrame(index=instruments, columns=ALL_FEATURE_COLS, dtype=float)

        # 确保 ma20 存在
        if "ma20" not in kline_w.columns:
            kline_w["ma20"] = (
                kline_w.groupby("instrument")["close"]
                .transform(lambda x: x.rolling(20, min_periods=10).mean())
            )

        # 按股票分组计算
        results: List[Dict] = []
        for inst, grp in kline_w.groupby("instrument"):
            grp = grp.sort_values("date").tail(160)  # 只保留最近160天
            # 计算单股票特征，包含 A/B/C/D 类特征
            row = self._compute_one(inst, feat_ts, grp)
            results.append(row)

        feat_df = pd.DataFrame(results).set_index("instrument")

        # 追加 E 类特征（历史强势信号）
        # 若 n<5 的早返回已将 E 类列置入 feat_df，先删除再 join，避免列名冲突
        e_cols = [PREV_TJ_BOARDS, PREV_NEW_HIGH_COUNT, PREV_POOL_APPEARANCES]
        feat_df = feat_df.drop(columns=[c for c in e_cols if c in feat_df.columns], errors="ignore")
        e_feats = self._compute_strong_pool_features(feat_ts, instruments)
        feat_df = feat_df.join(e_feats, how="left")

        # 追加 F 类特征（估值）
        f_cols = [LOG_FLOAT_CAP, PE_TTM, PB, PS_TTM, PE_SECT_RANK, PB_HIST_RANK]
        feat_df = feat_df.drop(columns=[c for c in f_cols if c in feat_df.columns], errors="ignore")
        f_feats = self._compute_valuation_features(feat_ts, instruments)
        feat_df = feat_df.join(f_feats, how="left")

        # 追加 G 类特征（周K线）
        from features.weekly import compute_weekly_features, WEEKLY_FEATURE_COLS
        g_cols = WEEKLY_FEATURE_COLS
        feat_df = feat_df.drop(columns=[c for c in g_cols if c in feat_df.columns], errors="ignore")
        g_feats = compute_weekly_features(self._kline, feat_date, instruments)
        feat_df = feat_df.join(g_feats, how="left")

        # 追加 H 类特征（概念热度）
        from features.concept import compute_concept_features, CONCEPT_FEATURE_COLS
        h_cols = CONCEPT_FEATURE_COLS
        feat_df = feat_df.drop(columns=[c for c in h_cols if c in feat_df.columns], errors="ignore")
        h_feats = self._compute_concept_features(feat_date, instruments)
        feat_df = feat_df.join(h_feats, how="left")

        # 追加 I 类特征（市场环境）
        from features.market_env import compute_market_env_features, MARKET_ENV_COLS
        i_cols = MARKET_ENV_COLS
        feat_df = feat_df.drop(columns=[c for c in i_cols if c in feat_df.columns], errors="ignore")
        i_feats = compute_market_env_features(
            market_ret=self._market_ret,
            kline=self._kline,
            feat_date=feat_date,
            instruments=instruments,
        )
        feat_df = feat_df.join(i_feats, how="left")

        # 追加 J 类特征（交互特征，依赖其他特征计算完成后再算）
        from features.cross import compute_cross_features, CROSS_FEATURE_COLS
        j_cols = CROSS_FEATURE_COLS
        feat_df = feat_df.drop(columns=[c for c in j_cols if c in feat_df.columns], errors="ignore")
        j_feats = compute_cross_features(feat_df)
        feat_df = feat_df.join(j_feats, how="left")

        # 确保所有特征列存在
        for col in ALL_FEATURE_COLS:
            if col not in feat_df.columns:
                feat_df[col] = np.nan

        return feat_df[ALL_FEATURE_COLS].astype(float)

    # ── 单股票特征计算 ────────────────────────────────────────────────────────

    def _compute_one(
        self,
        instrument: str,
        feat_ts: pd.Timestamp,
        grp: pd.DataFrame,
    ) -> Dict:
        """对单只股票计算 A/B/C/D 类特征（共 28 个）。"""
        row: Dict = {"instrument": instrument}

        close  = grp["close"].values
        volume = grp["volume"].values
        high   = grp["high"].values
        low    = grp["low"].values
        dates  = grp["date"].values
        ma20   = grp["ma20"].values if "ma20" in grp.columns else None

        n = len(close)
        if n < 5:
            for col in ALL_FEATURE_COLS:
                row[col] = np.nan
            return row

        cur_price = float(close[-1])

        # ── 预计算多周期 MA（供多个特征复用）────────────────────────────────
        def _ma(w):
            return float(np.mean(close[-min(w, n):])) if n >= 5 else np.nan

        ma5  = _ma(5)
        ma10 = _ma(10)
        ma60 = _ma(60)

        # ── A 类：趋势斜率 ────────────────────────────────────────────────────

        # A1: 成交量 60 日归一化斜率
        row[VOL_SLOPE_60D] = _normalized_slope(volume[-min(60, n):])

        # A2: 价格超额 60 日斜率（剔除大盘影响，累计超额收益斜率）
        def _excess_slope(window):
            w = min(window + 1, n)
            price_ret = np.diff(close[-w:]) / (close[-w:-1] + 1e-10)
            mkt_ret = self._get_market_ret(dates[-min(window, n):])
            if len(price_ret) >= 5 and len(mkt_ret) == len(price_ret):
                return _normalized_slope(np.cumsum(price_ret - mkt_ret))
            return np.nan

        row[PRICE_SLOPE_60D] = _excess_slope(60)

        # A3: 主力资金流 30 日斜率
        row[MF_SLOPE_30D] = self._compute_mf_slope(instrument, feat_ts)

        # A4: 主力资金流 10 日斜率（短期资金加速信号）
        row[MF_SLOPE_10D] = self._compute_mf_slope_n(instrument, feat_ts, n=10)

        # A5: 价格超额 20 日斜率（短周期趋势）
        row[PRICE_SLOPE_20D] = _excess_slope(20)

        # ── B 类：多周期动量 / 比率变化 ──────────────────────────────────────

        # B1: 量能加速比（近20日均量 / 近60日均量）
        vol_20_mean = volume[-min(20, n):].mean()
        vol_60_mean = volume[-min(60, n):].mean()
        row[VOL_ACCEL] = float(vol_20_mean / vol_60_mean) if vol_60_mean > 0 else np.nan

        # B2: 短期量能爆发（近5日均量 / 近20日均量）
        vol_5_mean = volume[-min(5, n):].mean()
        row[VOL_RATIO_5_20] = float(vol_5_mean / vol_20_mean) if vol_20_mean > 0 else np.nan

        # B3: 振幅收窄（近20日ATR / 近60日ATR，<1 表示震荡收敛=蓄力）
        atr_20 = float(np.mean(high[-min(20, n):] - low[-min(20, n):]))
        atr_60 = float(np.mean(high[-min(60, n):] - low[-min(60, n):]))
        row[RANGE_COMPRESS] = float(atr_20 / atr_60) if atr_60 > 0 else np.nan

        # B4: 动量加速度（近20日收益 - 近60日收益/3）
        if n >= 60:
            r20 = float(close[-1] / close[-20] - 1)
            r60 = float(close[-1] / close[-60] - 1)
            row[MOMENTUM_ACCEL] = r20 - r60 / 3.0
        elif n >= 20:
            row[MOMENTUM_ACCEL] = float(close[-1] / close[-20] - 1)
        else:
            row[MOMENTUM_ACCEL] = np.nan

        # B5: 超额收益变化（近20日超额 - 近60日超额）
        if n >= 60:
            mkt_20 = self._get_market_ret_cumsum(dates[-20:])
            mkt_60 = self._get_market_ret_cumsum(dates[-60:])
            pr_20  = float(close[-1] / close[-20] - 1)
            pr_60  = float(close[-1] / close[-60] - 1)
            row[EXCESS_RET_CHANGE] = (
                (pr_20 - mkt_20) - (pr_60 - mkt_60)
                if mkt_20 is not None and mkt_60 is not None else np.nan
            )
        else:
            row[EXCESS_RET_CHANGE] = np.nan

        # B6-B9: 多周期涨幅
        row[RET_5D]  = float(close[-1] / close[-min(6,  n)] - 1) if n >= 5  else np.nan
        row[RET_10D] = float(close[-1] / close[-min(11, n)] - 1) if n >= 10 else np.nan
        row[RET_20D] = float(close[-1] / close[-min(21, n)] - 1) if n >= 20 else np.nan
        row[RET_60D] = float(close[-1] / close[-min(61, n)] - 1) if n >= 60 else np.nan

        # B10: 布林带宽度（近20日std*4 / MA20，越窄=蓄力状态）
        if n >= 20 and ma20 is not None:
            ma20_cur = float(ma20[-1]) if not np.isnan(ma20[-1]) else None
            if ma20_cur and ma20_cur > 0:
                std20 = float(np.std(close[-20:]))
                row[BB_WIDTH] = std20 * 4 / ma20_cur
            else:
                row[BB_WIDTH] = np.nan
        else:
            row[BB_WIDTH] = np.nan

        # ── C 类：持续时间 / 形态确认 ─────────────────────────────────────────

        # C1: 连续站上 MA20 的天数
        if ma20 is not None:
            above = close >= ma20
            above_clean = above[~np.isnan(ma20)]
            row[DAYS_ABOVE_MA20] = float(_consec_true_from_end(above_clean))
        else:
            row[DAYS_ABOVE_MA20] = np.nan

        # C2: 连续创更高低点（上升通道确认）
        if n >= 3:
            lows = low[-min(30, n):]
            row[CONSEC_HIGHER_LOW] = float(
                _consec_true_from_end(np.array([lows[i] >= lows[i-1] for i in range(1, len(lows))]))
            )
        else:
            row[CONSEC_HIGHER_LOW] = np.nan

        # C3: 主力净流入连续天数
        row[DAYS_POSITIVE_MF] = self._compute_days_positive_mf(instrument, feat_ts)

        # C4: 收盘价位置比（近5日收盘/最高，越高越强势，代表连续阳线）
        if n >= 5:
            h5 = high[-5:]; l5 = low[-5:]; c5 = close[-5:]
            hl_range = h5 - l5
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(hl_range > 0, (c5 - l5) / hl_range, 0.5)
            row[HIGH_CLOSE_RATIO] = float(np.mean(ratio))
        else:
            row[HIGH_CLOSE_RATIO] = np.nan

        # C5: 末端连续上涨天数
        if n >= 2:
            up = close[-min(20, n):][1:] > close[-min(20, n):][:-1]
            row[CONSEC_GREEN] = float(_consec_true_from_end(up))
        else:
            row[CONSEC_GREEN] = np.nan

        # C6: 近20日放量次数（单日成交量 > 20日均量 × 1.5）
        if n >= 20:
            avg20 = float(volume[-20:].mean())
            row[VOL_SPIKE_COUNT] = float(np.sum(volume[-20:] > avg20 * 1.5))
        else:
            row[VOL_SPIKE_COUNT] = np.nan

        # ── D 类：层级对比 ────────────────────────────────────────────────────

        # D1: 距52周高点距离（越小越接近年高）
        high_252 = float(high[-min(252, n):].max())
        row[DIST_52W_HIGH] = float(1 - cur_price / high_252) if high_252 > 0 else np.nan

        # D2: 从60日低点反弹幅度
        low_60 = float(low[-min(60, n):].min())
        row[DIST_60D_LOW_REBOUND] = float(cur_price / low_60 - 1) if low_60 > 0 else np.nan

        # D3: 现价相对MA5偏离度（正值=站上，负值=跌破）
        row[PRICE_VS_MA5] = float((cur_price - ma5) / ma5) if ma5 and ma5 > 0 else np.nan

        # D4: 现价相对MA60偏离度（越高=中长期趋势越强）
        row[PRICE_VS_MA60] = float((cur_price - ma60) / ma60) if ma60 and ma60 > 0 else np.nan

        # D5/D6/D7/D8: 筹码类特征
        chip_row = self._get_chips(instrument, feat_ts)
        if chip_row is not None:
            row[WIN_PERCENT]        = chip_row.get("win_percent", np.nan)
            row[CHIP_CONCENTRATION] = chip_row.get("concentration", np.nan)
            avg_cost = chip_row.get("avg_cost", np.nan)
            row[PRICE_TO_AVGCOST]   = (
                float(cur_price / avg_cost) if avg_cost and avg_cost > 0 else np.nan
            )
            # D8: (close - avg_cost) / avg_cost
            #   0附近 = 价格刚在成本区（套牢盘解套=买入窗口）
            #   大正值 = 价格大幅高于成本（浮盈丰厚=抛压区）
            #   负值   = 价格低于成本（套牢状态）
            row[CHIP_VS_AVG_COST]   = (
                float((cur_price - avg_cost) / avg_cost) if avg_cost and avg_cost > 0 else np.nan
            )
        else:
            row[WIN_PERCENT] = row[CHIP_CONCENTRATION] = row[PRICE_TO_AVGCOST] = np.nan
            row[CHIP_VS_AVG_COST] = np.nan

        return row

    # ── E 类：历史强势信号 ────────────────────────────────────────────────────

    def _compute_strong_pool_features(
        self,
        feat_ts: pd.Timestamp,
        instruments: List[str],
    ) -> pd.DataFrame:
        """
        批量计算 E 类特征（历史强势信号），一次聚合所有股票。

        E 类特征的含义：股票在成为"当前强势股"之前，是否已经有过强势历史。
        这类特征描述的是"惯性"——曾经强势的股票更可能再次强势。

        观察窗口：feat_ts 往前 120 个自然日（约 4 个月）。
        使用自然日而非交易日，是因为 strong_pool 数据按日历日存储，
        且 4 个月足以捕捉到一个完整的板块轮动周期。

        Args:
            feat_ts:     特征提取日期（T_feat），严格排除该日及之后的数据，
                         避免未来数据泄漏（lookahead bias）。
            instruments: 目标股票列表（含正负样本）。

        Returns:
            DataFrame，index=instrument，columns=[prev_tj_boards,
            prev_new_high_count, prev_pool_appearances]。
            不在窗口内出现过的股票填 0.0（而非 NaN），
            表示"没有历史强势记录"而不是"数据缺失"。
        """
        # 观察窗口：feat_ts 往前 120 个自然日（约 4 个月）
        lookback = feat_ts - pd.Timedelta(days=120)

        # 严格用 < feat_ts（不含当天），防止特征提取日期当天进池的信息泄漏
        sp = self._sp[
            (self._sp["date"] >= lookback) &
            (self._sp["date"] <  feat_ts) &
            (self._sp["instrument"].isin(instruments))
        ]

        # 窗口内完全没有强势记录时，所有股票特征值均为 0（非 NaN）
        if sp.empty:
            return pd.DataFrame(
                index=instruments,
                columns=[PREV_TJ_BOARDS, PREV_NEW_HIGH_COUNT, PREV_POOL_APPEARANCES],
                data=0.0,
            )

        agg = (
            sp.groupby("instrument")
            .agg(
                **{
                    # 取窗口内最大涨停板数：反映历史最强一次连板的强度
                    PREV_TJ_BOARDS:        ("tj_boards", "max"),
                    # 出现新高的总次数：次数越多说明上涨持续性越强
                    PREV_NEW_HIGH_COUNT:   ("new_high",  "sum"),
                    # 进入强势池的总天数：频率越高说明反复受资金关注
                    PREV_POOL_APPEARANCES: ("date",      "count"),
                }
            )
        )
        # reindex 补全未出现在强势池的股票，填 0 而非 NaN
        return agg.reindex(instruments, fill_value=0.0)

    # ── 辅助：资金流特征 ──────────────────────────────────────────────────────

    def _compute_mf_slope(
        self,
        instrument: str,
        feat_ts: pd.Timestamp,
    ) -> float:
        """计算主力净流入 30 日归一化斜率（O(1) 字典查找）。"""
        return self._compute_mf_slope_n(instrument, feat_ts, n=30)

    def _compute_mf_slope_n(
        self,
        instrument: str,
        feat_ts: pd.Timestamp,
        n: int = 10,
    ) -> float:
        """计算主力净流入任意 N 日归一化斜率。"""
        grp = self._mf_by_inst.get(instrument)
        if grp is None or len(grp) < 5:
            return np.nan
        mf = grp[grp["date"] <= feat_ts].tail(n)
        if len(mf) < 5:
            return np.nan
        vals = mf["netflow_amount_main"].fillna(0).values
        return _normalized_slope(vals)

    def _compute_days_positive_mf(
        self,
        instrument: str,
        feat_ts: pd.Timestamp,
    ) -> float:
        """计算主力净流入连续正值天数（O(1) 字典查找）。"""
        grp = self._mf_by_inst.get(instrument)
        if grp is None or grp.empty:
            return np.nan
        mf = grp[grp["date"] <= feat_ts].tail(20)
        if mf.empty:
            return np.nan
        positive = (mf["netflow_amount_main"].fillna(0).values > 0)
        return float(_consec_true_from_end(positive))

    def _get_chips(
        self,
        instrument: str,
        feat_ts: pd.Timestamp,
    ) -> Optional[Dict]:
        """获取指定日期最近一条筹码记录（O(1) 字典查找）。"""
        grp = self._chips_by_inst.get(instrument)
        if grp is None or grp.empty:
            return None
        chip = grp[grp["date"] <= feat_ts]
        if chip.empty:
            return None
        return chip.iloc[-1].to_dict()

    # ── 辅助：市场收益率 ──────────────────────────────────────────────────────

    def _get_market_ret(self, dates: np.ndarray) -> np.ndarray:
        """获取指定日期序列的市场日收益率（与 diff 对齐后的序列）。"""
        ts_dates = pd.DatetimeIndex(dates)
        ret = self._market_ret.reindex(ts_dates).fillna(0).values
        return ret

    def _get_market_ret_cumsum(self, dates: np.ndarray) -> Optional[float]:
        """获取日期范围内市场累计收益率。"""
        ts_dates = pd.DatetimeIndex(dates)
        ret = self._market_ret.reindex(ts_dates).fillna(0)
        if ret.empty:
            return None
        return float((1 + ret).prod() - 1)

    # ── F 类：估值特征 ────────────────────────────────────────────────────────

    def _compute_valuation_features(
        self,
        feat_ts: pd.Timestamp,
        instruments: List[str],
    ) -> pd.DataFrame:
        """
        计算 F 类估值特征（point-in-time，只用 <= feat_ts 的最新一条记录）。

        特征列表：
            log_float_cap   : log(流通市值)，捕捉市值量级差异
            pe_ttm          : 市盈率 TTM（负值保留，代表亏损；NaN 代表无数据）
            pb              : 市净率
            ps_ttm          : 市销率 TTM
            pe_sect_rank    : pe_ttm 在截面全市场的百分位（0~1），越低越便宜
                              （使用全市场而非板块，因板块映射需要额外数据）
            pb_hist_rank    : pb 在该股过去 252 日的历史百分位（0~1），
                              越低说明当前市净率相对自身历史偏低（估值低洼）

        Returns:
            DataFrame，index=instrument，columns=F 类特征名
            无估值数据时所有列返回 NaN（不阻断流程）。
        """
        f_cols = [LOG_FLOAT_CAP, PE_TTM, PB, PS_TTM, PE_SECT_RANK, PB_HIST_RANK]

        if not self._val_by_inst:
            return pd.DataFrame(index=instruments, columns=f_cols, dtype=float)

        # ── 取每只股票 <= feat_ts 的最新一条估值快照 ──────────────────────────
        rows = {}
        for inst in instruments:
            grp = self._val_by_inst.get(inst)
            if grp is None or grp.empty:
                continue
            snap = grp[grp["date"] <= feat_ts]
            if snap.empty:
                continue
            rows[inst] = snap.iloc[-1]

        if not rows:
            return pd.DataFrame(index=instruments, columns=f_cols, dtype=float)

        snap_df = pd.DataFrame(rows).T  # index=instrument

        result = pd.DataFrame(index=instruments, columns=f_cols, dtype=float)

        # log_float_cap
        if "float_market_cap" in snap_df.columns:
            fc = pd.to_numeric(snap_df["float_market_cap"], errors="coerce")
            result[LOG_FLOAT_CAP] = np.log1p(fc.clip(lower=0))

        # pe_ttm, pb, ps_ttm（直接取值，保留负数/NaN）
        for src_col, dst_col in [("pe_ttm", PE_TTM), ("pb", PB), ("ps_ttm", PS_TTM)]:
            if src_col in snap_df.columns:
                result[dst_col] = pd.to_numeric(snap_df[src_col], errors="coerce")

        # pe_sect_rank：截面内所有股票 pe_ttm 的百分位（只对正值排名，负=亏损置 NaN）
        if PE_TTM in result.columns:
            pe_pos = result[PE_TTM][result[PE_TTM] > 0]
            if len(pe_pos) > 5:
                ranks = pe_pos.rank(pct=True)
                result[PE_SECT_RANK] = ranks  # 未赋值（亏损/NaN）保持 NaN

        # pb_hist_rank：该股 pb 在其过去 252 日历史中的百分位
        if "pb" in snap_df.columns:
            hist_lookback = feat_ts - pd.Timedelta(days=365)
            for inst in instruments:
                grp = self._val_by_inst.get(inst)
                if grp is None or grp.empty:
                    continue
                hist = grp[
                    (grp["date"] >= hist_lookback) & (grp["date"] <= feat_ts)
                ]["pb"].dropna()
                if len(hist) < 10:
                    continue
                cur_pb = rows.get(inst, {}).get("pb") if isinstance(rows.get(inst), dict) else getattr(rows.get(inst), "pb", None)
                if cur_pb is None or pd.isna(cur_pb):
                    continue
                result.loc[inst, PB_HIST_RANK] = float((hist < float(cur_pb)).mean())

        return result

    def _compute_concept_features(
        self,
        feat_date: str,
        instruments: List[str],
    ) -> pd.DataFrame:
        """
        计算 H 类概念热度特征（委托 features/concept.py）。

        self._concept_bar        : concept_bar1d 全量数据（由构造方或 compute 注入）
        self._concept_comp_range : concept_component 快照（由构造方或 compute 注入）

        两个属性若不存在则返回全 NaN（不阻断流程）。
        """
        from features.concept import compute_concept_features, CONCEPT_FEATURE_COLS
        h_cols = CONCEPT_FEATURE_COLS
        empty = pd.DataFrame(index=instruments, columns=h_cols, dtype=float)

        concept_bar = getattr(self, "_concept_bar", None)
        concept_comp = getattr(self, "_concept_comp_range", None)
        if concept_bar is None or concept_comp is None:
            logger.debug("H类: _concept_bar 或 _concept_comp_range 未加载，跳过")
            return empty

        try:
            return compute_concept_features(
                concept_bar=concept_bar,
                concept_comp_range=concept_comp,
                feat_date=feat_date,
                instruments=instruments,
            )
        except Exception as exc:
            logger.warning(f"H类概念特征计算失败: {exc}")
            return empty
