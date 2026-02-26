"""
标签生成模块：30日收盘收益

核心设计原则（解决旧项目根因问题）：
1. Point-in-time 严格对齐：标签 label(t) = Close(t+30) / Close(t) - 1
   其中 t 日的 Close 为截面日期当天收盘价（已知），
   t+30 日的 Close 在 t 日时属于"未来"，模型训练时利用，预测时不可用。
2. 可买性约束（可选）：过滤 t+1 日涨停/停牌导致无法买入的样本。
3. 横截面排名标签（用于 LambdaRank）：将收益率转为组内分位数排名。

主要接口：
    LabelGenerator.generate(dates, universe)  → 面板标签 DataFrame
    LabelGenerator.add_rank_label(df)         → 追加横截面排名列
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config import LABEL_COL, LABEL_FORWARD_DAYS
from data.db_loader import get_trading_days, offset_trading_day
from data.market_loader import load_kline, load_price_limit_status

RANK_LABEL_COL = "rank_label"   # 组内分位数排名（0~1），用于 LambdaRank


class LabelGenerator:
    """30日收盘收益标签生成器。"""

    def __init__(
        self,
        forward_days: int = LABEL_FORWARD_DAYS,
        apply_buyability_filter: bool = True,
        data_start: Optional[str] = None,
        data_end: Optional[str] = None,
    ):
        """
        Args:
            forward_days:            向前多少个交易日计算收盘收益（默认30）。
            apply_buyability_filter: 是否过滤 t+1 日不可买的样本（涨停/停牌）。
            data_start:              kline 数据加载起始日（默认自动推算）。
            data_end:                kline 数据加载结束日（默认自动推算）。
        """
        self.forward_days = forward_days
        self.apply_buyability_filter = apply_buyability_filter
        self._data_start = data_start
        self._data_end = data_end
        self._kline_cache: Optional[pd.DataFrame] = None
        self._pls_cache: Optional[pd.DataFrame] = None

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def generate(
        self,
        dates: List[str],
        instruments: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        为一组截面日期批量生成标签。

        Args:
            dates:       截面日期列表（str 'YYYY-MM-DD'，即 t 日）
            instruments: 目标股票列表；None 表示不限制。

        Returns:
            DataFrame 列：[date, instrument, fwd_return_30d, rank_label (可选)]
            - date 为截面日期 t
            - fwd_return_30d = Close(t+30) / Close(t) - 1
            - 若 t+30 日数据不存在（超出数据库范围），该行被剔除
        """
        # 确定数据加载范围
        start, end = self._resolve_date_range(dates)

        # 加载 kline（一次）
        logger.info(f"Loading kline for labels [{start} ~ {end}]...")
        self._kline_cache = load_kline(start, end, instruments=instruments)
        self._kline_cache["date"] = pd.to_datetime(self._kline_cache["date"])

        if self.apply_buyability_filter:
            self._pls_cache = load_price_limit_status(start, end, instruments=instruments)
            if not self._pls_cache.empty:
                self._pls_cache["date"] = pd.to_datetime(self._pls_cache["date"])

        records = []
        cal = get_trading_days(start, end)

        for date in tqdm(dates, desc="Generating labels"):
            ts = pd.Timestamp(date)

            # 找到 t+forward_days 日
            try:
                t_plus_n = offset_trading_day(date, self.forward_days)
            except IndexError:
                logger.debug(f"Label: t+{self.forward_days} out of range for {date}, skipping")
                continue

            # 获取 t 日和 t+N 日的收盘价
            close_t = self._get_close(ts, instruments)
            close_t_n = self._get_close(t_plus_n, instruments)

            if close_t.empty or close_t_n.empty:
                continue

            # 合并
            merged = close_t.rename(columns={"close": "close_t"}).merge(
                close_t_n.rename(columns={"close": "close_tN"}),
                on="instrument",
                how="inner",
            )
            if merged.empty:
                continue

            merged[LABEL_COL] = (
                merged["close_tN"] / merged["close_t"].replace(0, np.nan) - 1
            )
            merged["date"] = ts

            # 过滤极端值（超过 ±200% 视为数据异常）
            merged = merged[merged[LABEL_COL].abs() <= 2.0]

            # 可买性过滤：t+1 日不能买（涨停/停牌）则从训练集中剔除
            if self.apply_buyability_filter and self._pls_cache is not None and not self._pls_cache.empty:
                try:
                    t_plus_1 = offset_trading_day(date, 1)
                    merged = self._filter_buyability(merged, t_plus_1)
                except IndexError:
                    pass

            records.append(merged[["date", "instrument", LABEL_COL]])

        if not records:
            logger.warning("Label generation produced no data")
            return pd.DataFrame(columns=["date", "instrument", LABEL_COL, RANK_LABEL_COL])

        result = pd.concat(records, ignore_index=True)
        result = self.add_rank_label(result)

        logger.info(
            f"Labels generated: {len(result):,} rows  "
            f"[{dates[0]} ~ {dates[-1]}]  "
            f"mean_return={result[LABEL_COL].mean():.3f}"
        )
        return result

    @staticmethod
    def add_rank_label(df: pd.DataFrame) -> pd.DataFrame:
        """
        在面板 DataFrame 中追加横截面分位数排名列 `rank_label`（0~1）。
        排名越高 = 收益越高（即 rank=1 表示当期收益最高的股票）。
        """
        if LABEL_COL not in df.columns:
            return df
        df = df.copy()
        df[RANK_LABEL_COL] = df.groupby("date")[LABEL_COL].rank(pct=True)
        return df

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _get_close(
        self,
        ts: pd.Timestamp,
        instruments: Optional[List[str]],
    ) -> pd.DataFrame:
        """从缓存 kline 中取指定日期的收盘价，返回 [instrument, close]。"""
        df = self._kline_cache[self._kline_cache["date"] == ts][["instrument", "close"]]
        if instruments:
            df = df[df["instrument"].isin(instruments)]
        return df.dropna(subset=["close"])

    def _filter_buyability(
        self,
        df: pd.DataFrame,
        buy_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """过滤在 buy_date（t+1）涨停或停牌的股票。"""
        pls = self._pls_cache
        if pls is None or pls.empty:
            return df

        buy_status = pls[pls["date"] == buy_date][
            ["instrument", "price_limit_status", "suspended"]
        ]
        if buy_status.empty:
            return df

        # 涨停当日买不进
        limit_up_insts = buy_status[
            buy_status["price_limit_status"].isin(["limit_up", "涨停"])
        ]["instrument"].tolist()

        # 停牌当日买不进
        suspended_insts = buy_status[
            buy_status["suspended"] == True
        ]["instrument"].tolist()

        exclude = set(limit_up_insts) | set(suspended_insts)
        if exclude:
            before = len(df)
            df = df[~df["instrument"].isin(exclude)]
            logger.debug(
                f"Buyability filter @ {buy_date.date()}: "
                f"removed {before - len(df)} rows"
            )
        return df

    def _resolve_date_range(self, dates: List[str]) -> tuple[str, str]:
        """推算数据加载的起止范围（需覆盖 t+forward_days）。"""
        start = self._data_start or _offset_cal_days(dates[0], -10)
        end = self._data_end or _offset_cal_days(dates[-1], self.forward_days + 10)
        return start, end


def _offset_cal_days(date: str, days: int) -> str:
    """日历日偏移（用于确定数据加载范围，不要求严格交易日）。"""
    ts = pd.Timestamp(date) + pd.Timedelta(days=days * 1.5)  # 留足缓冲
    return ts.strftime("%Y-%m-%d")
