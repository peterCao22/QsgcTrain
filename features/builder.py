"""
特征构建器（Feature Builder）

职责：
1. 协调 price_volume / chip / sector 三个特征模块
2. 对特征做横截面标准化（rank / zscore 可选）
3. 单特征 Spearman 筛查（训练时过滤低信噪比特征）
4. 输出 (X, instruments) 供模型使用

主要接口：
    FeatureBuilder.build(date, universe)   → 单截面特征 DataFrame
    FeatureBuilder.build_panel(dates, universe, label_df)
        → 多截面面板特征 + label，供 Walk-forward 训练使用
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

from config import (
    EXCLUDE_PREFIXES,
    FEATURE_MIN_SPEARMAN,
    LABEL_COL,
    MIN_LISTING_DAYS,
    MIN_PRICE,
)
from data.market_loader import (
    load_chips,
    load_dragon_seats,
    load_kline,
    load_moneyflow,
    load_price_limit_status,
    load_sector_component,
    load_sector_moneyflow,
)
from features.chip import compute_chip_features
from features.price_volume import compute_price_volume_features
from features.sector import compute_sector_features


class FeatureBuilder:
    """
    特征构建器。

    使用方式（典型）：
        builder = FeatureBuilder(start="2023-01-01", end="2025-12-31")
        panel = builder.build_panel(dates=weekly_dates, label_df=labels)
    """

    def __init__(
        self,
        start: str,
        end: str,
        normalize: str = "rank",  # "rank" | "zscore" | "none"
        min_spearman: float = FEATURE_MIN_SPEARMAN,
    ):
        """
        Args:
            start:         数据加载起始日期（须早于第一个截面日期足够多，如 252 日）
            end:           数据加载结束日期
            normalize:     特征标准化方式（'rank' = 横截面分位数排名，推荐）
            min_spearman:  单特征筛查阈值（仅在 build_panel 的 filter_features 步骤生效）
        """
        self.start = start
        self.end = end
        self.normalize = normalize
        self.min_spearman = min_spearman

        # 延迟加载的缓存数据（全局一次性拉取，跨截面共享）
        self._kline: Optional[pd.DataFrame] = None
        self._chips: Optional[pd.DataFrame] = None
        self._moneyflow: Optional[pd.DataFrame] = None
        self._sector_mf: Optional[pd.DataFrame] = None
        self._price_limit: Optional[pd.DataFrame] = None

        # 筛查后保留的特征列名（首次 filter_features 后缓存）
        self._valid_features: Optional[List[str]] = None

    # ── 数据预加载 ────────────────────────────────────────────────────────────

    def preload(self) -> None:
        """一次性加载所有数据到内存（推荐在 build_panel 前调用）。"""
        logger.info(f"Preloading market data [{self.start} ~ {self.end}]...")
        self._kline = load_kline(self.start, self.end)
        self._chips = load_chips(self.start, self.end)
        self._moneyflow = load_moneyflow(self.start, self.end)
        self._sector_mf = load_sector_moneyflow(self.start, self.end)
        logger.info("Preload complete.")

    # ── 单截面特征 ───────────────────────────────────────────────────────────

    def build(
        self,
        date: str,
        universe: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        构建单个截面日期的特征矩阵。

        Args:
            date:      截面日期 'YYYY-MM-DD'
            universe:  股票代码列表；为 None 时自动从数据中取当日有效标的。

        Returns:
            DataFrame: index=instrument, columns=特征列（已标准化）
        """
        kline = self._kline if self._kline is not None else load_kline(
            _offset_days(date, -300), date
        )
        chips = self._chips if self._chips is not None else load_chips(
            _offset_days(date, -30), date
        )
        moneyflow = self._moneyflow if self._moneyflow is not None else load_moneyflow(
            _offset_days(date, -20), date
        )
        sector_mf = self._sector_mf if self._sector_mf is not None else load_sector_moneyflow(
            _offset_days(date, -30), date
        )
        sector_comp = load_sector_component(date)

        # 确定当日有效标的
        if universe is None:
            ts = pd.Timestamp(date)
            kline_today = kline[kline["date"] == ts]
            universe = _filter_universe(kline_today)

        if not universe:
            logger.warning(f"Empty universe on {date}")
            return pd.DataFrame()

        # 各子特征
        pv_df = compute_price_volume_features(kline, date, universe)
        chip_df = compute_chip_features(chips, kline, date, universe)
        sec_df = compute_sector_features(
            kline, sector_mf, sector_comp, moneyflow, date, universe
        )

        # 合并
        dfs = [df for df in [pv_df, chip_df, sec_df] if not df.empty]
        if not dfs:
            return pd.DataFrame()

        feat_df = dfs[0]
        for d in dfs[1:]:
            feat_df = feat_df.join(d, how="outer")

        feat_df = feat_df.loc[feat_df.index.isin(universe)]
        feat_df = feat_df.fillna(0.0)

        # 标准化
        if self.normalize == "rank":
            feat_df = _rank_normalize(feat_df)
        elif self.normalize == "zscore":
            feat_df = _zscore_normalize(feat_df)

        return feat_df

    # ── 多截面面板特征 ────────────────────────────────────────────────────────

    def build_panel(
        self,
        dates: List[str],
        label_df: pd.DataFrame,
        filter_features: bool = True,
    ) -> pd.DataFrame:
        """
        批量构建面板特征，并与 label_df 合并。

        Args:
            dates:           截面日期列表（升序）
            label_df:        标签 DataFrame，须包含 [date, instrument, LABEL_COL]
            filter_features: True 时对特征做单特征 Spearman 筛查

        Returns:
            面板 DataFrame：[date, instrument, feat1, ..., featN, LABEL_COL]
        """
        if self._kline is None:
            self.preload()

        label_df = label_df.copy()
        label_df["date"] = pd.to_datetime(label_df["date"])

        frames: List[pd.DataFrame] = []

        for date in tqdm(dates, desc="Building features"):
            feat_df = self.build(date)
            if feat_df.empty:
                continue

            feat_df = feat_df.reset_index()  # instrument → column
            feat_df["date"] = pd.Timestamp(date)

            # 合并 label
            date_labels = label_df[label_df["date"] == pd.Timestamp(date)][
                ["instrument", LABEL_COL]
            ]
            if date_labels.empty:
                continue

            merged = feat_df.merge(date_labels, on="instrument", how="inner")
            if merged.empty:
                continue

            frames.append(merged)

        if not frames:
            logger.warning("build_panel produced no data")
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)

        if filter_features:
            panel = self._filter_by_spearman(panel)

        logger.info(
            f"Panel built: {len(panel):,} rows × {len(panel.columns)} cols  "
            f"({len(dates)} dates)"
        )
        return panel

    # ── 特征有效性筛查 ────────────────────────────────────────────────────────

    def _filter_by_spearman(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个特征与 LABEL_COL 的横截面 Spearman 均值，
        移除 |mean_spearman| < min_spearman 的特征。
        """
        if LABEL_COL not in panel.columns:
            return panel

        meta_cols = {"date", "instrument", LABEL_COL}
        feat_cols = [c for c in panel.columns if c not in meta_cols]

        spearman_map: Dict[str, float] = {}
        for col in feat_cols:
            # 按截面日期分别计算 Spearman，再取均值
            srs = []
            for _, gdf in panel.groupby("date"):
                valid = gdf[[col, LABEL_COL]].dropna()
                if len(valid) < 10:
                    continue
                corr, _ = spearmanr(valid[col], valid[LABEL_COL])
                if not np.isnan(corr):
                    srs.append(corr)
            mean_corr = float(np.mean(srs)) if srs else 0.0
            spearman_map[col] = mean_corr

        valid_feats = [
            col for col, corr in spearman_map.items()
            if abs(corr) >= self.min_spearman
        ]
        removed = [c for c in feat_cols if c not in valid_feats]

        # 日志输出筛查结果
        logger.info(
            f"Feature Spearman filter: "
            f"{len(valid_feats)}/{len(feat_cols)} kept, {len(removed)} removed"
        )
        if removed:
            logger.debug(f"  Removed: {removed}")
        _log_feature_spearman(spearman_map)

        self._valid_features = valid_feats
        keep_cols = ["date", "instrument"] + valid_feats + [LABEL_COL]
        return panel[[c for c in keep_cols if c in panel.columns]]

    def get_feature_columns(self) -> Optional[List[str]]:
        """返回上次 filter_features 后保留的特征列名。"""
        return self._valid_features


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _filter_universe(kline_today: pd.DataFrame) -> List[str]:
    """从当日 kline 中过滤出有效标的（排除科创/北交所/ST/仙股）。"""
    if kline_today.empty:
        return []

    df = kline_today.copy()
    # 排除科创（688xxx）、北交所（8xxxxx / 4xxxxx）
    for prefix in EXCLUDE_PREFIXES:
        df = df[~df["instrument"].str.startswith(prefix)]

    # 排除 ST
    if "isST" in df.columns:
        df = df[df["isST"] != 1]

    # 排除仙股 & 停牌
    if "close" in df.columns:
        df = df[df["close"] >= MIN_PRICE]

    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == 1]

    return df["instrument"].tolist()


def _rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """横截面分位数排名（0~1），保留 NaN 为 0。"""
    ranked = df.rank(axis=0, pct=True, na_option="bottom")
    return ranked.fillna(0.0)


def _zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score 标准化，Winsorize 在 ±3σ。"""
    mean = df.mean(axis=0)
    std = df.std(axis=0).replace(0, 1.0)
    z = (df - mean) / std
    return z.clip(-3, 3).fillna(0.0)


def _offset_days(date: str, days: int) -> str:
    """粗略日期偏移（用于确定数据拉取范围），不需要严格交易日。"""
    ts = pd.Timestamp(date) + pd.Timedelta(days=days)
    return ts.strftime("%Y-%m-%d")


def _log_feature_spearman(spearman_map: Dict[str, float]) -> None:
    """打印特征 Spearman 排行。"""
    sorted_feats = sorted(spearman_map.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = ["Feature Spearman ranking (top 20):"]
    for col, corr in sorted_feats[:20]:
        flag = "✓" if abs(corr) >= FEATURE_MIN_SPEARMAN else "✗"
        lines.append(f"  {flag} {col:<40} {corr:+.4f}")
    logger.info("\n".join(lines))
