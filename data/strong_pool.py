"""
强势股池加载模块

支持三种来源（由 config.STRONG_POOL_SOURCE 控制）：
1. "local"  — 读取本地 CSV 快照（离线回测首选）
2. "zhitu"  — 智兔 API 实时拉取
3. "moma"   — 魔码云服 API 实时拉取

CSV 快照格式（data/strong_pool_snapshots/YYYY-MM-DD.csv）：
    date,instrument
    2025-01-20,000001.SZ
    2025-01-20,600519.SH
    ...

StrongPoolLoader.get(date) 返回指定日期的股票代码列表（List[str]）。
StrongPoolLoader.get_snapshot_dates() 返回所有可用快照日期列表。
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger

from config import (
    MOMA_API_KEY,
    MOMA_API_URL,
    STRONG_POOL_DIR,
    STRONG_POOL_SOURCE,
    ZHITU_API_KEY,
    ZHITU_API_URL,
)


class StrongPoolLoader:
    """强势股池加载器（支持本地快照 / 智兔 API / 魔码云服 API）。"""

    def __init__(self, source: str = STRONG_POOL_SOURCE):
        self.source = source
        self._cache: Dict[str, List[str]] = {}

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def get(self, date: str) -> List[str]:
        """
        获取指定日期的强势股池股票代码列表。

        Args:
            date: 日期字符串，格式 'YYYY-MM-DD'

        Returns:
            股票代码列表，如 ['000001.SZ', '600519.SH']
        """
        if date in self._cache:
            return self._cache[date]

        if self.source == "local":
            stocks = self._load_local(date)
        elif self.source == "zhitu":
            stocks = self._load_zhitu(date)
        elif self.source == "moma":
            stocks = self._load_moma(date)
        else:
            raise ValueError(f"Unknown strong pool source: {self.source!r}")

        self._cache[date] = stocks
        logger.info(f"Strong pool [{self.source}] @ {date}: {len(stocks)} stocks")
        return stocks

    def get_snapshot_dates(self) -> List[str]:
        """
        返回本地快照目录中所有可用日期（仅 local 模式有效）。
        日期按升序排列，格式 'YYYY-MM-DD'。
        """
        if self.source != "local":
            logger.warning("get_snapshot_dates() only works in 'local' mode")
            return []

        dates = []
        for p in sorted(STRONG_POOL_DIR.glob("*.csv")):
            stem = p.stem  # e.g. '2025-01-20'
            try:
                datetime.strptime(stem, "%Y-%m-%d")
                dates.append(stem)
            except ValueError:
                pass
        return dates

    def save_snapshot(self, date: str, instruments: List[str]) -> Path:
        """
        将强势股池保存为本地 CSV 快照（实时模式辅助存档用）。

        Returns:
            保存的文件路径
        """
        STRONG_POOL_DIR.mkdir(parents=True, exist_ok=True)
        path = STRONG_POOL_DIR / f"{date}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "instrument"])
            for inst in instruments:
                writer.writerow([date, inst])
        logger.info(f"Strong pool snapshot saved: {path}")
        return path

    def load_all_snapshots(self) -> pd.DataFrame:
        """
        批量加载所有本地快照，返回合并的 DataFrame。

        Returns:
            DataFrame: columns=[date (datetime), instrument]
        """
        dates = self.get_snapshot_dates()
        if not dates:
            logger.warning("No local snapshots found in %s", STRONG_POOL_DIR)
            return pd.DataFrame(columns=["date", "instrument"])

        frames = []
        for d in dates:
            stocks = self._load_local(d)
            if stocks:
                df = pd.DataFrame({"date": pd.Timestamp(d), "instrument": stocks})
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["date", "instrument"])

        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values("date").reset_index(drop=True)

    # ── 本地 CSV ──────────────────────────────────────────────────────────────

    def _load_local(self, date: str) -> List[str]:
        path = STRONG_POOL_DIR / f"{date}.csv"
        if not path.exists():
            logger.warning(f"Local snapshot not found: {path}")
            return []

        df = pd.read_csv(path)
        if "instrument" not in df.columns:
            logger.error(f"Column 'instrument' missing in {path}")
            return []
        return df["instrument"].dropna().astype(str).tolist()

    # ── 智兔 API ──────────────────────────────────────────────────────────────

    def _load_zhitu(self, date: str) -> List[str]:
        """
        调用智兔 API 获取强势股池。

        请根据实际 API 文档调整 endpoint / 参数 / 响应解析。
        占位实现：返回空列表并记录警告。
        """
        if not ZHITU_API_URL or not ZHITU_API_KEY:
            logger.error("ZHITU_API_URL / ZHITU_API_KEY not configured")
            return []

        url = f"{ZHITU_API_URL.rstrip('/')}/strong_pool"
        headers = {"Authorization": f"Bearer {ZHITU_API_KEY}"}
        params = {"date": date}

        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # 期望结构：{"stocks": ["000001.SZ", ...]}
            # 根据实际 API 文档修改以下解析逻辑
            stocks: List[str] = data.get("stocks", [])
            return stocks
        except Exception as exc:
            logger.error(f"Zhitu API error: {exc}")
            return []

    # ── 魔码云服 API ──────────────────────────────────────────────────────────

    def _load_moma(self, date: str) -> List[str]:
        """
        调用魔码云服 API 获取强势股池。

        请根据实际 API 文档调整 endpoint / 参数 / 响应解析。
        占位实现：返回空列表并记录警告。
        """
        if not MOMA_API_URL or not MOMA_API_KEY:
            logger.error("MOMA_API_URL / MOMA_API_KEY not configured")
            return []

        url = f"{MOMA_API_URL.rstrip('/')}/strong_pool"
        headers = {"X-API-Key": MOMA_API_KEY}
        params = {"date": date, "format": "json"}

        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # 期望结构：{"list": [{"code": "000001.SZ"}, ...]}
            # 根据实际 API 文档修改以下解析逻辑
            stocks: List[str] = [item["code"] for item in data.get("list", [])]
            return stocks
        except Exception as exc:
            logger.error(f"Moma API error: {exc}")
            return []


# ─── 便捷函数 ─────────────────────────────────────────────────────────────────

_default_loader: Optional[StrongPoolLoader] = None


def get_strong_pool(date: str) -> List[str]:
    """使用全局默认 loader 获取强势股池（单次调用入口）。"""
    global _default_loader
    if _default_loader is None:
        _default_loader = StrongPoolLoader()
    return _default_loader.get(date)
