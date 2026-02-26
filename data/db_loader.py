"""
PostgreSQL 数据加载基础模块

职责：
- 管理数据库连接（单例引擎）
- 提供通用 SQL 查询方法
- 加载交易日历
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DB_URL


# ─── 连接管理（线程安全单例）──────────────────────────────────────────────────

_engine: Optional[Engine] = None
_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create_engine(
                    DB_URL,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                logger.info(f"DB engine created: {DB_URL.split('@')[-1]}")
    return _engine


def read_sql(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """执行 SQL 并返回 DataFrame，params 使用 :key 占位符。"""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def read_sql_chunked(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """分批读取大结果集，合并后返回。"""
    engine = get_engine()
    chunks: List[pd.DataFrame] = []
    with engine.connect() as conn:
        for chunk in pd.read_sql(text(sql), conn, params=params or {}, chunksize=chunksize):
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ─── 交易日历 ────────────────────────────────────────────────────────────────

_trading_days_cache: Optional[pd.DatetimeIndex] = None


def get_trading_days(
    start: str = "2020-01-01",
    end: str = "2030-12-31",
) -> pd.DatetimeIndex:
    """
    从 kline_all 表取出所有实际交易日期（去重、排序）。
    结果缓存在内存中，调用多次不重复查询。
    """
    global _trading_days_cache
    if _trading_days_cache is not None:
        mask = (_trading_days_cache >= pd.Timestamp(start)) & (
            _trading_days_cache <= pd.Timestamp(end)
        )
        return _trading_days_cache[mask]

    sql = """
        SELECT DISTINCT date
        FROM kline_all
        ORDER BY date
    """
    df = read_sql(sql)
    df["date"] = pd.to_datetime(df["date"])
    _trading_days_cache = pd.DatetimeIndex(df["date"].sort_values())
    logger.info(f"Trading calendar loaded: {len(_trading_days_cache)} days")
    mask = (_trading_days_cache >= pd.Timestamp(start)) & (
        _trading_days_cache <= pd.Timestamp(end)
    )
    return _trading_days_cache[mask]


def offset_trading_day(date: Union[str, pd.Timestamp], n: int) -> pd.Timestamp:
    """
    从 date 向后（n>0）或向前（n<0）偏移 n 个交易日。
    date 自身必须是交易日，否则从最近交易日开始计。
    """
    cal = get_trading_days()
    ts = pd.Timestamp(date)
    idx = cal.searchsorted(ts)
    # 若 date 不在日历中，取最近前一个交易日
    if idx >= len(cal) or cal[idx] != ts:
        idx = max(0, idx - 1)
    target_idx = idx + n
    if target_idx < 0 or target_idx >= len(cal):
        raise IndexError(
            f"Offset {n} from {date} goes out of trading calendar range."
        )
    return cal[target_idx]


# ─── 股票列表 ─────────────────────────────────────────────────────────────────

def get_stock_list() -> pd.DataFrame:
    """返回 stock_list 表：columns=[instrument, name]。"""
    return read_sql("SELECT instrument, name FROM stock_list ORDER BY instrument")


# ─── 全市场有效股票域（已应用基础过滤）────────────────────────────────────────

def get_tradable_universe(
    date: str,
    exclude_st: bool = True,
    exclude_suspended: bool = True,
    min_price: float = 1.0,
) -> List[str]:
    """
    返回指定日期的可交易股票列表（排除 ST / 停牌 / 仙股）。

    过滤逻辑（全部 AND）：
    1. 科创板 (688xxx)、北交所 (8xxxxx / 4xxxxx) 在 config 层已通过
       instrument 前缀排除（由调用方在特征构建时处理）。
    2. ST / 退市：price_limit_status.st_status = FALSE
    3. 停牌：price_limit_status.suspended = FALSE
    4. 仙股：kline_all.close >= min_price
    """
    params: Dict[str, Any] = {"date": date, "min_price": min_price}
    # price_limit_status 只记录特殊状态股票（ST/停牌），未出现的行说明状态正常（NULL 视为 False）
    st_clause = "AND (pls.st_status IS NULL OR pls.st_status = FALSE)" if exclude_st else ""
    sus_clause = "AND (pls.suspended IS NULL OR pls.suspended = FALSE)" if exclude_suspended else ""

    sql = f"""
        SELECT k.instrument
        FROM kline_all k
        LEFT JOIN price_limit_status pls
            ON k.instrument = pls.instrument AND k.date = pls.date
        WHERE k.date = :date
          AND k.close >= :min_price
          {st_clause}
          {sus_clause}
        ORDER BY k.instrument
    """
    df = read_sql(sql, params)
    return df["instrument"].tolist()
