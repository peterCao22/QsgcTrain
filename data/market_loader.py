"""
全市场数据加载模块

职责（point-in-time 严格对齐，只能使用 ≤ t 日收盘后可得数据）：
- load_kline()            日线价量数据（复权）
- load_chips()            筹码分布数据（avg_cost / win_percent / concentration）
- load_moneyflow()        个股资金流数据
- load_sector_moneyflow() 板块资金流（概念/行业）
- load_dragon_seats()     龙虎榜（近N日上榜 & 机构净买入）
- load_price_limit_status() 涨跌停 / 停牌状态
- load_index_kline()      大盘指数日K线（来自 index_bar1d 表）
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from loguru import logger

from data.db_loader import read_sql, read_sql_chunked


# ─── K线数据 ─────────────────────────────────────────────────────────────────

def load_kline(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载日线价量数据。

    Returns:
        DataFrame with columns:
            date, instrument, open, high, low, close, volume, amount,
            turn (换手率%), pctChg, pre_close, ma5, ma10, ma20, ma60,
            tradestatus, isST
    """
    default_cols = [
        "date", "instrument", "open", "high", "low", "close",
        "volume", "amount", "turn", "change_ratio", "pre_close",
        "ma5", "ma10", "ma20", "ma60",
    ]
    cols = columns or default_cols
    col_str = ", ".join(cols)

    instrument_clause = ""
    params = {"start": start_date, "end": end_date}
    if instruments:
        params["instruments"] = list(instruments)
        instrument_clause = "AND instrument = ANY(:instruments)"

    sql = f"""
        SELECT {col_str}
        FROM kline_all
        WHERE date >= :start
          AND date <= :end
          {instrument_clause}
        ORDER BY date, instrument
    """
    df = read_sql_chunked(sql, params)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(
        f"kline loaded: {len(df):,} rows  "
        f"[{start_date} ~ {end_date}]"
        + (f"  instruments={len(instruments)}" if instruments else "")
    )
    return df


# ─── 筹码数据 ─────────────────────────────────────────────────────────────────

def load_chips(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载筹码分布数据。

    Returns:
        date, instrument, avg_cost, win_percent, concentration
    """
    instrument_clause = ""
    params = {"start": start_date, "end": end_date}
    if instruments:
        params["instruments"] = list(instruments)
        instrument_clause = "AND instrument = ANY(:instruments)"

    sql = f"""
        SELECT date, instrument, avg_cost, win_percent, concentration
        FROM chips_all
        WHERE date >= :start
          AND date <= :end
          {instrument_clause}
        ORDER BY date, instrument
    """
    df = read_sql_chunked(sql, params)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"chips loaded: {len(df):,} rows")
    return df


# ─── 个股资金流 ───────────────────────────────────────────────────────────────

def load_moneyflow(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载个股资金流数据。

    主要字段：
        netflow_amount_main     主力净流入金额（元）
        netflow_amount_rate_main  主力净流入占比
        active_buy_amount_all   全量主动买入金额
    """
    instrument_clause = ""
    params = {"start": start_date, "end": end_date}
    if instruments:
        params["instruments"] = list(instruments)
        instrument_clause = "AND instrument = ANY(:instruments)"

    sql = f"""
        SELECT date, instrument,
               netflow_amount_main,
               netflow_amount_large,
               netflow_amount_rate_main,
               inflow_amount_rate_main,
               outflow_amount_rate_main,
               active_buy_amount_all
        FROM moneyflow
        WHERE date >= :start
          AND date <= :end
          {instrument_clause}
        ORDER BY date, instrument
    """
    df = read_sql_chunked(sql, params)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"moneyflow loaded: {len(df):,} rows")
    return df


# ─── 板块资金流 ───────────────────────────────────────────────────────────────

def load_sector_moneyflow(
    start_date: str,
    end_date: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    加载板块（概念/行业）资金流数据。

    表名：concept_bar1d（旧项目迁移过来的概念板块日线数据）
    如果表不存在，返回空 DataFrame 而不抛异常（降级处理）。

    Returns:
        date, concept_code, concept_name, pct_change, net_amount, ...
    """
    try:
        params = {"start": start_date, "end": end_date}
        sql = """
            SELECT date, concept_code, concept_name,
                   pct_change, net_amount, amount,
                   rise_count, fall_count
            FROM concept_bar1d
            WHERE date >= :start
              AND date <= :end
            ORDER BY date, net_amount DESC NULLS LAST
        """
        df = read_sql_chunked(sql, params)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"sector moneyflow loaded: {len(df):,} rows")
        return df
    except Exception as exc:
        logger.warning(f"sector moneyflow unavailable ({exc}); returning empty DataFrame")
        return pd.DataFrame()


# ─── 板块成分映射 ─────────────────────────────────────────────────────────────

def load_sector_component(
    date: str,
) -> pd.DataFrame:
    """
    加载指定日期的股票→概念板块映射。

    Returns:
        instrument, concept_code, concept_name
    """
    try:
        params = {"date": date}
        sql = """
            SELECT instrument, concept_code, concept_name
            FROM concept_component
            WHERE date = :date
        """
        df = read_sql(sql, params)
        logger.info(f"sector component on {date}: {len(df):,} rows")
        return df
    except Exception as exc:
        logger.warning(f"sector component unavailable ({exc})")
        return pd.DataFrame()


def load_concept_component_range(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    加载日期范围内所有 concept_component 快照（约每周一次，共约 97 个快照）。

    concept_component 是周期性快照，不是每日数据。调用方需自行处理
    "取最近快照"的逻辑（对任意 T 日，使用 <= T 的最新快照）。

    Returns:
        DataFrame，列：date（快照日）, concept_code, instrument
        按 date, concept_code, instrument 升序排列。
        如表不存在或无数据，返回空 DataFrame。
    """
    try:
        sql = """
            SELECT date, concept_code, instrument
            FROM concept_component
            WHERE date >= :start AND date <= :end
            ORDER BY date, concept_code, instrument
        """
        df = read_sql_chunked(sql, {"start": start_date, "end": end_date})
        df["date"] = pd.to_datetime(df["date"])
        n_snaps = df["date"].nunique()
        logger.info(
            f"concept_component loaded: {len(df):,} rows  "
            f"{n_snaps} snapshots  [{start_date} ~ {end_date}]"
        )
        return df
    except Exception as exc:
        logger.warning(f"concept_component unavailable ({exc}); returning empty DataFrame")
        return pd.DataFrame()


# ─── 估值数据 ─────────────────────────────────────────────────────────────────

def load_valuation(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载个股财务估值数据（valuation_all 表）。

    字段：date, instrument, pe_ttm, pb, ps_ttm, pcf_net_ttm,
          total_market_cap, float_market_cap

    point-in-time 对齐：调用方应只取 <= T_feat 的最新一条记录，
    避免使用未来财务数据。

    Returns:
        DataFrame，按 date, instrument 升序排列。
        表不存在或无数据时返回空 DataFrame。
    """
    try:
        instrument_clause = ""
        params: dict = {"start": start_date, "end": end_date}
        if instruments:
            params["instruments"] = list(instruments)
            instrument_clause = "AND instrument = ANY(:instruments)"

        sql = f"""
            SELECT date, instrument,
                   pe_ttm, pb, ps_ttm, pcf_net_ttm,
                   total_market_cap, float_market_cap
            FROM valuation_all
            WHERE date >= :start AND date <= :end
            {instrument_clause}
            ORDER BY date, instrument
        """
        df = read_sql_chunked(sql, params)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(
            f"valuation loaded: {len(df):,} rows  "
            f"[{start_date} ~ {end_date}]  "
            f"instruments={len(df['instrument'].unique()) if not df.empty else 0}"
        )
        return df
    except Exception as exc:
        logger.warning(f"valuation_all unavailable ({exc}); returning empty DataFrame")
        return pd.DataFrame()


# ─── 龙虎榜 ───────────────────────────────────────────────────────────────────

def load_dragon_seats(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载龙虎榜数据（dragon_seats 表，由旧项目迁移）。

    关键字段：
        date, instrument,
        inst_net_buy      机构席位净买入（元）
        total_net_buy     全席位净买入（元）
        reason            上榜原因

    如果表不存在，返回空 DataFrame（降级处理）。
    """
    try:
        instrument_clause = ""
        params = {"start": start_date, "end": end_date}
        if instruments:
            params["instruments"] = list(instruments)
            instrument_clause = "AND instrument = ANY(:instruments)"

        sql = f"""
            SELECT date, instrument,
                   SUM(CASE WHEN seat_type = 'institution' THEN net_buy ELSE 0 END) AS inst_net_buy,
                   SUM(net_buy) AS total_net_buy,
                   MAX(reason) AS reason
            FROM dragon_seats
            WHERE date >= :start
              AND date <= :end
              {instrument_clause}
            GROUP BY date, instrument
            ORDER BY date, instrument
        """
        df = read_sql_chunked(sql, params)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"dragon seats loaded: {len(df):,} rows")
        return df
    except Exception as exc:
        logger.warning(f"dragon seats unavailable ({exc}); returning empty DataFrame")
        return pd.DataFrame()


# ─── 涨跌停 / 停牌状态 ───────────────────────────────────────────────────────

def load_price_limit_status(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载涨跌停 / 停牌 / ST 状态。

    Returns:
        date, instrument, price_limit_status, suspended, st_status
    """
    instrument_clause = ""
    params = {"start": start_date, "end": end_date}
    if instruments:
        params["instruments"] = list(instruments)
        instrument_clause = "AND instrument = ANY(:instruments)"

    try:
        sql = f"""
            SELECT date, instrument, price_limit_status, suspended, st_status
            FROM price_limit_status
            WHERE date >= :start
              AND date <= :end
              {instrument_clause}
            ORDER BY date, instrument
        """
        df = read_sql_chunked(sql, params)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:
        logger.warning(f"price_limit_status unavailable ({exc})")
        return pd.DataFrame()


# ─── 大盘指数日K线 ────────────────────────────────────────────────────────────

def load_index_kline(
    start_date: str,
    end_date: str,
    instruments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载大盘指数日K线数据（来自 index_bar1d 表）。

    表需先通过 scripts/import_index_bar1d.py 导入。
    若表不存在或无数据，返回空 DataFrame（调用方降级处理）。

    Args:
        start_date:  起始日期（含），格式 'YYYY-MM-DD'
        end_date:    结束日期（含），格式 'YYYY-MM-DD'
        instruments: 指定指数代码列表，如 ['000001.SH']；
                     None 表示加载全部指数

    Returns:
        DataFrame，列：date, instrument, name, close, change_ratio，
        按 date 升序排列。若表不存在则返回空 DataFrame。
    """
    try:
        instrument_clause = ""
        params = {"start": start_date, "end": end_date}
        if instruments:
            if len(instruments) == 1:
                # 单元素时用 = :inst，避免 ANY(tuple) 的 psycopg2 格式问题
                params["inst"] = instruments[0]
                instrument_clause = "AND instrument = :inst"
            else:
                params["instruments"] = list(instruments)
                instrument_clause = "AND instrument = ANY(:instruments)"

        sql = f"""
            SELECT date, instrument, name, pre_close, open, high, low,
                   close, volume, amount, change, change_ratio
            FROM index_bar1d
            WHERE date >= :start
              AND date <= :end
              {instrument_clause}
            ORDER BY date, instrument
        """
        df = read_sql_chunked(sql, params)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(
            f"index_bar1d loaded: {len(df):,} rows [{start_date} ~ {end_date}]"
            + (f"  instruments={instruments}" if instruments else "")
        )
        return df
    except Exception as exc:
        logger.warning(f"index_bar1d unavailable ({exc}); returning empty DataFrame")
        return pd.DataFrame()

