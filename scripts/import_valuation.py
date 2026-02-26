"""
导入个股财务估值数据到 valuation_all 表

CSV 列格式：
    date, instrument, pe_ttm, pb, ps_ttm, pcf_net_ttm,
    total_market_cap, float_market_cap

数据来源：D:\\myCursor\\AlphaSignalCN-Standalone\\data\\raw\\financial\\valuation_all.csv
数据范围：2021-01-04 ~ 2026

运行方式：
    conda activate rqsdk
    cd D:\\myCursor\\QsgcTrain

    # 导入（文件 703MB，分块写入，约需 5~10 分钟）
    python scripts/import_valuation.py --csv "D:\\myCursor\\AlphaSignalCN-Standalone\\data\\raw\\financial\\valuation_all.csv"

    # 重建表后导入
    python scripts/import_valuation.py --csv "..." --drop

    # 查看统计
    python scripts/import_valuation.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import psycopg2
import psycopg2.extras
from loguru import logger

from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

# ─── 表结构 ───────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS valuation_all (
    id               SERIAL        PRIMARY KEY,
    date             DATE          NOT NULL,
    instrument       VARCHAR(20)   NOT NULL,
    pe_ttm           FLOAT,                    -- 市盈率（TTM）
    pb               FLOAT,                    -- 市净率
    ps_ttm           FLOAT,                    -- 市销率（TTM）
    pcf_net_ttm      FLOAT,                    -- 市现率（TTM，净现金流）
    total_market_cap FLOAT,                    -- 总市值（元）
    float_market_cap FLOAT,                    -- 流通市值（元）
    created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_valuation_date       ON valuation_all(date);
CREATE INDEX IF NOT EXISTS idx_valuation_instrument ON valuation_all(instrument);
CREATE INDEX IF NOT EXISTS idx_valuation_date_inst  ON valuation_all(date, instrument);
"""

DROP_TABLE_SQL = "DROP TABLE IF EXISTS valuation_all CASCADE;"

UPSERT_SQL = """
    INSERT INTO valuation_all (
        date, instrument, pe_ttm, pb, ps_ttm, pcf_net_ttm,
        total_market_cap, float_market_cap
    ) VALUES %s
    ON CONFLICT (date, instrument) DO UPDATE SET
        pe_ttm           = EXCLUDED.pe_ttm,
        pb               = EXCLUDED.pb,
        ps_ttm           = EXCLUDED.ps_ttm,
        pcf_net_ttm      = EXCLUDED.pcf_net_ttm,
        total_market_cap = EXCLUDED.total_market_cap,
        float_market_cap = EXCLUDED.float_market_cap
"""

# ─── 数据库连接 ───────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT,
        user=POSTGRES_USER,  password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
    )

# ─── 建表 ─────────────────────────────────────────────────────────────────────

def create_table(drop: bool = False) -> None:
    conn = get_conn()
    with conn:
        cur = conn.cursor()
        if drop:
            cur.execute(DROP_TABLE_SQL)
            logger.info("valuation_all 表已删除")
        cur.execute(CREATE_TABLE_SQL)
        logger.info("valuation_all 表创建/确认完成")
    conn.close()

# ─── 分块导入 ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 200_000   # 每批 20 万行，703MB 约 45 批

def _nan_to_none(v):
    return None if pd.isna(v) else float(v)

def import_csv(csv_path: Path) -> int:
    """
    分块读取 CSV 并 UPSERT 到 valuation_all 表。

    文件 703MB，约 1700 万行，分批处理避免内存溢出。
    Returns:
        总写入行数
    """
    logger.info(f"开始读取：{csv_path}  (分块大小 {CHUNK_SIZE:,})")

    total_written = 0
    chunk_idx = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        dtype={
            "instrument": str,
            "pe_ttm": float, "pb": float, "ps_ttm": float,
            "pcf_net_ttm": float, "total_market_cap": float, "float_market_cap": float,
        },
    )

    conn = get_conn()
    try:
        for chunk in reader:
            chunk_idx += 1

            # 标准化列名（防止大小写差异）
            chunk.columns = [c.strip().lower() for c in chunk.columns]

            # 日期格式
            chunk["date"] = pd.to_datetime(chunk["date"]).dt.date

            # 排除全估值字段都是 NaN 的行（无意义数据）
            val_cols = ["pe_ttm", "pb", "ps_ttm", "pcf_net_ttm", "total_market_cap", "float_market_cap"]
            chunk = chunk.dropna(subset=val_cols, how="all")

            if chunk.empty:
                continue

            values = [
                (
                    row["date"],
                    row["instrument"],
                    _nan_to_none(row.get("pe_ttm")),
                    _nan_to_none(row.get("pb")),
                    _nan_to_none(row.get("ps_ttm")),
                    _nan_to_none(row.get("pcf_net_ttm")),
                    _nan_to_none(row.get("total_market_cap")),
                    _nan_to_none(row.get("float_market_cap")),
                )
                for _, row in chunk.iterrows()
            ]

            with conn:
                psycopg2.extras.execute_values(
                    conn.cursor(), UPSERT_SQL, values, page_size=1000
                )

            total_written += len(values)
            logger.info(
                f"  批次 {chunk_idx}: {len(values):,} 行  "
                f"累计 {total_written:,} 行  "
                f"日期 {chunk['date'].min()} ~ {chunk['date'].max()}"
            )
    finally:
        conn.close()

    return total_written

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM valuation_all")
    total = cur.fetchone()[0]
    cur.execute("SELECT MIN(date), MAX(date) FROM valuation_all")
    mn, mx = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT instrument) FROM valuation_all")
    n_inst = cur.fetchone()[0]
    cur.execute("""
        SELECT
            SUM(CASE WHEN pe_ttm IS NOT NULL THEN 1 ELSE 0 END)::float / COUNT(*) as pe_rate,
            SUM(CASE WHEN pb IS NOT NULL THEN 1 ELSE 0 END)::float / COUNT(*) as pb_rate,
            SUM(CASE WHEN float_market_cap IS NOT NULL THEN 1 ELSE 0 END)::float / COUNT(*) as cap_rate
        FROM valuation_all
    """)
    rates = cur.fetchone()
    cur.close()
    conn.close()

    logger.info(
        f"\n{'='*55}\n"
        f"valuation_all 统计：\n"
        f"  总记录数：{total:,}\n"
        f"  日期范围：{mn} ~ {mx}\n"
        f"  股票数量：{n_inst:,}\n"
        f"  字段覆盖率：pe_ttm={rates[0]*100:.1f}%  pb={rates[1]*100:.1f}%  float_cap={rates[2]*100:.1f}%\n"
        f"{'='*55}"
    )

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="导入个股估值数据到 valuation_all 表")
    parser.add_argument("--csv",   default=None, help="CSV 文件路径")
    parser.add_argument("--drop",  action="store_true", help="重建表（清空已有数据）")
    parser.add_argument("--stats", action="store_true", help="只查看统计，不导入")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if not args.csv:
        parser.error("请指定 --csv 文件路径")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"文件不存在：{csv_path}")
        return

    create_table(drop=args.drop)
    n = import_csv(csv_path)

    print(f"\n导入完成：共写入 {n:,} 条记录到 valuation_all 表")
    print_stats()


if __name__ == "__main__":
    main()
