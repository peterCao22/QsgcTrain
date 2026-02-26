"""
导入大盘指数日K线数据到 index_bar1d 表

CSV 列格式（与 index_bar1d.csv 对齐）：
    date, instrument, name, pre_close, open, high, low, close,
    volume, amount, change, change_ratio

支持多个指数（上证/深证/沪深300等），按 instrument 区分。

运行方式：
    conda activate rqsdk
    cd D:\\myCursor\\QsgcTrain

    # 指定 CSV 路径
    python scripts/import_index_bar1d.py --csv data/index_bar1d.csv

    # 重建表（清空后重新导入）
    python scripts/import_index_bar1d.py --csv data/index_bar1d.csv --drop

    # 查看表统计
    python scripts/import_index_bar1d.py --stats
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
CREATE TABLE IF NOT EXISTS index_bar1d (
    id           SERIAL        PRIMARY KEY,
    date         DATE          NOT NULL,
    instrument   VARCHAR(20)   NOT NULL,       -- 指数代码（如 000001.SH）
    name         VARCHAR(50),                  -- 指数名称
    pre_close    DECIMAL(12,4),                -- 昨收
    open         DECIMAL(12,4),
    high         DECIMAL(12,4),
    low          DECIMAL(12,4),
    close        DECIMAL(12,4)  NOT NULL,
    volume       BIGINT,                       -- 成交量（手）
    amount       DECIMAL(20,2),               -- 成交额（元）
    change       DECIMAL(12,4),               -- 涨跌额
    change_ratio DECIMAL(10,6),               -- 涨跌幅（小数，非百分比）
    created_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, instrument)
);

CREATE INDEX IF NOT EXISTS idx_index_bar1d_date       ON index_bar1d(date);
CREATE INDEX IF NOT EXISTS idx_index_bar1d_instrument ON index_bar1d(instrument);
CREATE INDEX IF NOT EXISTS idx_index_bar1d_date_inst  ON index_bar1d(date, instrument);
"""

DROP_TABLE_SQL = "DROP TABLE IF EXISTS index_bar1d CASCADE;"

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
            logger.info("index_bar1d 表已删除")
        cur.execute(CREATE_TABLE_SQL)
        logger.info("index_bar1d 表创建/确认完成")
    conn.close()

# ─── 读取 CSV ─────────────────────────────────────────────────────────────────

# CSV 列名到数据库列名的映射（兼容大小写差异）
COL_RENAME = {
    "Date": "date", "Instrument": "instrument", "Name": "name",
    "Pre_close": "pre_close", "Open": "open", "High": "high",
    "Low": "low", "Close": "close", "Volume": "volume",
    "Amount": "amount", "Change": "change", "Change_ratio": "change_ratio",
}

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取 index_bar1d.csv，标准化列名和日期格式。

    Args:
        csv_path: CSV 文件路径

    Returns:
        清洗后的 DataFrame，列顺序与数据库一致
    """
    logger.info(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path)

    # 标准化列名（首字母大写 → 小写）
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})
    df.columns = [c.lower() for c in df.columns]

    # 必须列校验
    required = {"date", "instrument", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}，实际列：{list(df.columns)}")

    # 日期标准化
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # change_ratio：有些数据源给百分比（如 1.23），有些给小数（0.0123）
    # 统一存为小数；若绝对值 > 1，认为是百分比格式，自动除以 100
    if "change_ratio" in df.columns:
        df["change_ratio"] = pd.to_numeric(df["change_ratio"], errors="coerce")
        mask = df["change_ratio"].abs() > 1
        df.loc[mask, "change_ratio"] = df.loc[mask, "change_ratio"] / 100

    logger.info(
        f"CSV 读取完成：{len(df):,} 行  "
        f"日期范围：{df['date'].min()} ~ {df['date'].max()}  "
        f"指数：{df['instrument'].unique().tolist()}"
    )
    return df

# ─── 写入数据库 ───────────────────────────────────────────────────────────────

UPSERT_SQL = """
    INSERT INTO index_bar1d (
        date, instrument, name, pre_close, open, high, low, close,
        volume, amount, change, change_ratio
    ) VALUES %s
    ON CONFLICT (date, instrument) DO UPDATE SET
        name         = EXCLUDED.name,
        pre_close    = EXCLUDED.pre_close,
        open         = EXCLUDED.open,
        high         = EXCLUDED.high,
        low          = EXCLUDED.low,
        close        = EXCLUDED.close,
        volume       = EXCLUDED.volume,
        amount       = EXCLUDED.amount,
        change       = EXCLUDED.change,
        change_ratio = EXCLUDED.change_ratio
"""

def upsert_df(df: pd.DataFrame) -> int:
    """
    批量 UPSERT 到 index_bar1d 表。

    Returns:
        实际写入行数
    """
    def _val(row, col):
        v = row.get(col)
        return None if pd.isna(v) else v

    values = [
        (
            row["date"], row["instrument"],
            _val(row, "name"),
            _val(row, "pre_close"), _val(row, "open"),
            _val(row, "high"),      _val(row, "low"),
            row["close"],
            _val(row, "volume"),    _val(row, "amount"),
            _val(row, "change"),    _val(row, "change_ratio"),
        )
        for _, row in df.iterrows()
    ]

    conn = get_conn()
    try:
        with conn:
            psycopg2.extras.execute_values(
                conn.cursor(), UPSERT_SQL, values, page_size=500
            )
        logger.info(f"写入 {len(values):,} 行到 index_bar1d")
        return len(values)
    finally:
        conn.close()

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM index_bar1d")
    total = cur.fetchone()[0]
    cur.execute("SELECT MIN(date), MAX(date) FROM index_bar1d")
    mn, mx = cur.fetchone()
    cur.execute(
        "SELECT instrument, name, COUNT(*) FROM index_bar1d "
        "GROUP BY instrument, name ORDER BY instrument"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(
        f"\n{'='*50}\n"
        f"index_bar1d 统计：\n"
        f"  总记录数：{total:,}\n"
        f"  日期范围：{mn} ~ {mx}\n"
        f"  指数列表：\n" +
        "\n".join(f"    {r[0]}  {r[1]}  {r[2]:,} 条" for r in rows) +
        f"\n{'='*50}"
    )

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="导入指数日K线到 index_bar1d 表")
    parser.add_argument("--csv",   default=None, help="CSV 文件路径")
    parser.add_argument("--drop",  action="store_true", help="重建表（清空已有数据）")
    parser.add_argument("--stats", action="store_true", help="只查看统计，不导入")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if not args.csv:
        parser.error("请指定 --csv 文件路径，例如：--csv data/index_bar1d.csv")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"文件不存在：{csv_path}")
        return

    create_table(drop=args.drop)
    df = load_csv(csv_path)
    n = upsert_df(df)
    print(f"\n导入完成：{n:,} 条记录写入 index_bar1d 表")
    print_stats()


if __name__ == "__main__":
    main()
