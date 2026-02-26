"""
创建 strong_pool 表

运行方式：
    python scripts/create_strong_pool_table.py
    python scripts/create_strong_pool_table.py --drop  # 先删再建（危险！会清空数据）
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import psycopg2
from loguru import logger

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER


def get_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
    )


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS strong_pool (
    id           SERIAL          PRIMARY KEY,
    date         DATE            NOT NULL,
    instrument   VARCHAR(20)     NOT NULL,
    name         VARCHAR(100),
    price        DECIMAL(12,4),
    limit_price  DECIMAL(12,4),
    pct_change   DECIMAL(10,4),
    amount       DECIMAL(20,2),
    float_cap    DECIMAL(20,2),
    total_cap    DECIMAL(20,2),
    speed        DECIMAL(10,4),
    new_high     SMALLINT        DEFAULT 0,
    vol_ratio    DECIMAL(10,4),
    turnover     DECIMAL(10,4),
    tj_days      SMALLINT,
    tj_boards    SMALLINT,
    source       VARCHAR(20)     DEFAULT 'api',
    created_at   TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, instrument)
);

COMMENT ON TABLE strong_pool IS '强势股池日快照：每天突破新高或连续涨停的强势股列表';
COMMENT ON COLUMN strong_pool.date        IS '快照日期';
COMMENT ON COLUMN strong_pool.instrument  IS '股票代码（标准格式，如 600693.SH）';
COMMENT ON COLUMN strong_pool.name        IS '股票名称';
COMMENT ON COLUMN strong_pool.price       IS '当日价格';
COMMENT ON COLUMN strong_pool.limit_price IS '涨停价';
COMMENT ON COLUMN strong_pool.pct_change  IS '当日涨幅%';
COMMENT ON COLUMN strong_pool.amount      IS '成交额（元）';
COMMENT ON COLUMN strong_pool.float_cap   IS '流通市值（元）';
COMMENT ON COLUMN strong_pool.total_cap   IS '总市值（元）';
COMMENT ON COLUMN strong_pool.speed       IS '涨速（实时）';
COMMENT ON COLUMN strong_pool.new_high    IS '是否突破新高（0=否, 1=是）';
COMMENT ON COLUMN strong_pool.vol_ratio   IS '量比';
COMMENT ON COLUMN strong_pool.turnover    IS '换手率%';
COMMENT ON COLUMN strong_pool.tj_days     IS '涨停统计：观察期天数（tj字段X天）';
COMMENT ON COLUMN strong_pool.tj_boards   IS '涨停统计：涨停板次数（tj字段Y板）';
COMMENT ON COLUMN strong_pool.source      IS '数据来源（api/manual）';

CREATE INDEX IF NOT EXISTS idx_sp_date            ON strong_pool(date);
CREATE INDEX IF NOT EXISTS idx_sp_instrument      ON strong_pool(instrument);
CREATE INDEX IF NOT EXISTS idx_sp_date_instrument ON strong_pool(date, instrument);
CREATE INDEX IF NOT EXISTS idx_sp_new_high        ON strong_pool(new_high);
CREATE INDEX IF NOT EXISTS idx_sp_tj_boards       ON strong_pool(tj_boards);
"""

DROP_TABLE_SQL = "DROP TABLE IF EXISTS strong_pool CASCADE;"


def create_table(drop_first: bool = False) -> None:
    conn = get_conn()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        if drop_first:
            logger.warning("Dropping existing strong_pool table...")
            cur.execute(DROP_TABLE_SQL)
            logger.info("Old table dropped.")

        logger.info("Creating strong_pool table...")
        cur.execute(CREATE_TABLE_SQL)
        logger.info("✓ strong_pool table created successfully.")

        # 验证
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'strong_pool'
            ORDER BY ordinal_position
        """)
        cols = cur.fetchall()
        logger.info(f"Table structure ({len(cols)} columns):")
        for col_name, col_type in cols:
            logger.info(f"  {col_name:<18} {col_type}")

        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'strong_pool'")
        idxs = cur.fetchall()
        logger.info(f"Indexes ({len(idxs)}):")
        for (idx_name,) in idxs:
            logger.info(f"  {idx_name}")

    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="创建 strong_pool 数据库表")
    parser.add_argument("--drop", action="store_true", help="先删除已有表再重建（危险！会清空数据）")
    args = parser.parse_args()

    if args.drop:
        confirm = input("⚠️  将删除并重建 strong_pool 表，所有数据会丢失。确认? (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("已取消。")
            return

    create_table(drop_first=args.drop)


if __name__ == "__main__":
    main()
