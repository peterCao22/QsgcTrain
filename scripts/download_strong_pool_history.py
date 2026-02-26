"""
下载历史强势股池数据（近1年）并存入 PostgreSQL

API 说明：
  智兔:    GET https://api.zhituapi.com/hs/pool/qsgc/{date}?token={token}
  魔码云服: GET http://api.momaapi.com/hslt/qsgc/{date}/{token}

  每个 API 每天限额 200 次，两个 API 交替使用（每20次切换），
  合计 400 次/天，足以覆盖全年 ~250 个交易日。

运行方式：
    # 下载近1年数据（默认）
    python scripts/download_strong_pool_history.py

    # 指定日期范围
    python scripts/download_strong_pool_history.py --start 2024-01-01 --end 2025-01-01

    # 只下载指定日期（快速测试）
    python scripts/download_strong_pool_history.py --date 2025-01-20

    # 强制重新下载（覆盖已有数据）
    python scripts/download_strong_pool_history.py --force

    # 查看当前数据库中的统计
    python scripts/download_strong_pool_history.py --stats
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras
from loguru import logger

from config import (
    MOMA_API_KEY,
    MOMA_API_URL,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    ZHITU_API_KEY,
    ZHITU_API_URL,
)

# ─── 常量 ─────────────────────────────────────────────────────────────────────

# 每个 API 连续调用多少次后切换到另一个
SWITCH_EVERY = 20

# 两个 API 每天各最多使用次数（留 20 次余量）
MAX_PER_API_PER_DAY = 180

# 请求超时（秒）
REQUEST_TIMEOUT = 20


# ─── 数据库 ─────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
    )


def get_downloaded_dates() -> set:
    """获取 strong_pool 表中已有数据的日期集合（断点续传）。"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT date FROM strong_pool ORDER BY date")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {str(r[0]) for r in rows}
    except Exception as e:
        logger.warning(f"无法读取已下载日期: {e}")
        return set()


def upsert_records(records: List[Dict]) -> int:
    """批量写入（UPSERT）strong_pool 表，返回实际写入行数。"""
    if not records:
        return 0

    sql = """
        INSERT INTO strong_pool (
            date, instrument, name, price, limit_price, pct_change,
            amount, float_cap, total_cap, speed, new_high,
            vol_ratio, turnover, tj_days, tj_boards, source
        ) VALUES %s
        ON CONFLICT (date, instrument) DO UPDATE SET
            name        = EXCLUDED.name,
            price       = EXCLUDED.price,
            limit_price = EXCLUDED.limit_price,
            pct_change  = EXCLUDED.pct_change,
            amount      = EXCLUDED.amount,
            float_cap   = EXCLUDED.float_cap,
            total_cap   = EXCLUDED.total_cap,
            speed       = EXCLUDED.speed,
            new_high    = EXCLUDED.new_high,
            vol_ratio   = EXCLUDED.vol_ratio,
            turnover    = EXCLUDED.turnover,
            tj_days     = EXCLUDED.tj_days,
            tj_boards   = EXCLUDED.tj_boards,
            source      = EXCLUDED.source
    """
    values = [
        (
            r["date"], r["instrument"], r["name"],
            r["price"], r["limit_price"], r["pct_change"],
            r["amount"], r["float_cap"], r["total_cap"],
            r["speed"], r["new_high"], r["vol_ratio"], r["turnover"],
            r["tj_days"], r["tj_boards"], r["source"],
        )
        for r in records
    ]

    conn = get_conn()
    try:
        with conn:
            psycopg2.extras.execute_values(conn.cursor(), sql, values, page_size=200)
        return len(values)
    finally:
        conn.close()


def print_stats() -> None:
    """打印 strong_pool 表的当前统计信息。"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM strong_pool")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT date) FROM strong_pool")
        days = cur.fetchone()[0]
        cur.execute("SELECT MIN(date), MAX(date) FROM strong_pool")
        min_d, max_d = cur.fetchone()
        cur.execute("SELECT source, COUNT(*) FROM strong_pool GROUP BY source ORDER BY source")
        by_source = cur.fetchall()
        cur.close()
        conn.close()

        logger.info(
            f"\n{'='*50}\n"
            f"strong_pool 表统计：\n"
            f"  总记录数：{total:,} 条\n"
            f"  日期范围：{min_d} ~ {max_d}（{days} 个交易日）\n"
            f"  按来源：\n"
            + "\n".join(f"    {src}: {cnt:,}" for src, cnt in by_source)
            + f"\n{'='*50}"
        )
    except Exception as e:
        logger.error(f"统计失败: {e}")


# ─── 字段解析 ────────────────────────────────────────────────────────────────

def parse_instrument(dm: str) -> str:
    """
    将 API 返回的代码格式转换为标准格式。
      sh600693 → 600693.SH
      sz000001 → 000001.SZ
    """
    dm = dm.strip().lower()
    if dm.startswith("sh"):
        return dm[2:].upper() + ".SH"
    elif dm.startswith("sz"):
        return dm[2:].upper() + ".SZ"
    else:
        # 兜底：6开头是沪市，其余是深市
        code = dm.upper()
        if code.startswith("6"):
            return code + ".SH"
        return code + ".SZ"


def parse_tj(tj_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    解析 tj 字段 "X天/Y板" 或 "X/Y"。
      "18/10" → (18, 10)
      "0/0"   → (0, 0)
      ""      → (None, None)
    """
    if not tj_str or str(tj_str).strip() in ("", "null", "None"):
        return None, None
    try:
        s = str(tj_str).replace("天", "").replace("板", "").strip()
        if "/" in s:
            parts = s.split("/")
            return int(parts[0]), int(parts[1])
        return None, None
    except (ValueError, IndexError, AttributeError):
        return None, None


def parse_new_high(v) -> int:
    """
    解析 nh（是否新高）字段，兼容多种格式：
      1 / '1' / True  → 1
      '是' / 'yes' / 'true' → 1
      其余（包括 0 / '' / '否' / None）→ 0
    """
    if v is None or v == "" or v == "null":
        return 0
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "是", "y"):
        return 1
    return 0


def parse_record(item: Dict, date: str, source: str) -> Optional[Dict]:
    """将 API 返回的单条记录转换为数据库记录格式。"""
    try:
        dm = str(item.get("dm", "")).strip()
        if not dm:
            return None

        tj_days, tj_boards = parse_tj(item.get("tj", ""))

        return {
            "date":        date,
            "instrument":  parse_instrument(dm),
            "name":        str(item.get("mc", "") or ""),
            "price":       _safe_float(item.get("p")),
            "limit_price": _safe_float(item.get("ztp")),
            "pct_change":  _safe_float(item.get("zf")),
            "amount":      _safe_float(item.get("cje")),
            "float_cap":   _safe_float(item.get("lt")),
            "total_cap":   _safe_float(item.get("zsz")),
            "speed":       _safe_float(item.get("zs")),
            "new_high":    parse_new_high(item.get("nh")),
            "vol_ratio":   _safe_float(item.get("lb")),
            "turnover":    _safe_float(item.get("hs")),
            "tj_days":     tj_days,
            "tj_boards":   tj_boards,
            "source":      source,
        }
    except Exception as e:
        logger.debug(f"解析记录失败 {item}: {e}")
        return None


def _safe_float(v) -> Optional[float]:
    """安全转换为 float，None/空值返回 None。"""
    if v is None or v == "" or v == "null":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── API 调用 ─────────────────────────────────────────────────────────────────

def fetch_zhitu(date: str, client: httpx.Client) -> List[Dict]:
    """
    智兔 API：GET https://api.zhituapi.com/hs/pool/qsgc/{date}?token={token}

    响应格式（根据样本推断）：
      直接返回数组：[{"dm": "sh600693", "mc": "...", ...}, ...]
      或包装对象：{"data": [...], "code": 0}
    """
    if not ZHITU_API_KEY:
        raise ValueError("ZHITU_API_KEY 未配置")

    url = f"{ZHITU_API_URL.rstrip('/')}/hs/pool/qsgc/{date}"
    params = {"token": ZHITU_API_KEY}

    resp = client.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    # 兼容直接返回数组 或 包装对象
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "list", "result", "stocks"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def fetch_moma(date: str, client: httpx.Client) -> List[Dict]:
    """
    魔码云服 API：GET http://api.momaapi.com/hslt/qsgc/{date}/{token}

    响应格式（根据样本推断）：
      直接返回数组：[{"dm": "sh600693", ...}, ...]
      或包装对象：{"data": [...]}
    """
    if not MOMA_API_KEY:
        raise ValueError("MOMA_API_KEY 未配置")

    url = f"{MOMA_API_URL.rstrip('/')}/hslt/qsgc/{date}/{MOMA_API_KEY}"

    resp = client.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "list", "result", "stocks"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


# ─── API 轮询调度器 ───────────────────────────────────────────────────────────

class ApiScheduler:
    """
    两个 API 交替使用调度器。

    规则：每 SWITCH_EVERY 次请求切换一次 API，
    每个 API 每天不超过 MAX_PER_API_PER_DAY 次。
    """

    def __init__(self):
        self.zhitu_count = 0
        self.moma_count  = 0
        self.total_count = 0
        self._current    = "zhitu"   # 当前使用的 API
        self._batch_count = 0        # 当前批次已用次数

    @property
    def current_source(self) -> str:
        return self._current

    def fetch(self, date: str, client: httpx.Client) -> Tuple[List[Dict], str]:
        """
        自动选择当前可用的 API，发起请求并返回 (记录列表, 使用的 source)。
        如果两个 API 都超限，抛出 RuntimeError。
        """
        # 当前批次已够，尝试切换
        if self._batch_count >= SWITCH_EVERY:
            self._switch()

        # 检查当前 API 是否还有配额
        if not self._current_has_quota():
            self._switch()
            if not self._current_has_quota():
                raise RuntimeError(
                    f"两个 API 今日配额均已用尽（智兔：{self.zhitu_count}，"
                    f"魔码：{self.moma_count}，各限 {MAX_PER_API_PER_DAY} 次）"
                )

        source = self._current
        if source == "zhitu":
            items = fetch_zhitu(date, client)
            self.zhitu_count += 1
        else:
            items = fetch_moma(date, client)
            self.moma_count += 1

        self.total_count  += 1
        self._batch_count += 1
        return items, source

    def _switch(self):
        """切换到另一个 API。"""
        self._current = "moma" if self._current == "zhitu" else "zhitu"
        self._batch_count = 0
        logger.debug(f"API 切换 → {self._current}")

    def _current_has_quota(self) -> bool:
        if self._current == "zhitu":
            return self.zhitu_count < MAX_PER_API_PER_DAY
        return self.moma_count < MAX_PER_API_PER_DAY

    def summary(self) -> str:
        return (
            f"API 使用统计：智兔 {self.zhitu_count} 次 / "
            f"魔码云服 {self.moma_count} 次 / "
            f"合计 {self.total_count} 次"
        )


# ─── 交易日历 ─────────────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str) -> List[str]:
    """
    从 kline_all 表获取指定范围内的交易日列表。
    如果数据库无数据，回退到工作日（周一至周五）估算。
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT date FROM kline_all WHERE date >= %s AND date <= %s ORDER BY date",
            (start, end),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if rows:
            logger.debug(f"交易日历来自 kline_all：{len(rows)} 天")
            return [str(r[0]) for r in rows]
    except Exception as e:
        logger.warning(f"无法从 DB 获取交易日历：{e}，改用工作日估算")

    days = pd.bdate_range(start=start, end=end)
    logger.info(f"使用工作日估算：{len(days)} 天")
    return [d.strftime("%Y-%m-%d") for d in days]


# ─── 主下载流程 ───────────────────────────────────────────────────────────────

def download_history(
    start: str,
    end: str,
    force: bool = False,
    sleep_seconds: float = 0.3,
) -> None:
    """
    批量下载历史强势股池数据。

    Args:
        start:         起始日期（含），格式 'YYYY-MM-DD'
        end:           结束日期（含），格式 'YYYY-MM-DD'
        force:         True 时强制重新下载，忽略已有记录
        sleep_seconds: 每次 API 请求后的等待时间
    """
    # 验证 API 配置
    if not ZHITU_API_KEY and not MOMA_API_KEY:
        logger.error("请在 .env 中配置 ZHITU_API_KEY 或 MOMA_API_KEY")
        return

    trading_days = get_trading_days(start, end)
    logger.info(f"目标交易日：{len(trading_days)} 天  [{start} ~ {end}]")

    # 断点续传
    if not force:
        downloaded = get_downloaded_dates()
        pending = [d for d in trading_days if d not in downloaded]
        skipped = len(trading_days) - len(pending)
        if skipped > 0:
            logger.info(f"已下载 {skipped} 天（跳过），待下载 {len(pending)} 天")
    else:
        pending = trading_days
        logger.info(f"强制模式：下载全部 {len(pending)} 天")

    if not pending:
        logger.info("所有日期均已下载，无需重复操作。")
        print_stats()
        return

    # 估算是否超出每日配额
    total_quota = MAX_PER_API_PER_DAY * 2
    if len(pending) > total_quota:
        logger.warning(
            f"待下载 {len(pending)} 天 > 今日总配额 {total_quota} 次，"
            f"今天最多下载 {total_quota} 天，剩余明天继续（断点续传）"
        )

    scheduler = ApiScheduler()
    total_records = 0
    success_days  = 0
    empty_days    = 0
    fail_days     = 0
    fail_dates    = []

    with httpx.Client() as client:
        for i, date in enumerate(pending, 1):
            prefix = f"[{i:>4}/{len(pending)}]  {date}"

            try:
                items, source = scheduler.fetch(date, client)

                if not items:
                    logger.debug(f"{prefix}  → 无数据（非交易日或当日无强势股）[{source}]")
                    empty_days += 1
                else:
                    records = [
                        r for item in items
                        if (r := parse_record(item, date, source)) is not None
                    ]
                    n = upsert_records(records) if records else 0
                    total_records += n
                    success_days  += 1
                    logger.info(
                        f"{prefix}  → {n:>3} 条  [{source}]  "
                        f"(智兔:{scheduler.zhitu_count} 魔码:{scheduler.moma_count})"
                    )

            except RuntimeError as e:
                # 两个 API 配额均耗尽
                logger.error(f"\n{e}")
                logger.info(f"今日已下载 {success_days} 天，明天继续运行可断点续传。")
                break

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"{prefix}  → HTTP {e.response.status_code}：{e.response.text[:120]}"
                )
                fail_days  += 1
                fail_dates.append(date)

            except httpx.RequestError as e:
                logger.warning(f"{prefix}  → 网络错误：{e}")
                fail_days  += 1
                fail_dates.append(date)

            except Exception as e:
                logger.error(f"{prefix}  → 未知异常：{e}")
                fail_days  += 1
                fail_dates.append(date)

            # 请求间隔
            if i < len(pending):
                time.sleep(sleep_seconds)

    # 汇总
    logger.info(
        f"\n{'='*60}\n"
        f"下载完成：\n"
        f"  成功入库：{success_days} 天，共 {total_records:,} 条记录\n"
        f"  无数据：  {empty_days} 天\n"
        f"  失败：    {fail_days} 天\n"
        f"  {scheduler.summary()}\n"
        + (f"  失败日期：{fail_dates[:10]}{'...' if len(fail_dates)>10 else ''}\n" if fail_dates else "")
        + f"{'='*60}"
    )
    print_stats()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="下载历史强势股池数据到 PostgreSQL（智兔+魔码云服交替）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    today        = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument("--start", default=one_year_ago,
                        help=f"起始日期（默认：1年前 {one_year_ago}）")
    parser.add_argument("--end",   default=today,
                        help=f"结束日期（默认：今天 {today}）")
    parser.add_argument("--date",  default=None,
                        help="只下载指定日期（优先于 --start/--end，用于测试）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载（覆盖已有数据）")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="每次请求后等待秒数（默认 0.3s）")
    parser.add_argument("--stats", action="store_true",
                        help="只显示数据库统计，不下载")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.date:
        args.start = args.date
        args.end   = args.date

    logger.info(
        f"下载配置：{args.start} ~ {args.end}  "
        f"force={args.force}  sleep={args.sleep}s\n"
        f"API：智兔 + 魔码云服 交替（每 {SWITCH_EVERY} 次切换，"
        f"各限 {MAX_PER_API_PER_DAY} 次/天）"
    )

    download_history(
        start=args.start,
        end=args.end,
        force=args.force,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
