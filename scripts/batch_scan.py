"""
批量历史回测扫描脚本

对多个历史日期依次运行 daily_scan，生成 results/scan_YYYYMMDD.csv。
跳过已存在的结果文件（断点续跑）。

运行方式：
    conda activate rqsdk
    python -X utf8 scripts/batch_scan.py
    python -X utf8 scripts/batch_scan.py --start 2024-01-01 --end 2025-12-31 --step 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from loguru import logger

from config import RESULTS_DIR
from data.db_loader import get_trading_days
from scripts.daily_scan import run_scan


def main():
    parser = argparse.ArgumentParser(description="批量历史扫描（每隔 step 个交易日扫描一次）")
    parser.add_argument("--start",  default="2023-01-01",  help="起始日期")
    parser.add_argument("--end",    default="2025-12-20",  help="结束日期（需留 22 个交易日做验证）")
    parser.add_argument("--step",   type=int, default=40,  help="每隔多少交易日扫描一次（默认40≈2个月）")
    parser.add_argument("--topk",   type=int, default=50,  help="每次输出 Top-K（默认50）")
    parser.add_argument("--force",  action="store_true",   help="强制重新扫描（覆盖已有结果）")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trading_days = list(get_trading_days(args.start, args.end))
    scan_dates   = [trading_days[i] for i in range(0, len(trading_days), args.step)]

    logger.info(f"Batch scan: {len(scan_dates)} dates  step={args.step}  topk={args.topk}")

    success = 0
    skipped = 0
    failed  = 0

    for dt in scan_dates:
        date_str  = dt.strftime("%Y-%m-%d")
        out_file  = RESULTS_DIR / f"scan_{dt.strftime('%Y%m%d')}.csv"

        if out_file.exists() and not args.force:
            logger.info(f"Skip (already exists): {out_file.name}")
            skipped += 1
            continue

        logger.info(f"Scanning {date_str} ...")
        try:
            run_scan(scan_date=date_str, topk=args.topk)   # returns (top_df, global_stats)
            success += 1
        except Exception as e:
            logger.warning(f"Failed on {date_str}: {e}")
            failed += 1

    logger.info(
        f"\nBatch scan complete: "
        f"success={success}  skipped={skipped}  failed={failed}"
    )


if __name__ == "__main__":
    main()
