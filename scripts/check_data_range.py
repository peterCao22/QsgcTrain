"""检查各数据表的日期覆盖范围，用于评估可扩展的训练窗口。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.db_loader import read_sql

tables = [
    ("kline_all",   "SELECT MIN(date) mn, MAX(date) mx, COUNT(DISTINCT date) nd, COUNT(*) cnt FROM kline_all"),
    ("chips_all",   "SELECT MIN(date) mn, MAX(date) mx, COUNT(DISTINCT date) nd, COUNT(*) cnt FROM chips_all"),
    ("moneyflow",   "SELECT MIN(date) mn, MAX(date) mx, COUNT(DISTINCT date) nd, COUNT(*) cnt FROM moneyflow"),
    ("strong_pool", "SELECT MIN(date) mn, MAX(date) mx, COUNT(DISTINCT date) nd, COUNT(*) cnt FROM strong_pool"),
    ("index_bar1d", "SELECT MIN(date) mn, MAX(date) mx, COUNT(DISTINCT date) nd, COUNT(*) cnt FROM index_bar1d"),
]

print("=" * 70)
print(f"{'Table':<15} {'Min Date':<12} {'Max Date':<12} {'Dates':>8} {'Rows':>12}")
print("-" * 70)
for name, sql in tables:
    try:
        r = read_sql(sql)
        row = r.iloc[0]
        print(f"{name:<15} {str(row['mn']):<12} {str(row['mx']):<12} {int(row['nd']):>8,} {int(row['cnt']):>12,}")
    except Exception as e:
        print(f"{name:<15} ERROR: {e}")
print("=" * 70)
