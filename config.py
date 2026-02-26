"""
全局配置模块

所有数值常量、路径、数据库连接参数均在此集中管理。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ─── 路径 ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

STRONG_POOL_DIR = DATA_DIR / "strong_pool_snapshots"

for _d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR, STRONG_POOL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── 环境变量 ───────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env", override=True)

# ─── 数据库 ─────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "192.168.21.39")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "stocks_data")

DB_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ─── 强势股池 API ────────────────────────────────────────────────────────────
STRONG_POOL_SOURCE = os.getenv("STRONG_POOL_SOURCE", "local")  # zhitu | moma | local
ZHITU_API_URL = os.getenv("ZHITU_API_URL", "")
ZHITU_API_KEY = os.getenv("ZHITU_API_KEY", "")
MOMA_API_URL = os.getenv("MOMA_API_URL", "")
MOMA_API_KEY = os.getenv("MOMA_API_KEY", "")

# ─── 标的域过滤 ──────────────────────────────────────────────────────────────
# 排除科创（688xxx）、北交所（8xxxxx / 4xxxxx）、ST/退市
EXCLUDE_PREFIXES: List[str] = ["688", "8", "4"]
MIN_LISTING_DAYS: int = 60          # 上市不足N日的新股剔除
MIN_PRICE: float = 1.0              # 剔除仙股（收盘价 < 1元）
MAX_PRICE: float = 3000.0           # 剔除极端高价股（如贵州茅台不影响，可按需调整）

# ─── 特征工程 ────────────────────────────────────────────────────────────────
# 价量动量窗口（交易日）
MOMENTUM_WINDOWS: List[int] = [5, 10, 20, 60]

# 特征有效性筛查阈值：单特征 Spearman < 此值则标记为低信噪比
FEATURE_MIN_SPEARMAN: float = 0.02

# ─── 标签 ───────────────────────────────────────────────────────────────────
LABEL_FORWARD_DAYS: int = 30        # 未来30个交易日收盘收益
LABEL_COL: str = "fwd_return_30d"

# ─── 模型 ───────────────────────────────────────────────────────────────────
TOPK_VALUES: List[int] = [20, 50]   # 评估 Top20 / Top50

# LightGBM 排序模型默认超参
LGB_DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [20, 50],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

# ─── Walk-forward 验证 ───────────────────────────────────────────────────────
# 训练窗口：用过去N个交易日构建训练集
WALKFORWARD_TRAIN_DAYS: int = 504   # ≈2年
# 每次向前滚动的步长（交易日）
WALKFORWARD_STEP_DAYS: int = 20     # 约1个月

# ─── 日志 ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
