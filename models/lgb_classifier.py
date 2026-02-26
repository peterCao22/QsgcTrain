"""
Phase 1A：LightGBM 二分类模型

正样本：进入强势股池的股票在 T-N 时的特征
负样本：同期全市场其他股票

两种预测用途：
  1. 训练模式  fit(X_train, y_train)
  2. 预测模式  predict_proba(X) → "成为强势股"的概率分
  3. 评估模式  evaluate(X_test, y_test) → AUC, precision@K, recall@K
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("lightgbm not installed. Run: pip install lightgbm")

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import RobustScaler
except ImportError:
    raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

from features.precursor import ALL_FEATURE_COLS

# 项目级常量（与 config.py 保持一致）
LABEL_COL      = "fwd_return_20d"    # 连续标签（T_feat->T_entry 的真实涨幅，用于 IC 分析）
TARGET_COL     = "is_strong_pos"     # 主训练目标：是否首次入强势池（画像匹配，AUC 0.65）
ALT_TARGET_COL = "is_top20pct"       # 辅助目标：截面前 20% 涨幅（IC/排名验证用）
MODELS_DIR     = Path(__file__).parent.parent / "models" / "saved"
FEATURE_LIB_PATH = Path(__file__).parent.parent / "data" / "feature_library.json"
TOPK_VALUES    = [10, 20, 50]


def load_feature_library(path: Optional[Path] = None) -> Optional[List[str]]:
    """
    从 feature_library.json 读取已筛选的入库特征列表。

    Returns:
        入库特征名列表；若文件不存在则返回 None（调用方应退回使用全部特征）。
    """
    lib_path = path or FEATURE_LIB_PATH
    if not lib_path.exists():
        logger.warning(f"Feature library not found: {lib_path}  (will use ALL_FEATURE_COLS)")
        return None
    with open(lib_path, "r", encoding="utf-8") as f:
        lib = json.load(f)
    selected = lib.get("selected_features", [])
    if not selected:
        logger.warning("feature_library.json has empty selected_features, using ALL_FEATURE_COLS")
        return None
    logger.info(f"Feature library loaded: {len(selected)} selected features  (version={lib.get('version')})")
    return selected


# ─── 默认超参 ─────────────────────────────────────────────────────────────────

DEFAULT_PARAMS: Dict = {
    "objective":             "binary",
    "metric":                "auc",
    "learning_rate":         0.02,     # 小学习率，让模型多学几轮
    "num_leaves":            63,       # 适当增大，允许更复杂的分裂
    "max_depth":             6,
    "min_child_samples":     15,       # 从30降到15，允许更细的分裂
    "feature_fraction":      0.8,
    "bagging_fraction":      0.8,
    "bagging_freq":          3,
    "lambda_l1":             0.05,
    "lambda_l2":             0.1,
    "n_estimators":          1000,
    "early_stopping_rounds": 80,       # 增大早停窗口，给模型更多机会
    "verbose":               -1,
    "n_jobs":                -1,
    "class_weight":          "balanced",
}


# ─── 主类 ────────────────────────────────────────────────────────────────────

class LGBClassifier:
    """
    LightGBM 二分类强势股预测模型。

    训练逻辑：
      X = 蓄力期特征矩阵（来自 precursor.py）
      y = is_strong_pos（1=后来进入强势池，0=否）
      输出 = predict_proba → 每只股票"成为强势股"的概率
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params  = {**DEFAULT_PARAMS, **(params or {})}
        self.model_: Optional[LGBMClassifier] = None
        self.feature_cols_: Optional[List[str]] = None
        self.threshold_: float = 0.5   # 二分类阈值（可通过 optimize_threshold 调整）
        self._train_meta: Dict = {}

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "LGBClassifier":
        """
        训练模型。

        Args:
            X:            特征矩阵（行=样本，列=特征）
            y:            标签（0/1）
            X_val:        验证集特征（用于 early stopping）
            y_val:        验证集标签
            feature_cols: 指定使用哪些特征列；None = 使用 ALL_FEATURE_COLS
        """
        self.feature_cols_ = feature_cols or [c for c in ALL_FEATURE_COLS if c in X.columns]
        X_feat = X[self.feature_cols_].copy()

        logger.info(
            f"Training LGBClassifier: {len(X_feat):,} samples  "
            f"pos={y.mean():.2%}  features={len(self.feature_cols_)}"
        )

        # 中位数填充缺失值
        self._fill_medians = X_feat.median()
        X_feat = X_feat.fillna(self._fill_medians)

        fit_kwargs: Dict = {}
        if X_val is not None and y_val is not None:
            X_val_feat = X_val[self.feature_cols_].fillna(self._fill_medians)
            fit_kwargs["eval_set"] = [(X_val_feat, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.get("early_stopping_rounds", 40)),
                lgb.log_evaluation(period=100),
            ]

        early = self.params.pop("early_stopping_rounds", 40)
        self.model_ = LGBMClassifier(**self.params)
        self.params["early_stopping_rounds"] = early  # 恢复

        self.model_.fit(X_feat, y, **fit_kwargs)

        self._train_meta = {
            "n_train":       len(X_feat),
            "pos_rate":      float(y.mean()),
            "n_features":    len(self.feature_cols_),
            "best_iteration": getattr(self.model_, "best_iteration_", None),
        }
        logger.info(f"Trained. best_iteration={self._train_meta['best_iteration']}")
        return self

    # ── 预测 ──────────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回每个样本"成为强势股"的概率（0~1）。"""
        self._assert_fitted()
        X_feat = X[self.feature_cols_].fillna(self._fill_medians)
        return self.model_.predict_proba(X_feat)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回二分类预测结果（0/1）。"""
        proba = self.predict_proba(X)
        return (proba >= self.threshold_).astype(int)

    # ── 评估 ──────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        topk_values: Optional[List[int]] = None,
    ) -> Dict:
        """
        输出完整评估指标。

        Returns dict with:
          auc:             ROC-AUC
          ap:              Average Precision（PR-AUC）
          precision_at_k:  Top-K 精确率（K=20,50）
          recall_at_k:     Top-K 召回率
        """
        self._assert_fitted()
        proba = self.predict_proba(X)
        y_arr = np.array(y)
        topk  = topk_values or TOPK_VALUES

        metrics: Dict = {}
        metrics["auc"] = float(roc_auc_score(y_arr, proba))
        metrics["ap"]  = float(average_precision_score(y_arr, proba))

        # Top-K 指标
        sorted_idx = np.argsort(proba)[::-1]
        for k in topk:
            top_k_idx  = sorted_idx[:k]
            precision  = float(y_arr[top_k_idx].mean())
            n_pos_total = y_arr.sum()
            recall     = float(y_arr[top_k_idx].sum() / n_pos_total) if n_pos_total > 0 else 0.0
            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"]    = recall

        logger.info(
            f"Evaluate: AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
            + "  ".join(f"P@{k}={metrics[f'precision@{k}']:.3f}" for k in topk)
        )
        return metrics

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        strategy: str = "f1",
    ) -> float:
        """
        在验证集上选择最佳分类阈值。

        strategy: 'f1' | 'precision' | 'recall'
        """
        proba = self.predict_proba(X_val)
        y_arr = np.array(y_val)

        best_score = -1.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.02):
            pred = (proba >= thresh).astype(int)
            tp = ((pred == 1) & (y_arr == 1)).sum()
            fp = ((pred == 1) & (y_arr == 0)).sum()
            fn = ((pred == 0) & (y_arr == 1)).sum()
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            if strategy == "f1":
                score = 2 * p * r / (p + r + 1e-10)
            elif strategy == "precision":
                score = p
            else:
                score = r
            if score > best_score:
                best_score  = score
                best_thresh = thresh

        self.threshold_ = float(best_thresh)
        logger.info(f"Threshold optimized ({strategy}): {self.threshold_:.2f}  score={best_score:.4f}")
        return self.threshold_

    # ── 特征重要性 ────────────────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """返回按重要性排序的特征列表。"""
        self._assert_fitted()
        imp = pd.Series(
            self.model_.feature_importances_,
            index=self.feature_cols_,
            name=importance_type,
        ).sort_values(ascending=False)
        return imp

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """保存模型到 pkl 文件。"""
        self._assert_fitted()
        save_path = path or (MODELS_DIR / "lgb_classifier.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved: {save_path}")

        # 同时保存特征重要性
        imp_path = save_path.with_suffix(".importance.csv")
        self.feature_importance().to_csv(imp_path)
        logger.info(f"Feature importance: {imp_path}")

        return save_path

    @classmethod
    def load(cls, path: Path) -> "LGBClassifier":
        """从文件加载模型。

        使用自定义 Unpickler 兼容以下两种保存方式：
        - python models/lgb_classifier.py  → 类名为 __main__.LGBClassifier
        - python -m models.lgb_classifier  → 类名为 models.lgb_classifier.LGBClassifier
        """
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "LGBClassifier":
                    return LGBClassifier
                return super().find_class(module, name)

        with open(path, "rb") as f:
            obj = _CompatUnpickler(f).load()
        logger.info(f"Model loaded: {path}")
        return obj

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _assert_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")


# ─── 快速训练入口 ─────────────────────────────────────────────────────────────

def train_from_dataset(
    dataset_path: Optional[Path] = None,
    val_ratio: float = 0.2,
    feature_cols: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    target_col: str = TARGET_COL,
    use_feature_library: bool = True,
) -> Tuple[LGBClassifier, Dict]:
    """
    从训练数据集文件直接训练并评估模型。

    Args:
        dataset_path:         parquet 路径（默认 data/training_dataset.parquet）
        val_ratio:            最后 val_ratio 的日期作为验证集（按时间排序）
        feature_cols:         特征列（None 时：优先读特征库，再退回全部特征）
        params:               LightGBM 超参覆盖
        target_col:           训练目标列（默认 is_top20pct；可切换为 is_strong_pos）
        use_feature_library:  True = 优先读 feature_library.json 的入库特征列表

    Returns:
        (model, metrics_dict)
    """
    path = dataset_path or (Path(__file__).parent.parent / "data" / "training_dataset.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if target_col not in df.columns:
        available = [c for c in [TARGET_COL, ALT_TARGET_COL] if c in df.columns]
        fallback = available[0] if available else None
        if fallback is None:
            raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
        logger.warning(f"Target '{target_col}' not found, falling back to '{fallback}'")
        target_col = fallback

    # 时间序列切分（按日期，不做随机打乱）
    dates = df["date"].unique()
    cutoff = dates[int(len(dates) * (1 - val_ratio))]
    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]

    pos_rate_train = train[target_col].mean()
    pos_rate_val   = val[target_col].mean()
    logger.info(
        f"Target: {target_col}\n"
        f"Train: {len(train):,} rows ({train['date'].min().date()} ~ {train['date'].max().date()})  "
        f"pos={pos_rate_train:.2%}\n"
        f"Val:   {len(val):,} rows ({val['date'].min().date()} ~ {val['date'].max().date()})  "
        f"pos={pos_rate_val:.2%}"
    )

    # 特征列选择优先级：
    #   1. 显式传入 feature_cols（最高优先级）
    #   2. use_feature_library=True 时从 feature_library.json 读取入库特征
    #   3. 退回使用全部 ALL_FEATURE_COLS
    if feature_cols is not None:
        feat_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Using explicitly provided feature_cols: {len(feat_cols)} features")
    elif use_feature_library:
        lib_cols = load_feature_library()
        if lib_cols:
            feat_cols = [c for c in lib_cols if c in df.columns]
            logger.info(f"Using feature library: {len(feat_cols)} features  "
                        f"(missing from dataset: {set(lib_cols) - set(df.columns)})")
        else:
            feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
            logger.info(f"Feature library unavailable, using all {len(feat_cols)} features")
    else:
        feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        logger.info(f"use_feature_library=False, using all {len(feat_cols)} features")

    X_train = train[feat_cols]
    y_train = train[target_col]
    X_val   = val[feat_cols]
    y_val   = val[target_col]

    model = LGBClassifier(params=params)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_cols=feat_cols)

    metrics = model.evaluate(X_val, y_val)

    # 打印特征重要性 Top10
    imp = model.feature_importance()
    logger.info(f"\nTop 10 特征重要性：\n{imp.head(10).to_string()}")

    # 保存模型
    model.save()

    return model, metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练 LightGBM 强势股分类模型")
    parser.add_argument("--dataset",   default=None,        help="训练数据集路径")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（时间切分）")
    parser.add_argument("--target",    default=TARGET_COL,
                        help=f"训练目标列（默认 {TARGET_COL}；可用 {ALT_TARGET_COL}）")
    parser.add_argument("--no-feature-library", action="store_true", default=True,
                        help="不使用特征库，改用全部特征训练（新架构默认：全量特征输入，自动发现重要性）")
    parser.add_argument("--use-feature-library", action="store_true", default=False,
                        help="使用已有 feature_library.json（仅在库已生成且验证通过后使用）")
    args = parser.parse_args()

    use_lib = args.use_feature_library and not args.no_feature_library
    model, metrics = train_from_dataset(
        dataset_path        = Path(args.dataset) if args.dataset else None,
        val_ratio           = args.val_ratio,
        target_col          = args.target,
        use_feature_library = use_lib,
    )
    print(f"\n训练完成，验证集指标（target={args.target}）：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
