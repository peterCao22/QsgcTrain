"""
路线A：LightGBM 横截面排序模型（LambdaRank）

设计要点：
1. 训练目标：按截面日期分组，对每日内所有股票做排序学习（LambdaRank）
2. 标签：rank_label（组内收益分位数，0~1），而非原始收益率
3. Walk-forward 切分：训练集 = 前 N 个截面，测试集 = 下一个截面
4. 特征重要性：训练后记录并输出，辅助后续特征工程决策

主要接口：
    BaselineRankModel.fit(panel)      训练模型
    BaselineRankModel.predict(X)      输出每只股票的排序分数（越高越好）
    BaselineRankModel.save/load       序列化
    BaselineRankModel.feature_importance()  特征重要性
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from config import LABEL_COL, LGB_DEFAULT_PARAMS, MODELS_DIR, TOPK_VALUES
from validation.labels import RANK_LABEL_COL

# 每组（截面日期）的查询规模上限，防止单天股票数过多导致内存问题
MAX_GROUP_SIZE = 5000


class BaselineRankModel:
    """
    LightGBM LambdaRank 横截面排序模型。

    使用示例：
        model = BaselineRankModel()
        model.fit(train_panel)
        scores = model.predict(test_X)
    """

    def __init__(self, params: Optional[Dict] = None, name: str = "baseline_lgb"):
        self.params = params or LGB_DEFAULT_PARAMS.copy()
        self.name = name
        self._booster: Optional[lgb.Booster] = None
        self._feature_cols: Optional[List[str]] = None

    # ── 训练 ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_panel: pd.DataFrame,
        val_panel: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "BaselineRankModel":
        """
        训练 LambdaRank 模型。

        Args:
            train_panel:  面板 DataFrame，含 [date, instrument, feat..., rank_label]
            val_panel:    验证集（可选，用于 early stopping）
            feature_cols: 特征列名列表；None 时自动推断（排除 meta 列和标签列）

        Returns:
            self
        """
        meta = {"date", "instrument", LABEL_COL, RANK_LABEL_COL}
        if feature_cols is None:
            feature_cols = [c for c in train_panel.columns if c not in meta]
        self._feature_cols = feature_cols

        X_train, y_train, groups_train = self._prepare_lgb_data(train_panel, feature_cols)

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            free_raw_data=False,
        )

        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets: List[lgb.Dataset] = [train_dataset]
        valid_names: List[str] = ["train"]

        if val_panel is not None and not val_panel.empty:
            X_val, y_val, groups_val = self._prepare_lgb_data(val_panel, feature_cols)
            val_dataset = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                reference=train_dataset,
                free_raw_data=False,
            )
            valid_sets.append(val_dataset)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.params.get("early_stopping_rounds", 50),
                verbose=False,
            ))

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}

        self._booster = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=self.params.get("n_estimators", 500),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(
            f"[{self.name}] Training done. "
            f"Best iteration: {self._booster.best_iteration}"
        )
        return self

    # ── 预测 ─────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        对特征矩阵打分。

        Args:
            X: DataFrame，columns 须与训练时 feature_cols 匹配。

        Returns:
            ndarray，shape=(len(X),)，排序分数（越高越好）。
        """
        if self._booster is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self._feature_cols is None:
            raise RuntimeError("feature_cols not set.")

        X_aligned = X[self._feature_cols].fillna(0.0)
        return self._booster.predict(X_aligned)

    def predict_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        对面板 DataFrame 打分，返回包含 score 列的 DataFrame。

        Returns:
            panel 中追加 'score' 列
        """
        result = panel.copy()
        result["score"] = self.predict(panel)
        return result

    # ── 评估 ─────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_panel: pd.DataFrame,
        topk_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        在测试面板上评估模型排序能力。

        指标：
            ic_mean      : 各截面 Spearman IC 均值
            ic_std       : IC 标准差
            icir         : IC / IC_std（信息比率）
            topk_win_rate: TopK 胜率（正收益比例）
            topk_mean_ret: TopK 均收益

        Returns:
            指标字典
        """
        if LABEL_COL not in test_panel.columns:
            raise ValueError(f"Column '{LABEL_COL}' missing in test_panel")

        topk_values = topk_values or TOPK_VALUES
        scored = self.predict_panel(test_panel)

        ics = []
        topk_rets: Dict[int, List[float]] = {k: [] for k in topk_values}
        topk_win: Dict[int, List[float]] = {k: [] for k in topk_values}

        for _, gdf in scored.groupby("date"):
            gdf = gdf.dropna(subset=["score", LABEL_COL])
            if len(gdf) < 5:
                continue

            # Spearman IC
            ic, _ = spearmanr(gdf["score"], gdf[LABEL_COL])
            if not np.isnan(ic):
                ics.append(ic)

            # TopK 收益
            gdf_sorted = gdf.nlargest(max(topk_values), "score")
            for k in topk_values:
                top = gdf_sorted.head(k)
                returns = top[LABEL_COL].values
                topk_rets[k].append(returns.mean())
                topk_win[k].append((returns > 0).mean())

        metrics: Dict[str, float] = {}
        if ics:
            metrics["ic_mean"] = float(np.mean(ics))
            metrics["ic_std"] = float(np.std(ics))
            metrics["icir"] = float(np.mean(ics) / (np.std(ics) + 1e-9))
        else:
            metrics["ic_mean"] = 0.0
            metrics["ic_std"] = 0.0
            metrics["icir"] = 0.0

        for k in topk_values:
            if topk_rets[k]:
                metrics[f"top{k}_mean_ret"] = float(np.mean(topk_rets[k]))
                metrics[f"top{k}_win_rate"] = float(np.mean(topk_win[k]))

        return metrics

    # ── 特征重要性 ────────────────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """返回特征重要性 Series（按 importance 降序）。"""
        if self._booster is None:
            raise RuntimeError("Model not trained.")
        imp = self._booster.feature_importance(importance_type=importance_type)
        names = self._booster.feature_name()
        s = pd.Series(imp, index=names, name=importance_type).sort_values(ascending=False)
        return s

    def log_feature_importance(self, top_n: int = 20) -> None:
        """打印 Top N 特征重要性。"""
        imp = self.feature_importance()
        lines = [f"Feature importance (top {top_n}):"]
        for name, val in imp.head(top_n).items():
            lines.append(f"  {name:<42} {val:.1f}")
        logger.info("\n".join(lines))

    # ── 序列化 ────────────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """保存模型（LightGBM 原生格式 + feature_cols JSON）。"""
        if self._booster is None:
            raise RuntimeError("Nothing to save.")
        path = path or MODELS_DIR / f"{self.name}.lgb"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = {
            "name": self.name,
            "feature_cols": self._feature_cols,
            "params": self.params,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info(f"Model saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BaselineRankModel":
        """从文件加载模型。"""
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}

        obj = cls(params=meta.get("params"), name=meta.get("name", path.stem))
        obj._booster = lgb.Booster(model_file=str(path))
        obj._feature_cols = meta.get("feature_cols")
        logger.info(f"Model loaded: {path}")
        return obj

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_lgb_data(
        panel: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        将面板 DataFrame 转换为 (X, y, groups) 格式供 LightGBM Dataset 使用。
        groups 是每个截面日期的样本数列表（LambdaRank 需要）。
        """
        # 按日期排序，保持同一日期的样本连续
        panel = panel.sort_values(["date", "instrument"]).reset_index(drop=True)

        X = panel[feature_cols].fillna(0.0).values.astype(np.float32)

        # rank_label 作为 LambdaRank 的 label（需要整数或浮点分位数排名）
        if RANK_LABEL_COL in panel.columns:
            # 将 0~1 分位数转换为 0~99 的整数（LightGBM LambdaRank 要求非负整数）
            y = (panel[RANK_LABEL_COL].fillna(0.0) * 99).round().astype(int).values
        else:
            y = panel[LABEL_COL].fillna(0.0).rank(pct=True).values
            y = (y * 99).round().astype(int)

        # groups：每个截面的样本数
        groups = panel.groupby("date", sort=False).size().tolist()

        return X, y, groups
