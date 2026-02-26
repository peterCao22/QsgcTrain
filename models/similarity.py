"""
路线B：强势原型相似度模型

核心思路：
1. 从强势股池在 t 日的特征向量中，计算"强势原型"
   （可选：均值质心 / 加权质心 / 主成分方向）
2. 对全市场所有股票计算与强势原型的余弦相似度
3. 相似度分数作为第二维度打分，与路线A融合

验证核心假设：
    "与强势原型相似的股票，是否有更高的30日收益？"

主要接口：
    SimilarityScorer.fit_prototype(strong_feat_df)
        从强势股池特征中构建原型
    SimilarityScorer.score(market_feat_df)
        返回全市场每只股票的相似度分数（-1 ~ 1，越高越相似）
    SimilarityScorer.evaluate_hypothesis(market_feat_df, label_df)
        验证"相似度高 → 收益高"假设，返回分组统计
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class SimilarityScorer:
    """
    强势原型相似度打分器。

    使用示例：
        scorer = SimilarityScorer(method='mean')
        scorer.fit_prototype(strong_feat_df)
        sim_scores = scorer.score(market_feat_df)

        # 验证假设
        result = scorer.evaluate_hypothesis(market_feat_df, label_df, date)
    """

    def __init__(
        self,
        method: Literal["mean", "weighted", "pca"] = "mean",
        n_pca_components: int = 3,
    ):
        """
        Args:
            method: 原型构建方式
                'mean'     — 强势股特征均值（质心，最简单）
                'weighted' — 按强势股池中股票近期表现加权（需提供权重）
                'pca'      — 取强势股特征矩阵的第一主成分方向作为原型
            n_pca_components: PCA 时提取的主成分数（仅 method='pca' 有效）
        """
        self.method = method
        self.n_pca_components = n_pca_components

        self._prototype: Optional[np.ndarray] = None   # shape=(n_features,)
        self._feature_cols: Optional[List[str]] = None
        self._pca: Optional[PCA] = None

    # ── 构建原型 ─────────────────────────────────────────────────────────────

    def fit_prototype(
        self,
        strong_feat_df: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> "SimilarityScorer":
        """
        从强势股池的特征矩阵中构建原型向量。

        Args:
            strong_feat_df: 强势股池在某日的特征 DataFrame
                            index=instrument, columns=特征列
                            （须已经过标准化，与全市场特征同分布）
            weights:        强势股的权重向量（仅 method='weighted' 有效），
                            shape=(len(strong_feat_df),)，可用近期收益率作为权重。

        Returns:
            self
        """
        if strong_feat_df.empty:
            logger.warning("Strong pool features empty; prototype not updated")
            return self

        self._feature_cols = list(strong_feat_df.columns)
        X = strong_feat_df.values.astype(np.float32)

        if self.method == "mean":
            proto = X.mean(axis=0)

        elif self.method == "weighted":
            if weights is None:
                logger.warning("method='weighted' but no weights provided; falling back to mean")
                proto = X.mean(axis=0)
            else:
                w = np.array(weights, dtype=np.float32)
                w = np.abs(w)  # 确保非负
                total = w.sum()
                proto = (X * w[:, np.newaxis]).sum(axis=0) / (total + 1e-9)

        elif self.method == "pca":
            n_comp = min(self.n_pca_components, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_comp)
            self._pca.fit(X)
            # 第一主成分方向作为原型
            proto = self._pca.components_[0]

        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        # L2 归一化（余弦相似度要求）
        self._prototype = _l2_normalize(proto)

        logger.info(
            f"Prototype fitted [{self.method}]: "
            f"{len(strong_feat_df)} strong stocks → "
            f"prototype shape={self._prototype.shape}"
        )
        return self

    # ── 打分 ─────────────────────────────────────────────────────────────────

    def score(
        self,
        market_feat_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        计算全市场每只股票与强势原型的余弦相似度。

        Args:
            market_feat_df: 全市场特征 DataFrame，index=instrument, columns=特征列
            feature_cols:   指定使用哪些特征列（须与 fit_prototype 时一致）；
                            None 则使用 _feature_cols。

        Returns:
            pd.Series：index=instrument, values=余弦相似度（-1 ~ 1）
        """
        if self._prototype is None:
            raise RuntimeError("Prototype not built. Call fit_prototype() first.")

        cols = feature_cols or self._feature_cols
        if cols is None:
            raise RuntimeError("feature_cols not set.")

        # 对齐特征列（训练/预测时可能有缺失列）
        missing = [c for c in cols if c not in market_feat_df.columns]
        if missing:
            logger.warning(f"Missing feature cols: {missing}; filling with 0")
            for c in missing:
                market_feat_df = market_feat_df.copy()
                market_feat_df[c] = 0.0

        X = market_feat_df[cols].fillna(0.0).values.astype(np.float32)

        # L2 归一化后做点积 = 余弦相似度
        X_norm = normalize(X, norm="l2")
        sim = X_norm @ self._prototype  # shape=(n_stocks,)

        return pd.Series(sim, index=market_feat_df.index, name="similarity_score")

    # ── 假设验证 ─────────────────────────────────────────────────────────────

    def evaluate_hypothesis(
        self,
        market_feat_df: pd.DataFrame,
        label_series: pd.Series,
        date: Optional[str] = None,
        n_quantiles: int = 5,
    ) -> Dict:
        """
        验证核心假设：相似度高的股票是否有更高的30日收益？

        Args:
            market_feat_df:  全市场特征 DataFrame，index=instrument
            label_series:    30日收益 Series，index=instrument
            date:            截面日期（仅用于日志）
            n_quantiles:     按相似度分成几组

        Returns:
            {
              'spearman_ic':  相似度与收益的 Spearman IC,
              'quantile_ret': 各分位组的平均收益 DataFrame,
              'top_vs_bottom': TopQ 收益 - BottomQ 收益（多空差）,
            }
        """
        sim = self.score(market_feat_df)

        # 对齐
        common = sim.index.intersection(label_series.index)
        if len(common) < 20:
            logger.warning(f"Only {len(common)} common instruments for hypothesis test")
            return {}

        sim_aligned = sim.loc[common]
        ret_aligned = label_series.loc[common]

        ic, pval = spearmanr(sim_aligned, ret_aligned)

        # 分组统计
        df_eval = pd.DataFrame({"sim": sim_aligned, "ret": ret_aligned})
        df_eval["quantile"] = pd.qcut(
            df_eval["sim"], q=n_quantiles, labels=False, duplicates="drop"
        )
        quantile_ret = df_eval.groupby("quantile")["ret"].agg(["mean", "std", "count"])
        quantile_ret.index = [f"Q{i+1}" for i in range(len(quantile_ret))]

        top_q_ret = quantile_ret.iloc[-1]["mean"]
        bot_q_ret = quantile_ret.iloc[0]["mean"]
        long_short_spread = top_q_ret - bot_q_ret

        result = {
            "date": date,
            "spearman_ic": float(ic),
            "pvalue": float(pval),
            "long_short_spread": float(long_short_spread),
            "quantile_ret": quantile_ret,
            "n_stocks": len(common),
        }

        _log_hypothesis_result(result)
        return result

    # ── 获取原型信息 ──────────────────────────────────────────────────────────

    def get_prototype_info(self) -> Dict:
        """返回当前原型的统计信息。"""
        if self._prototype is None:
            return {}
        return {
            "method": self.method,
            "n_features": len(self._prototype),
            "prototype_norm": float(np.linalg.norm(self._prototype)),
            "top_features": self._get_top_features(10),
        }

    def _get_top_features(self, top_n: int) -> List[Tuple[str, float]]:
        """返回在原型中权重最大的 top_n 个特征（绝对值最大）。"""
        if self._prototype is None or self._feature_cols is None:
            return []
        idx = np.argsort(np.abs(self._prototype))[::-1][:top_n]
        return [
            (self._feature_cols[i], float(self._prototype[i]))
            for i in idx
        ]


# ─── 多日期批量假设验证 ───────────────────────────────────────────────────────

def batch_evaluate_hypothesis(
    feature_builder,            # FeatureBuilder 实例
    strong_pool_loader,         # StrongPoolLoader 实例
    label_df: pd.DataFrame,
    dates: List[str],
    method: str = "mean",
) -> pd.DataFrame:
    """
    在多个截面日期批量验证"强势相似度 → 30日收益"假设。

    Args:
        feature_builder:    FeatureBuilder（已 preload）
        strong_pool_loader: StrongPoolLoader
        label_df:           [date, instrument, fwd_return_30d] 标签 DataFrame
        dates:              截面日期列表
        method:             原型构建方式（'mean' / 'pca'）

    Returns:
        汇总 DataFrame：[date, spearman_ic, pvalue, long_short_spread, n_stocks]
    """
    from config import LABEL_COL

    results = []
    scorer = SimilarityScorer(method=method)
    label_df = label_df.copy()
    label_df["date"] = pd.to_datetime(label_df["date"])

    for date in dates:
        ts = pd.Timestamp(date)

        # 强势股特征
        strong_insts = strong_pool_loader.get(date)
        if not strong_insts:
            logger.warning(f"Empty strong pool @ {date}")
            continue

        # 全市场特征
        market_feat = feature_builder.build(date)
        if market_feat.empty:
            continue

        # 强势股池在全市场特征中的子集
        strong_feat = market_feat.loc[
            market_feat.index.isin(strong_insts)
        ]
        if strong_feat.empty:
            logger.warning(f"No strong pool stocks in market features @ {date}")
            continue

        scorer.fit_prototype(strong_feat)

        # 当日标签
        day_labels = label_df[label_df["date"] == ts].set_index("instrument")[LABEL_COL]

        hyp = scorer.evaluate_hypothesis(market_feat, day_labels, date=date)
        if hyp:
            results.append({
                "date": date,
                "spearman_ic": hyp["spearman_ic"],
                "pvalue": hyp["pvalue"],
                "long_short_spread": hyp["long_short_spread"],
                "n_stocks": hyp["n_stocks"],
            })

    if not results:
        return pd.DataFrame()

    summary = pd.DataFrame(results)
    mean_ic = summary["spearman_ic"].mean()
    mean_spread = summary["long_short_spread"].mean()
    logger.info(
        f"\n{'='*60}\n"
        f"Hypothesis Validation Summary ({len(results)} dates)\n"
        f"  Mean Spearman IC:     {mean_ic:+.4f}\n"
        f"  Mean L/S Spread:      {mean_spread:+.4f}\n"
        f"  IC > 0 ratio:         {(summary['spearman_ic'] > 0).mean():.2%}\n"
        f"{'='*60}"
    )
    return summary


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _log_hypothesis_result(result: Dict) -> None:
    date_str = result.get("date", "")
    ic = result.get("spearman_ic", 0.0)
    spread = result.get("long_short_spread", 0.0)
    n = result.get("n_stocks", 0)
    pval = result.get("pvalue", 1.0)

    if "quantile_ret" in result and result["quantile_ret"] is not None:
        qdf = result["quantile_ret"]
        q_lines = []
        for idx, row in qdf.iterrows():
            q_lines.append(f"  {idx}: mean={row['mean']:+.4f}  n={int(row['count'])}")
        q_str = "\n".join(q_lines)
    else:
        q_str = ""

    logger.info(
        f"Hypothesis [{date_str}] IC={ic:+.4f} (p={pval:.3f})  "
        f"L/S Spread={spread:+.4f}  n={n}\n"
        + (q_str if q_str else "")
    )
