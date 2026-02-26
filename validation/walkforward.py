"""
Walk-forward 验证框架

验证协议：
- 每周选一个截面日期（如周五收盘后）
- 用前 N 个截面训练模型，对下一个截面的全市场股票打分
- 选出 TopK，计算 30日后的收盘收益
- 关键指标：均值收益 / 胜率 / 相对基准超额 / 最大回撤

两种运行模式：
    Mode 1 — baseline_only：仅运行路线A（LGB排序模型）
    Mode 2 — full：运行路线A + 路线B（相似度分数融合）

主要接口：
    WalkForwardEvaluator.run(panel, label_df, strong_pool_loader) → 结果 DataFrame
    WalkForwardEvaluator.plot_report(result_df)                  → 可视化
    WalkForwardEvaluator.print_summary(result_df)                → 文字摘要
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config import (
    LABEL_COL,
    RESULTS_DIR,
    TOPK_VALUES,
    WALKFORWARD_STEP_DAYS,
    WALKFORWARD_TRAIN_DAYS,
)
from models.baseline import BaselineRankModel
from validation.labels import RANK_LABEL_COL


class WalkForwardEvaluator:
    """
    Walk-forward 验证器。

    使用示例（典型流程）：
        evaluator = WalkForwardEvaluator(topk_values=[20, 50])
        result = evaluator.run(panel, label_df)
        evaluator.print_summary(result)
    """

    def __init__(
        self,
        topk_values: Optional[List[int]] = None,
        train_window: int = WALKFORWARD_TRAIN_DAYS,
        step: int = WALKFORWARD_STEP_DAYS,
        similarity_weight: float = 0.3,   # 路线B权重（0 = 仅路线A）
    ):
        """
        Args:
            topk_values:      评估的 K 值列表，如 [20, 50]
            train_window:     训练窗口大小（截面数量）
            step:             每次向前滚动的截面数（1 = 每截面都测试）
            similarity_weight: 相似度分数的融合权重（0~1）
        """
        self.topk_values = topk_values or TOPK_VALUES
        self.train_window = train_window
        self.step = step
        self.similarity_weight = similarity_weight

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def run(
        self,
        panel: pd.DataFrame,
        label_df: pd.DataFrame,
        strong_pool_loader=None,
        feature_builder=None,
        save_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        执行完整 Walk-forward 验证。

        Args:
            panel:               多截面面板特征 DataFrame
                                 [date, instrument, feat..., rank_label, fwd_return_30d]
            label_df:            标签 DataFrame [date, instrument, fwd_return_30d]
            strong_pool_loader:  StrongPoolLoader（如需融合相似度分数）
            feature_builder:     FeatureBuilder（如需实时构建特征用于相似度打分）
            save_dir:            结果保存目录（None 则不保存）

        Returns:
            结果 DataFrame，每行为一个测试截面：
            [date, topK_mean_ret, topK_win_rate, topK_excess_ret,
             topK_max_dd, ic, n_train, n_test, topK_instruments]
        """
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        label_df = label_df.copy()
        label_df["date"] = pd.to_datetime(label_df["date"])

        # 取所有截面日期（排序）
        all_dates = sorted(panel["date"].dt.strftime("%Y-%m-%d").unique())
        if len(all_dates) < self.train_window + 1:
            raise ValueError(
                f"Not enough dates for walk-forward: "
                f"{len(all_dates)} < train_window({self.train_window}) + 1"
            )

        # 市场基准：等权全市场每日30日收益（用于超额计算）
        benchmark = self._compute_benchmark(label_df)

        records = []
        test_start_idx = self.train_window
        test_indices = range(test_start_idx, len(all_dates), self.step)

        logger.info(
            f"Walk-forward: {len(test_indices)} test dates  "
            f"train_window={self.train_window}  step={self.step}"
        )

        for test_idx in tqdm(test_indices, desc="Walk-forward"):
            train_dates = all_dates[max(0, test_idx - self.train_window): test_idx]
            test_date = all_dates[test_idx]

            # 切分训练集 / 测试集
            train_panel = panel[panel["date"].isin(
                [pd.Timestamp(d) for d in train_dates]
            )]
            test_panel = panel[panel["date"] == pd.Timestamp(test_date)]

            if train_panel.empty or test_panel.empty:
                continue

            # ── 训练路线A模型 ─────────────────────────────────────────────
            meta_cols = {"date", "instrument", LABEL_COL, RANK_LABEL_COL}
            feat_cols = [c for c in train_panel.columns if c not in meta_cols]

            model = BaselineRankModel(name=f"wf_{test_date}")
            try:
                model.fit(train_panel, feature_cols=feat_cols)
            except Exception as exc:
                logger.warning(f"Model fit failed @ {test_date}: {exc}")
                continue

            # ── 预测 ─────────────────────────────────────────────────────
            test_X = test_panel[feat_cols].fillna(0.0)
            scores_a = model.predict(test_X)  # ndarray

            # ── 融合路线B相似度分数（可选）────────────────────────────────
            if self.similarity_weight > 0 and strong_pool_loader is not None and feature_builder is not None:
                scores_b = self._get_similarity_scores(
                    test_date, strong_pool_loader, feature_builder, test_panel
                )
                if scores_b is not None:
                    # 标准化到同一量纲后融合
                    a_norm = _rank_scale(scores_a)
                    b_norm = _rank_scale(scores_b)
                    final_scores = (
                        (1 - self.similarity_weight) * a_norm
                        + self.similarity_weight * b_norm
                    )
                else:
                    final_scores = scores_a
            else:
                final_scores = scores_a

            # ── 选出 TopK ────────────────────────────────────────────────
            test_panel_scored = test_panel.copy()
            test_panel_scored["score"] = final_scores

            # 获取测试日期的真实标签
            test_labels = label_df[
                label_df["date"] == pd.Timestamp(test_date)
            ][["instrument", LABEL_COL]]

            if test_labels.empty:
                continue

            scored_with_label = test_panel_scored.merge(
                test_labels, on="instrument", how="inner", suffixes=("_feat", "")
            )
            if LABEL_COL + "_feat" in scored_with_label.columns:
                scored_with_label = scored_with_label.drop(columns=[LABEL_COL + "_feat"])

            if scored_with_label.empty:
                continue

            # 计算各 K 值的指标
            record: Dict = {"date": test_date, "n_train": len(train_panel)}

            # 全市场基准收益
            bench_ret = benchmark.get(test_date, 0.0)

            for k in self.topk_values:
                top = scored_with_label.nlargest(k, "score")
                rets = top[LABEL_COL].dropna().values

                if len(rets) == 0:
                    continue

                mean_ret = float(np.mean(rets))
                win_rate = float((rets > 0).mean())
                excess_ret = mean_ret - bench_ret
                max_dd = float(_max_drawdown_from_returns(rets))

                record[f"top{k}_mean_ret"] = mean_ret
                record[f"top{k}_win_rate"] = win_rate
                record[f"top{k}_excess_ret"] = excess_ret
                record[f"top{k}_max_dd"] = max_dd
                record[f"top{k}_instruments"] = ",".join(top["instrument"].tolist())

            # Spearman IC
            valid = scored_with_label.dropna(subset=["score", LABEL_COL])
            if len(valid) >= 10:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid["score"], valid[LABEL_COL])
                record["ic"] = float(ic) if not np.isnan(ic) else 0.0
            else:
                record["ic"] = 0.0

            record["bench_ret"] = bench_ret
            records.append(record)

        result = pd.DataFrame(records)
        if result.empty:
            logger.warning("Walk-forward produced no results")
            return result

        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)

        # 保存结果
        if save_dir is None:
            save_dir = RESULTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        result_path = save_dir / f"walkforward_{ts_str}.csv"
        result.to_csv(result_path, index=False)
        logger.info(f"Walk-forward results saved: {result_path}")

        self.print_summary(result)
        return result

    # ── 报告 ─────────────────────────────────────────────────────────────────

    def print_summary(self, result: pd.DataFrame) -> None:
        """打印验证结果摘要。"""
        if result.empty:
            logger.info("No results to summarize.")
            return

        lines = [
            "\n" + "=" * 70,
            "Walk-forward Validation Summary",
            f"  Period:  {result['date'].min().date()} ~ {result['date'].max().date()}",
            f"  N Dates: {len(result)}",
            "",
        ]

        for k in self.topk_values:
            col_ret = f"top{k}_mean_ret"
            col_win = f"top{k}_win_rate"
            col_exc = f"top{k}_excess_ret"
            col_dd  = f"top{k}_max_dd"

            if col_ret not in result.columns:
                continue

            mean_ret = result[col_ret].mean()
            win_rate = result[col_win].mean()
            excess   = result[col_exc].mean() if col_exc in result.columns else 0.0
            pos_ratio = (result[col_ret] > 0).mean()

            lines += [
                f"  Top{k:>3}:",
                f"    Mean 30d Return:   {mean_ret:+.2%}",
                f"    Win Rate:          {win_rate:.2%}",
                f"    Excess vs Bench:   {excess:+.2%}",
                f"    Positive Periods:  {pos_ratio:.2%}",
            ]
            if col_dd in result.columns:
                lines.append(f"    Avg Max Drawdown:  {result[col_dd].mean():.2%}")
            lines.append("")

        if "ic" in result.columns:
            ic_mean = result["ic"].mean()
            ic_std  = result["ic"].std()
            icir    = ic_mean / (ic_std + 1e-9)
            lines += [
                f"  IC Mean: {ic_mean:+.4f}",
                f"  IC Std:  {ic_std:.4f}",
                f"  ICIR:    {icir:+.4f}",
            ]

        lines.append("=" * 70)
        logger.info("\n".join(lines))

    def plot_report(self, result: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """生成可视化报告（需要 matplotlib）。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return

        n_topk = len(self.topk_values)
        fig, axes = plt.subplots(n_topk + 1, 2, figsize=(14, 5 * (n_topk + 1)))
        if n_topk == 0:
            return

        result = result.sort_values("date")

        for i, k in enumerate(self.topk_values):
            ax_ret = axes[i][0]
            ax_win = axes[i][1]
            col_ret = f"top{k}_mean_ret"
            col_win = f"top{k}_win_rate"
            col_exc = f"top{k}_excess_ret"

            if col_ret in result.columns:
                # 累计收益曲线
                cum_ret = (1 + result[col_ret]).cumprod() - 1
                ax_ret.plot(result["date"], cum_ret * 100, label=f"Top{k}")
                if "bench_ret" in result.columns:
                    cum_bench = (1 + result["bench_ret"]).cumprod() - 1
                    ax_ret.plot(
                        result["date"], cum_bench * 100,
                        linestyle="--", label="Benchmark", alpha=0.7
                    )
                ax_ret.set_title(f"Top{k} Cumulative Return (%)")
                ax_ret.legend()
                ax_ret.grid(True, alpha=0.3)

            if col_win in result.columns:
                ax_win.bar(
                    result["date"],
                    result[col_win] * 100,
                    width=20,
                    alpha=0.7,
                    label=f"Top{k} Win Rate",
                )
                ax_win.axhline(50, color="red", linestyle="--", alpha=0.5)
                ax_win.set_title(f"Top{k} Win Rate (%)")
                ax_win.grid(True, alpha=0.3)

        # IC 图
        ax_ic = axes[-1][0]
        if "ic" in result.columns:
            ax_ic.bar(result["date"], result["ic"], width=20, alpha=0.7)
            ax_ic.axhline(0, color="black", linewidth=0.8)
            ax_ic.set_title("Spearman IC")
            ax_ic.grid(True, alpha=0.3)

        axes[-1][1].axis("off")

        plt.suptitle("Walk-forward Validation Report", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path is None:
            save_path = RESULTS_DIR / "walkforward_report.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Report plot saved: {save_path}")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_benchmark(label_df: pd.DataFrame) -> Dict[str, float]:
        """计算每个截面日期的等权全市场平均收益（用于超额计算）。"""
        label_df = label_df.copy()
        label_df["date"] = pd.to_datetime(label_df["date"])
        bench = (
            label_df.groupby("date")[LABEL_COL]
            .mean()
            .reset_index()
        )
        return dict(zip(
            bench["date"].dt.strftime("%Y-%m-%d"),
            bench[LABEL_COL]
        ))

    def _get_similarity_scores(
        self,
        date: str,
        strong_pool_loader,
        feature_builder,
        test_panel: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """获取路线B相似度分数（返回与 test_panel 对齐的 ndarray）。"""
        from models.similarity import SimilarityScorer

        try:
            strong_insts = strong_pool_loader.get(date)
            if not strong_insts:
                return None

            market_feat = feature_builder.build(date)
            if market_feat.empty:
                return None

            strong_feat = market_feat.loc[market_feat.index.isin(strong_insts)]
            if strong_feat.empty:
                return None

            scorer = SimilarityScorer(method="mean")
            scorer.fit_prototype(strong_feat)

            # 按 test_panel 的 instrument 顺序对齐
            test_insts = test_panel["instrument"].tolist()
            feat_aligned = market_feat.loc[market_feat.index.isin(test_insts)]
            feat_aligned = feat_aligned.reindex(test_insts).fillna(0.0)

            sim = scorer.score(feat_aligned)
            return sim.values

        except Exception as exc:
            logger.warning(f"Similarity score failed @ {date}: {exc}")
            return None


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _rank_scale(arr: np.ndarray) -> np.ndarray:
    """将数组转换为 [0, 1] 排名分位数。"""
    n = len(arr)
    if n == 0:
        return arr
    ranks = arr.argsort().argsort().astype(float)
    return ranks / (n - 1 + 1e-9)


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    """
    从一批截面收益率（非时序，各股票独立）计算"平均最大回撤"近似值。
    这里用各股票中最大亏损（负收益最大值绝对值）作为代理指标。
    """
    neg_rets = returns[returns < 0]
    if len(neg_rets) == 0:
        return 0.0
    return float(np.abs(neg_rets).max())
