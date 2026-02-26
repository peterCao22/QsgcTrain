# QsgcTrain 操作使用手册

**版本**：v1.2  
**更新日期**：2026-02-26  
**虚拟环境**：`conda activate rqsdk`（每次打开终端都需先执行）

---

## 概览：闭环流程

```
① 数据更新          ② 训练数据构建        ③ 模型重训练
 强势池 / K线         正负样本 + 37特征       LightGBM 分类
      │                    │                     │
      ▼                    ▼                     ▼
④ 特征库更新  ──→  ⑤ 全市场扫描  ──→  ⑥ 批量历史回测  ──→  ⑦ 持续验证
  feature_library.json    Top-K 候选              评估历史准确率         验证最新预测
```

每月执行一次完整循环（步骤 ① → ⑥），日常只需执行步骤 ⑤⑦。

---

## 第一步：更新强势池数据

从智兔 + 魔码云服 API 下载最新强势股池数据，断点续传（跳过已有日期）。

```bash
conda activate rqsdk
cd D:\myCursor\QsgcTrain

# 下载最新数据（自动从数据库最新日期续传到今天）
python scripts/download_strong_pool_history.py

# 指定日期范围
python scripts/download_strong_pool_history.py --start 2026-02-01 --end 2026-02-26

# 查看当前数据库中已有数据统计
python scripts/download_strong_pool_history.py --stats
```

**限制**：每个 API 每天 200 次，两个 API 交替（每 20 次切换），合计 400 次/天，约可覆盖 20 个交易日。API 配额耗尽后次日继续运行即可断点续传。

**预期输出**：
```
strong_pool 统计：
  总记录数：134,830 条
  日期范围：2022-01-04 ~ 2026-02-25（525 个交易日）
```

---

## 第二步：构建训练数据

从强势池提取"首次入池"正样本，计算 T_feat 特征（入池日前 20 个交易日），并构建分层负样本。

```bash
# 标准构建（2023年至今，每月最多100个正样本）
python -X utf8 scripts/build_training_data.py \
    --start 2023-01-01 \
    --end 2026-01-31 \
    --month-samples 100

# 快速测试（仅用2024年数据）
python -X utf8 scripts/build_training_data.py \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --month-samples 50
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start` | 2023-01-01 | 正样本时间起点 |
| `--end` | （当前）| 正样本时间终点 |
| `--month-samples` | 100 | 每月最多抽取的正样本数 |
| `--neg-ratio` | 5.0 | 负样本与正样本比例 |
| `--precursor-n` | 20 | T_feat = T_entry - N（交易日） |

**输出**：
- `data/training_dataset.parquet` — 带标签的训练样本（约 8,000–12,000 行）
- `data/training_stats.json` — 正负样本统计、标签分布

**样本构成**（正负比约 1:5）：
- 正样本：首次入强势池的股票，在 `T_entry - 20` 日的特征，`is_strong_pos=1`
- 负样本（50%）：同板块、同期表现最差的股票（硬负例）
- 负样本（50%）：市场随机非强势股（软负例）

---

## 第三步：训练 LightGBM 模型

使用全部 37 个特征训练二分类模型（`is_strong_pos` 为目标标签）。

```bash
# 使用特征库中的特征（推荐，默认）
python -X utf8 -m models.lgb_classifier

# 使用全部 37 个特征（特征发现阶段）
python -X utf8 -m models.lgb_classifier --no-feature-library
```

**输出**：
- `models/saved/lgb_classifier.pkl` — 训练好的模型文件
- `models/saved/lgb_classifier.importance.csv` — 特征重要性（gain 排序）

**评估指标参考**（当前模型）：

| 指标 | 当前值 | 目标 |
|------|--------|------|
| AUC | 0.7618 | > 0.70 |
| Precision@10 | 90% | > 70% |
| Precision@20 | 75% | > 60% |

---

## 第四步：更新特征库

将模型训练后的特征重要性排名写入 `feature_library.json`，供后续扫描使用。

```bash
python -X utf8 scripts/update_feature_library.py
```

**输出**：`data/feature_library.json` — 按 gain 降序排列的 37 个特征。

> 如果某些特征的 `importance=0`，会在下次训练时自动跳过。

---

## 第五步：全市场扫描

对全市场（约 4,500 只股票）计算"入强势池概率"，输出 Top-K 候选名单。

```bash
# 扫描今天（最新 K 线日期）
python -X utf8 scripts/daily_scan.py

# 扫描指定日期
python -X utf8 scripts/daily_scan.py --date 2026-01-22

# 调整输出数量（默认 50）
python -X utf8 scripts/daily_scan.py --date 2026-01-22 --topk 30
```

**输出**：`results/scan_YYYYMMDD.csv`

**控制台示例输出**：
```
排名  代码           概率     主要特征
 1   002456.SZ     0.9124   价格压缩 / 量能加速 / 资金持续流入
 2   600123.SH     0.8876   布林带收窄 / PB历史低位 / 板块强势20日
...
```

---

## 第六步：批量历史回测扫描（定期运行）

对多个历史日期依次扫描，生成历史预测文件，用于回测验证。

```bash
# 默认：2023年至今，每隔 40 个交易日（约 2 个月）扫描一次
python -X utf8 scripts/batch_scan.py

# 自定义参数
python -X utf8 scripts/batch_scan.py \
    --start 2024-01-01 \
    --end 2025-12-20 \
    --step 20 \
    --topk 50

# 强制重新扫描（覆盖已有文件）
python -X utf8 scripts/batch_scan.py --force
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start` | 2023-01-01 | 回测起始日期 |
| `--end` | 2025-12-20 | 回测结束日期（需保留 22 个交易日做验证） |
| `--step` | 40 | 扫描间隔（交易日数） |
| `--topk` | 50 | 每次输出数量 |
| `--force` | False | 是否覆盖已有扫描结果 |

> 已有 `scan_*.csv` 的日期默认跳过，支持断点续跑。

---

## 第七步：验证预测准确性

计算历史扫描 Top-K 名单的实际收益率（持有 22 个交易日 ≈ 1 个月）。

```bash
# 验证所有历史扫描（批量）
python -X utf8 scripts/validate_predictions.py --all

# 验证指定日期
python -X utf8 scripts/validate_predictions.py --scan-date 2026-01-22

# 短持有期验证（数据不足 22 天时）
python -X utf8 scripts/validate_predictions.py --scan-date 2026-01-22 --forward-days 5

# 验证最新一次扫描（默认行为）
python -X utf8 scripts/validate_predictions.py
```

**输出**：
- `results/validation_YYYYMMDD.csv` — 每只股票的预测分与实际收益
- `results/validation_YYYYMMDD.png` — 分布图表
- `results/validation_summary.csv` — 所有日期的汇总指标（`--all` 时生成）

**指标解读**：

| 指标 | 含义 | 当前均值（21期） |
|------|------|------|
| `top10_avg_return` | Top-10 平均实际涨幅 | +4.65% |
| `top10_excess` | Top-10 超额收益（vs 市场） | +1.33% |
| `top10_beat_market` | Top-10 中跑赢大市的比例 | 71.4% |
| `spearman_corr` | 预测分与涨幅的秩相关 | 0.052 |

---

## 日常使用（每月节奏）

### 月初（约 1 小时）

```bash
conda activate rqsdk
cd D:\myCursor\QsgcTrain

# 1. 更新强势池数据
python scripts/download_strong_pool_history.py

# 2. 重建训练数据
python -X utf8 scripts/build_training_data.py --start 2023-01-01

# 3. 重训模型
python -X utf8 -m models.lgb_classifier

# 4. 更新特征库
python -X utf8 scripts/update_feature_library.py

# 5. 扫描最新日期
python -X utf8 scripts/daily_scan.py
```

### 验证上月预测（月末）

```bash
# 验证上个月扫描的预测结果
python -X utf8 scripts/validate_predictions.py --all
```

---

## 当前验证结果（2023–2025 回测，21期）

| 市场环境 | 期数 | Top-10 超额均值 | 胜率 |
|---------|------|-----------------|------|
| 上涨市（+3% 以上） | 10 | **+1.47%** | **80%** |
| 下跌市（-1% 以下） | 7  | **+2.32%** | **71%** |
| 震荡市 | 4  | -0.79% | 50% |
| **整体** | **21** | **+1.33%** | **71.4%** |

**结论**：模型在上涨市和下跌市均有超额收益，在震荡分化行情中表现较弱，需注意仓位控制。

---

## 数据库状态（2026-02-26）

| 数据表 | 最新日期 | 说明 |
|--------|---------|------|
| `strong_pool` | 2026-02-25 | 134,830 条，2022-01-04 起 |
| `kline_all` | 2026-01-30 | 日线行情（需外部更新） |
| `index_bar1d` | 2026-01-26 | 大盘指数日线 |
| `chips_all` | 2026-01-26 | 筹码分布 |
| `moneyflow` | 2026-01-16 | 个股资金流（更新较慢） |
| `valuation_all` | 2026-01-30 | 估值数据（PE/PB/市值等） |

> kline_all 等数据由 rqsdk 或其他数据源更新，需使用对应的数据导入脚本补充。  
> 2026-01-22 的完整 22 天验证（需数据到 2026-02-24）等待 kline 数据更新后运行。

---

## 特征说明（37 个）

当前模型使用的特征分为六类，通过 LightGBM feature importance 自动筛选：

| 类别 | 特征数 | 代表特征 | 含义 |
|------|--------|---------|------|
| A 趋势斜率 | 6 | `vol_slope_60d`, `mf_slope_10d` | 量能/价格/资金的趋势方向 |
| B 比率变化 | 8 | `range_compress`, `vol_ratio_5_20` | 近期相对历史的加速/压缩 |
| C 持续时间 | 3 | `consec_green`, `vol_spike_count` | 特定状态持续天数 |
| D 层级对比 | 4 | `dist_52w_high`, `price_vs_ma5` | 短期/长期位置比较 |
| E 强势池信号 | 3 | `prev_tj_boards`, `prev_pool_appearances` | 历史入池信息 |
| F 估值因子 | 6 | `log_float_cap`, `pe_ttm`, `pb` | 财务估值指标 |

**Top-5 最重要特征**（最新模型）：
1. `range_compress` — 振幅收窄，盘整蓄势
2. `log_float_cap` — 流通市值（中小盘偏好）
3. `vol_ratio_5_20` — 近期量能加速比
4. `mf_slope_10d` — 主力资金 10 日流入斜率
5. `bb_width` — 布林带宽度（越窄越好）

---

## 脚本速查表

| 脚本 | 用途 | 运行频率 |
|------|------|---------|
| `download_strong_pool_history.py` | 下载强势池 API 数据 | 每月（或配额允许时） |
| `build_training_data.py` | 生成训练样本 | 每月 |
| `models/lgb_classifier.py` | 训练模型 | 每月 |
| `update_feature_library.py` | 更新特征库 | 每次训练后 |
| `daily_scan.py` | 全市场扫描 | 每周（或需要时） |
| `batch_scan.py` | 批量历史扫描（回测用） | 数据更新后一次 |
| `validate_predictions.py` | 验证预测准确性 | 每月（验证上月预测） |
| `ic_analysis.py` | 特征 IC 分析 | 特征评估时 |
| `diagnose_dataset.py` | 训练集诊断 | 排查问题时 |
