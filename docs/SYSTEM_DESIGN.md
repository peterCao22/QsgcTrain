# QsgcTrain 系统设计文档

**版本**：v1.0  
**创建日期**：2026-02-24  
**项目目录**：`D:\myCursor\QsgcTrain`

---

## 一、旧项目根因与新项目对应解法

旧项目（AlphaSignalCN V3）的核心问题（来自 `ROOT_CAUSE_ANALYSIS.md`）：

| 问题 | 损失占比 | 新项目解法 |
|------|---------|-----------|
| 硬筛选过严（AND规则）| 54% | 无前置硬筛选，全市场连续打分 |
| Spearman 仅 0.13，预测力弱 | 36% | 改用"蓄力期特征"，信号更清晰 |
| 标签与可买性不对齐 | 10% | 标签对齐到特征提取时间点 |
| 特征堆砌无验证（78维）| - | 先验证单特征 IC，再进模型 |

---

## 二、核心思路

### 关键认知

强势股池（每天 API 提供的突破新高/连续涨停股票）**已经涨了**。

目标不是买这些股票，而是：

> **找出当前市场中，"现在的状态"与这些强势股"当初变强之前 30~60 天的状态"相似的股票**

这批股票尚未启动，但具备与强势股相同的"蓄力特征"，预测它们在未来 1 个月会有强势涨幅。

### 时间轴

```
T-60        T-45        T-30         T            T+30
 │           │           │            │             │
 └───────────┴───────────┘            │             │
       特征提取窗口                强势股入池         验证期
   （股票还没有开始大涨）          （已突破/连板）    （检验预测）

模型学习：T-N 时的特征 → 预测股票是否会在 T 时变为强势
预测应用：当前市场哪些股票处于"T-N 状态" → 这批股票未来会变强
```

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    数据输入层                             │
│  强势股池快照（每周 API）→ strong_pool 表                  │
│  全市场日线数据（已有）→ kline_all 表                      │
│  筹码/资金流/板块（已有）→ chips_all / moneyflow 等         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    特征工程层                             │
│  在 T-N 日（N=45天）提取"蓄力期特征"                       │
│  多时间窗口：5/20/60/120/250 日                           │
│  4类动态特征：趋势斜率 / 比率变化 / 持续时间 / 层级对比     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    模型训练层                             │
│  Phase 1：LightGBM（分类 + 排序，快速验证）                │
│  Phase 2：LSTM（数据充足后，捕捉更细腻时序模式）            │
│  持续学习：每月追加新样本重训，滑动12月窗口                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    预测与验证层                           │
│  每周扫描全市场（含强势股本身，不过滤）                    │
│  输出 TopK 候选 → 30日后验证实际涨幅                      │
│  闭环：有效因子保留，无效因子移除                          │
└─────────────────────────────────────────────────────────┘
```

---

## 四、数据层设计

### 4.1 strong_pool 表（新增）

存储每天 API 返回的强势股池快照。

```sql
CREATE TABLE strong_pool (
    id           SERIAL PRIMARY KEY,
    date         DATE         NOT NULL,          -- 快照日期
    instrument   VARCHAR(20)  NOT NULL,          -- 股票代码（标准格式：600693.SH）
    name         VARCHAR(100),                   -- 股票名称
    price        DECIMAL(12,4),                  -- 当前价格
    limit_price  DECIMAL(12,4),                  -- 涨停价
    pct_change   DECIMAL(10,4),                  -- 当日涨幅%
    amount       DECIMAL(20,2),                  -- 成交额（元）
    float_cap    DECIMAL(20,2),                  -- 流通市值（元）
    total_cap    DECIMAL(20,2),                  -- 总市值（元）
    speed        DECIMAL(10,4),                  -- 涨速
    new_high     SMALLINT     DEFAULT 0,         -- 是否新高（0/1）
    vol_ratio    DECIMAL(10,4),                  -- 量比
    turnover     DECIMAL(10,4),                  -- 换手率%
    tj_days      SMALLINT,                       -- 涨停统计-观察天数（tj字段X）
    tj_boards    SMALLINT,                       -- 涨停统计-涨停板数（tj字段Y）
    source       VARCHAR(20)  DEFAULT 'api',     -- 数据来源
    created_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, instrument)
);
```

**API 字段映射**：

| API 字段 | 含义 | 转换规则 |
|---------|------|---------|
| `dm` | 股票代码 | `sh600693` → `600693.SH`，`sz000001` → `000001.SZ` |
| `mc` | 股票名称 | 直接存储 |
| `p` | 当前价 | → `price` |
| `ztp` | 涨停价 | → `limit_price` |
| `zf` | 涨幅% | → `pct_change` |
| `cje` | 成交额 | → `amount` |
| `lt` | 流通市值 | → `float_cap` |
| `zsz` | 总市值 | → `total_cap` |
| `zs` | 涨速 | → `speed` |
| `nh` | 是否新高 | → `new_high` |
| `lb` | 量比 | → `vol_ratio` |
| `hs` | 换手率% | → `turnover` |
| `tj` | 涨停统计"X天/Y板" | 解析为 `tj_days=X`, `tj_boards=Y` |

### 4.2 现有表（复用）

| 表名 | 用途 |
|------|------|
| `kline_all` | 日线价量数据（复权）|
| `chips_all` | 筹码分布（avg_cost / win_percent / concentration）|
| `moneyflow` | 个股资金流 |
| `concept_bar1d` | 板块/概念日线 |
| `price_limit_status` | 涨跌停/停牌状态 |
| `stock_list` | 股票基础信息 |

---

## 五、特征工程规范

### 5.1 核心原则

- **特征提取时间点**：`T_feat = T_strong - N`（N=45天，待实验验证30/45/60）
- **Point-in-time 严格对齐**：只能用 T_feat 日收盘后可得的数据
- **时序感知**：不是单点截面，而是描述"从过去到 T_feat 的演变趋势"

### 5.2 完整特征列表（约22个）

#### A 类：趋势斜率（这个方向在变好吗？）

| 特征名 | 计算方式 | 经济含义 |
|--------|---------|---------|
| `vol_slope_60d` | 成交量60日线性回归斜率（归一化） | 量能在缓慢积累 |
| `price_slope_60d` | 价格60日斜率（vs大盘相对） | 价格温和上升趋势 |
| `mf_slope_30d` | 主力净流入30日斜率 | 主力持续介入 |
| `sector_rank_slope_20d` | 板块强度排名近20日改善速度 | 所属板块开始升温 |

#### B 类：比率变化（现在比过去强了多少？）

| 特征名 | 计算方式 | 经济含义 |
|--------|---------|---------|
| `vol_accel` | 近20日均量 / 近60日均量 | 量能加速比（>1 = 放量） |
| `range_compress` | 近20日ATR / 近60日ATR | 振幅收窄（<1 = 盘整压缩） |
| `momentum_accel` | 近20日收益 - 近60日收益/3 | 近期动量加速度 |
| `excess_ret_change` | 近20日超额收益 - 近60日超额收益 | 相对强度在提升 |

#### C 类：持续时间（这个好状态持续多久了？）

| 特征名 | 计算方式 | 经济含义 |
|--------|---------|---------|
| `days_above_ma20` | 连续站上MA20的天数 | 支撑有效性 |
| `days_sector_strong` | 板块连续强于大盘的天数 | 板块轮动持续性 |
| `consec_higher_low` | 连续创更高低点的天数 | 上升通道确认 |
| `days_positive_mf` | 主力净流入连续天数 | 主力持续积累 |

#### D 类：层级对比（短期 vs 长期位置）

| 特征名 | 计算方式 | 经济含义 |
|--------|---------|---------|
| `dist_52w_high` | 1 - 当前价/近252日最高价 | 距年高点还有多少空间 |
| `dist_60d_low_rebound` | 当前价/近60日最低价 - 1 | 从低点反弹了多少 |
| `win_percent` | 筹码盈利比例（chips_all） | 获利盘压力 |
| `chip_concentration` | 筹码集中度 | 锁仓情况 |
| `sector_rank_pct` | 板块强度在全板块中的分位 | 板块位置感知 |

#### E 类：直接信号（来自strong_pool API字段）

| 特征名 | 来源 | 经济含义 |
|--------|------|---------|
| `prev_tj_boards` | strong_pool.tj_boards | 近N月内曾有几板 |
| `prev_new_high_count` | strong_pool.new_high | 近N月内出现新高次数 |
| `prev_pool_appearances` | strong_pool 出现次数 | 历史上进过几次强势池 |

### 5.3 特征有效性验证标准

每次训练前自动计算每个特征的 Spearman IC（与30日实际收益的相关性）：

- `|IC| ≥ 0.05`：强有效，进模型
- `0.02 ≤ |IC| < 0.05`：弱有效，观察期，不立刻移除
- `|IC| < 0.02`：无效，从下次训练起移除

---

## 六、模型设计

### Phase 1：LightGBM（立即可用）

```
训练方式 A（分类）：
  正样本：进入强势池的股票在 T-N 时的特征
  负样本：同期未进入强势池的全市场股票特征
  标签：涨幅 > 15% = 1，其他 = 0
  模型：LightGBM Classifier
  输出："成为强势股"的概率分

训练方式 B（排序）：
  标签：30日实际收益率（连续值）
  模型：LightGBM LambdaRank
  输出：预测收益排序分数

对比两种方式，选择 IC/AUC 更高的作为主方案
```

**超参数（初始值，后续通过 Optuna 调优）**：

```python
{
    "objective": "binary",          # Phase 1A
    "learning_rate": 0.05,
    "num_leaves": 31,               # 保持简单，防止过拟合
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.5,
    "n_estimators": 300,
}
```

### Phase 2：LSTM（数据积累12个月后）

```
输入：每只股票在 T_feat 往前120个交易日的日线时序
      shape = (120, n_features_daily)
输出：30日涨幅预测

适用条件：
  - 训练样本 > 5000 个正样本（约2年数据）
  - LightGBM 遇到瓶颈（IC 无法继续提升）
  
与 LightGBM 融合方式：
  final_score = 0.6 * lgb_score + 0.4 * lstm_score
```

---

## 七、持续学习机制

```
Week 0  → 存档强势池 S0（50只）
Week 1  → 存档强势池 S1
Week 2  → 存档强势池 S2
Week 3  → 存档强势池 S3
...
Week 4  → S0 的30日后标签可以计算了 → 生成训练样本 B0
Week 5  → S1 生成训练样本 B1
...
每积累 4批（约1个月）→ 触发重训练：
  1. 追加 B0..Bn 到训练集
  2. 滑动窗口保留近12个月样本
  3. 重新计算特征 IC，输出特征报告
  4. 更新模型
  5. 用最新模型对全市场打分，输出 TopK

滚动验证：
  每次重训后，对比上月预测的 TopK 实际表现
  更新特征权重报告
```

---

## 八、全市场预测规范

### 预测范围

- **包含**：A 股主板 + 中小板 + 创业板
- **排除**：科创板（688xxx）、北交所（8xxxxx / 4xxxxx）、ST / 退市、停牌
- **特别说明**：强势股池本身的股票**不排除**——它们如果充分调整后特征重新符合"蓄力模式"，会自然出现在 TopK 中

### 输出格式

每次预测输出两份列表：

1. **新机会列表**：不在当前强势池中，但高分的股票（尚未启动）
2. **延续机会列表**：当前在强势池中，且仍然高分的股票（可能继续强势）

---

## 九、验证体系

### 验证指标

| 指标 | 计算方式 | 目标阈值 |
|------|---------|---------|
| TopK 30日平均收益 | TopK 所有股票30日收益均值 | > 8% |
| 胜率 | 正收益股票比例 | > 60% |
| 超额收益 | TopK均值 - 等权全市场均值 | > 5% |
| IC 均值 | 预测分与实际收益的 Spearman | > 0.05 |
| 与强势池重合率 | TopK中后来进入强势池的比例 | 参考指标 |

### 验证时序

```
第1次预测（T0）
  ↓ 30日后（T0+30）
验证第1次预测结果 → 生成验证报告
  ↓
同时触发增量训练（加入新数据）
  ↓
第2次预测（T0+30）
  ↓ 30日后...
```

---

## 十、关键参数（待实验确定）

| 参数 | 候选值 | 确定方式 |
|------|--------|---------|
| N（特征提取偏移天数） | 30 / 45 / 60 | 哪个 IC 最高 |
| 正样本涨幅阈值 | 10% / 15% / 20% | 哪个 AUC 最好 |
| 训练窗口大小 | 6个月 / 9个月 / 12个月 | 验证集表现 |
| 特征数量 | 15~22 | IC 筛查后确定 |

---

## 十一、目录结构

```
QsgcTrain/
├── docs/
│   └── SYSTEM_DESIGN.md              ← 本文档
├── config.py                         全局配置（DB/API/特征/模型超参）
├── pipeline.py                       主流程入口（walkforward/predict/report）
├── README.md                         项目说明
│
├── scripts/                          一次性/离线脚本
│   ├── create_strong_pool_table.py   建表：strong_pool
│   ├── import_index_bar1d.py         建表+导入：大盘指数日K线
│   ├── download_strong_pool_history.py  下载历史强势股池（智兔+魔码云服交替）
│   ├── build_training_data.py        生成带标签训练样本 → data/training_dataset.parquet
│   └── ic_analysis.py                单特征 IC 报告 → results/ic_report.csv + 图表
│
├── data/                             数据加载层
│   ├── db_loader.py                  数据库连接、交易日历、可交易股票域
│   ├── market_loader.py              全市场数据（kline/chips/moneyflow/指数等）
│   └── strong_pool.py                强势股池加载（DB/API/本地CSV）
│
├── features/                         特征工程层
│   ├── precursor.py                  ★ 蓄力期特征（核心）：A/B/C/D/E 五类共18个
│   ├── builder.py                    截面特征构建器（旧版，供参考）
│   ├── price_volume.py               价量特征（旧版）
│   ├── chip.py                       筹码特征（旧版）
│   └── sector.py                     板块特征（旧版）
│
├── models/                           模型层
│   ├── lgb_classifier.py             ★ Phase 1A：LightGBM 二分类（主方案）
│   ├── baseline.py                   LightGBM LambdaRank（对比方案）
│   └── similarity.py                 相似度评分器（强势原型余弦相似度）
│
├── validation/                       验证层
│   ├── labels.py                     30日收盘收益标签生成
│   └── walkforward.py                Walk-forward 滚动回测
│
├── data/                             数据文件（自动生成，不提交 git）
│   ├── training_dataset.parquet      训练样本集
│   ├── training_stats.json           训练集统计摘要
│   └── strong_pool_snapshots/        本地快照 CSV（离线使用）
│
├── models/                           模型文件（自动生成，不提交 git）
│   ├── lgb_classifier.pkl            训练好的分类模型
│   └── lgb_classifier.importance.csv 特征重要性
│
├── results/                          预测结果存档（自动生成）
│   ├── ic_report.csv                 特征 IC 分析报告
│   ├── ic_report.png                 IC 条形图
│   └── ic_timeseries.png             IC 时序图
│
└── logs/                             运行日志
```

---

## 十二、开发路线图

| 阶段 | 内容 | 状态 | 说明 |
|------|------|------|------|
| **Phase 0** | 建表、下载历史强势股池数据、导入大盘指数 | ✅ 完成（2026-02-25） | strong_pool 164天/57,206条；index_bar1d 743条（2023~2026）；剩余69天明日补跑 |
| **Phase 1** | 蓄力期特征工程（precursor.py）+ 训练样本生成 + 单特征 IC 验证 | 🔄 进行中 | `features/precursor.py`（18个特征）、`scripts/build_training_data.py`、`scripts/ic_analysis.py` 已完成编码；待执行生成样本 |
| **Phase 2** | LightGBM 分类模型训练 + Walk-forward 验证 | ⏳ 待开始 | `models/lgb_classifier.py` 已编码；等 Phase 1 IC 报告确认有效特征后训练 |
| **Phase 3** | 持续学习闭环：每月追加数据 + 重训 | ⏳ 待开始 | 目标指标：TopK 胜率 > 55%，IC > 0.05 |
| **Phase 4** | LSTM 实验对比 | ⏳ 待开始 | 需正样本 > 5000 条（约积累2年数据后） |

### 当前待执行步骤

```bash
# 1. 补全剩余 69 天强势股池数据（明日 API 配额重置后）
python scripts/download_strong_pool_history.py --sleep 0.3

# 2. 生成训练样本（约 5~10 分钟）
python scripts/build_training_data.py

# 3. 查看特征 IC 报告（决策点：IC < 0.02 的特征需调整）
python scripts/ic_analysis.py

# 4. 训练 LightGBM 分类模型
python models/lgb_classifier.py
```
