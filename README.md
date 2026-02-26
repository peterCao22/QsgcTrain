# QsgcTrain — 强势相似选股系统

以**强势股池快照**为参照，从A股全市场（排除科创/北交所/ST）检索特征相似的潜在标的，预测**未来30个交易日收盘收益**，并用**周频 Walk-forward 验证**评估策略稳定性。

---

## 旧项目根因与新项目对应解法

| 旧项目痛点 | 根因 | 新项目解法 |
|-----------|------|-----------|
| 硬筛选过严（-54%召回） | 7条AND规则无数据验证 | 取消前置硬筛选，全市场连续打分 |
| 模型Spearman仅0.13 | 3日标签信噪比极低 | 30日收盘收益，信噪比更高 |
| 标签与可买性不对齐 | 训练用涨幅，实盘看可买+涨幅 | 标签改为收盘收益，可追加可买性约束 |
| 特征有效性从未验证 | 经验堆砌导致大量噪声特征 | 每个特征先过单特征Spearman筛查 |

---

## 项目结构

```
QsgcTrain/
├── config.py                  # 全局配置（数据库/路径/超参）
├── data/
│   ├── db_loader.py           # PostgreSQL连接与通用查询
│   ├── market_loader.py       # 全市场日线/筹码/板块/龙虎榜
│   └── strong_pool.py         # 强势股池加载（API / 本地快照）
├── features/
│   ├── price_volume.py        # 价量特征（动量/量比/K线形态）
│   ├── chip.py                # 筹码特征（换手率/成本集中度）
│   ├── sector.py              # 板块特征（强度/资金流）
│   └── builder.py             # 特征合并、标准化、Spearman筛查
├── models/
│   ├── baseline.py            # 路线A：LightGBM LambdaRank横截面排序
│   └── similarity.py          # 路线B：强势原型余弦相似度打分
├── validation/
│   ├── labels.py              # 30日收盘收益标签（无泄漏）
│   └── walkforward.py         # 周频Walk-forward验证框架
├── pipeline.py                # 主流程：特征→标签→训练→验证→报告
├── requirements.txt
├── .env.example
└── README.md
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 复制并填写环境变量
cp .env.example .env
# 编辑 .env，填入 PostgreSQL 密码和强势股池 API Key

# 3. 准备强势股池快照（离线模式）
# 在 data/strong_pool_snapshots/ 目录下放置 CSV 文件
# 格式：date,instrument（每行一只，date为入池日期）
# 示例文件名：2025-01-20.csv

# 4. 运行完整 Pipeline
python pipeline.py --start_date 2023-01-01 --end_date 2025-12-31

# 5. 验证单个日期的选股结果
python pipeline.py --mode predict --date 2025-01-20 --topk 20
```

---

## 双路线架构

```
[强势股池 API / 本地快照]  ──  t日快照（point-in-time）
          │
          ▼
[特征层]  ──  全市场 × 特征向量
          │       价量动量（5/10/20/60日）
          │       筹码分布（换手率/集中度）
          │       板块强度（资金流/排名）
          │       龙虎榜（近N日机构净买入）
          │
          ├──  路线A：直接横截面打分
          │        LightGBM LambdaRank
          │        30日收盘收益为排序目标
          │
          └──  路线B：强势相似打分
                   强势原型（强势股池特征质心）
                   全市场余弦相似度 → 分数
          │
          ▼
[融合]  ──  A + B 加权打分 → TopK候选
          │
          ▼
[Walk-forward验证]  ──  每周1次 × 30日收益
          指标：TopK均值 / 胜率 / 超额收益 / 最大回撤
```

---

## 验证协议

- **时间范围**：2023–2025年（建议覆盖牛/震荡/跌至少2种市场状态）
- **验证频率**：每周1次（如每周五收盘后选股）
- **评估指标**：
  - `mean_return`：TopK 30日均收益
  - `win_rate`：正收益比例
  - `excess_return`：相对等权全市场的超额收益
  - `max_drawdown`：t→t+30 区间最大回撤
  - `ic`：预测分与实际收益的 Spearman 相关系数

---

## 关键设计原则

1. **Point-in-time 严格对齐**：所有特征只能用 t 日收盘后可得的数据，强势股池快照时间戳必须 ≤ t
2. **特征有效性前置验证**：每个特征先算单特征 Spearman，< 0.02 的特征不进模型
3. **标签清洁**：30日收盘收益，训练集可追加"t+1可买"约束（排除涨停/停牌）
4. **无前置硬筛选**：全市场打分，让模型和相似度分数说话
