# LLMQuant

LLMQuant 是一个面向量化因子挖掘的工程化闭环：
- 用可配置的 `base` 因子库构建上下文（CSS + CoE）。
- 调用 LLM 生成新因子表达式（DSL）。
- 计算新因子值、RIC、相关性并做筛选。
- 通过筛选后写入 `LLM_library`，进入后续迭代。

推荐入口：`scripts/workflow_llm_factors.py`

---

## 1. 快速开始（3 分钟）

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 配置 API Key（可选）

默认使用 SiliconFlow，在 `fama/config/defaults.yaml` 里配置。

```bash
export SiliconFlow_API_KEY="your_key"
```

无可用 key 时，`fama/mining/llm_client.py` 会走 fallback（用于链路联调，不建议长期用于实盘研究）。

### 1.3 检查关键输入

- 行情文件存在：`data/fof_price_updating_intact.parquet`
- base 因子源存在：`data/factor_cache_new/alpha101.yaml`、`data/factor_cache_new/alpha158.yaml`
- `fama/config/defaults.yaml` 中路径与本机目录一致

### 1.4 运行 workflow

```bash
python scripts/workflow_llm_factors.py
```

可选覆盖参数示例：

```bash
python scripts/workflow_llm_factors.py \
  --config fama/config/defaults.yaml \
  --iterations 20 \
  --ric-threshold 0.08 \
  --corr-threshold 0.95 \
  --llm-self-corr-threshold 0.95 \
  --min-corr-obs 1000
```

---

## 2. 项目流程（单轮迭代）

1. 读取配置并解析路径。
2. 根据 `base_catalog.selected_sources` 合并 base 源，生成 `tmp/base_factor_cache_resolved.yaml`。
3. `PromptOrchestrator.run(use_css=True, use_coe=True)`：基于 base 因子值与 RIC 构造 CSS/CoE 上下文并请求 LLM。
4. 将本轮新因子写入 `tmp/llm_factor_cache_tmp.yaml`。
5. 仅计算本轮新因子值，输出 `scripts/LLM_factors/dsl_LLM_factors_new.parquet`。
6. 计算本轮新因子 RIC，输出 `tmp/llm_factor_ric.csv`。
7. 做 `new-vs-base` 相关性筛选。
8. 做 `new-vs-old-llm` 相关性筛选。
9. 做 `new-vs-new-llm` 去重。
10. 通过筛选后追加到 `data/factor_cache_new/LLM_library.yaml`。

---

## 3. 目录说明（关键文件）

```text
LLMQuant/
├── scripts/
│   ├── workflow_llm_factors.py         # 主流程入口
│   └── LLM_factors/
│       ├── dsl_LLM_factors_new.parquet # 本轮新因子值（覆盖）
│       └── existing_llm_factors.parquet# 历史 LLM 因子值（筛选阶段生成）
├── fama/
│   ├── config/defaults.yaml            # 统一配置盘
│   ├── mining/orchestrator.py          # CSS/CoE/Prompt/LLM 编排
│   ├── mining/llm_client.py            # LLM 客户端与 fallback
│   └── data/factor_space.py            # Factor / FactorSet YAML 结构
├── utils/
│   ├── factor_catalog.py               # 合并 base 源（alpha101/alpha158/...）
│   ├── factor_collection_dsl.py        # DSL 因子值计算（KunQuant）
│   ├── kun_backend.py                  # DSL AST -> KunQuant 执行
│   ├── ric_engine.py                   # 统一 RIC/IC/ICIR 计算引擎
│   ├── compute_correlation_new.py      # Spearman 相关性计算
│   ├── factor_collection.py            # 因子集合基类
│   └── backtest_utils.py               # prepare_price_data 等基础工具
├── data/
│   ├── fof_price_updating_intact.parquet
│   ├── base_factors/                   # base 因子值目录（相关性 reference）
│   └── factor_cache_new/
│       ├── alpha101.yaml               # base 源：alpha101
│       ├── alpha158.yaml               # base 源：alpha158
│       ├── LLM_factors.yaml            # LLM 工作缓存
│       └── LLM_library.yaml            # LLM 正式入库库（持久）
├── factor_value_prepared/
│   └── data/factors/
│       ├── dsl_factors_new.parquet     # base 因子值长表（供 CSS/CoE）
│       └── factor_ric.csv              # base 因子 RIC（供 CSS/CoE）
└── tmp/                                # workflow 中间产物（大多覆盖）
```

---

## 4. 配置说明（defaults.yaml）

核心配置文件：`fama/config/defaults.yaml`

### 4.1 workflow / selection 超参数

- `workflow.iterations`：迭代轮数
- `selection.ric_threshold`：新因子 RIC 门槛（按目标资产逐个校验）
- `selection.corr_threshold`：new-vs-base 最大允许相关性
- `selection.llm_self_corr_threshold`：new-vs-old/new-vs-new 最大允许相关性
- `selection.min_corr_obs`：相关性最小重叠样本
- `selection.complexity.enabled / max_ops / max_depth`：复杂度过滤开关与阈值（在 RIC+相关性筛选后执行）

兼容说明：
- `workflow.ric_threshold / corr_threshold / llm_self_corr_threshold / min_corr_obs` 仍可作为回退键，但建议统一迁移到 `selection.*`。

### 4.2 RIC 参数

- `ric.assets`
- `ric.min_obs`
- `ric.start_date`
- `ric.end_date`

### 4.3 base 源配置

- `base_catalog.selected_sources: ["alpha101", "alpha158"]`
- `base_catalog.include_llm_library_in_base`

运行时会把选定源合并到：
- `paths.base_factor_cache_resolved`（默认 `tmp/base_factor_cache_resolved.yaml`）

### 4.4 关键路径

- 行情：`paths.market_data`
- base 因子值：`paths.factor_base_parquet`
- base RIC：`paths.factor_ric_base`
- 本轮 LLM 因子值：`paths.llm_factor_parquet`
- 本轮 LLM RIC：`paths.factor_ric_llm`
- 相关性输出：`paths.corr_output_new_vs_base` / `paths.corr_output_new_vs_old_llm` / `paths.corr_output_new_vs_new_llm`
- 入库目标：`paths.llm_factor_library`

---

## 5. 数据格式约定

### 5.1 因子 YAML（FactorSet）

每个因子条目包含：
- `name`
- `expression`
- `explanation`（可空）
- `references`（可空）

实现：`fama/data/factor_space.py`

### 5.2 因子值长表（统一列）

- `time`
- `unique_id`
- `factor_tag`
- `value`

### 5.3 RIC 输出

`utils/ric_engine.py` 输出（csv/parquet）常用字段：
- `unique_id` / `asset`
- `factor_tag`
- `ric` / `abs_ric`
- `sample_count`
- `start_date` / `end_date`
- 可选：`ic` / `icir`

### 5.4 相关性输出

`utils/compute_correlation_new.py` 输出字段：
- `llm_factor`
- `base_factor`
- `weighted_corr`
- `abs_corr`
- `total_obs`
- `asset_pairs`

---

## 6. 产物与覆盖策略

### 6.1 相对稳定的基础产物

- `factor_value_prepared/data/factors/dsl_factors_new.parquet`
- `factor_value_prepared/data/factors/factor_ric.csv`

这两份通常用于 CSS/CoE 的 base 上下文。

### 6.2 每轮覆盖的中间产物

- `tmp/llm-factor.yaml`
- `tmp/llm_factor_cache_tmp.yaml`
- `scripts/LLM_factors/dsl_LLM_factors_new.parquet`
- `tmp/llm_factor_ric.csv`
- `tmp/new_llm_vs_base_corr.csv`
- `tmp/new_llm_vs_old_llm_corr.csv`
- `tmp/new_llm_vs_new_llm_corr.csv`

### 6.3 持久化入库

- `data/factor_cache_new/LLM_library.yaml`

仅通过筛选的因子会追加到这里。

---

## 7. 常见问题

### 7.1 为什么有表达式被跳过？

常见原因：
- 使用了不在白名单中的算子。
- 使用了 `>` / `<` / `>=` / `<=`，而不是 `GT/GE/LT/LE/EQ`。
- 参数个数不符合约束。

### 7.2 为什么没有因子入库？

可能卡在任一关：
- RIC 不达标
- new-vs-base 相关性过高
- new-vs-old/new-vs-new 去重淘汰

### 7.3 `base_factor_cache_resolved.yaml` 什么时候生成？

在 workflow 初始化与 orchestrator 初始化阶段都会触发 base 源解析，写入同一路径（覆盖）。

### 7.4 相关性筛选的时间段由谁控制？

与 RIC 一致，来自：
- `ric.start_date`
- `ric.end_date`

并可被命令行 `--ric-start` / `--ric-end` 覆盖。

---

## 8. 单模块运行（调试用）

### 8.1 单独计算 DSL 因子值

```bash
python utils/factor_collection_dsl.py
```

说明：
- `__main__` 默认按 `batch_size=500` 运行。
- 默认读取 base 缓存（解析后路径），输出到 base 因子目录。

### 8.2 单独计算相关性

```bash
python utils/compute_correlation_new.py \
  --llm-path scripts/LLM_factors/dsl_LLM_factors_new.parquet \
  --base-dir data/base_factors \
  --output-dir tmp
```

注意：该脚本的默认 CLI 路径仍带历史示例，建议始终显式传参。

---

## 9. 开发约定

1. 路径与阈值优先在 `defaults.yaml` 配置，不在代码里硬编码。
2. 计算引擎能力尽量统一到 `utils/`，workflow 只做编排。
3. 长表列标准统一为 `time/unique_id/factor_tag/value`。
4. 新增功能优先保证“可替换、可观测、可复用”（日志、参数、模块边界清晰）。
