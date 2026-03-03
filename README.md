# LLMQuant

LLMQuant 是一个面向量化因子挖掘的闭环工程，当前主流程是：

1. 用可配置的 `base` 因子库构建上下文（CSS + CoE）。
2. 调用 LLM 生成 DSL 因子表达式。
3. 计算新因子值与训练/验证指标（RIC + 相关性 + 复杂度）。
4. 走交集筛选规则，成功因子入正式库，全部因子进入成功/失败记忆库。

主入口：`scripts/workflow_llm_factors.py`

---

## 1. 快速开始

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 配置模型 API（可选）

默认配置在 `fama/config/defaults.yaml` 的 `llm` 段。  
如使用 SiliconFlow：

```bash
export SiliconFlow_API_KEY="your_key"
```

无可用 key 时会走 fallback，仅用于链路联调。

### 1.3 检查关键输入

- 行情文件：`data/fof_price_updating_intact.parquet`
- base 源文件：
`data/factor_cache_new/alpha101.yaml`、`data/factor_cache_new/alpha158.yaml`
- 配置文件：`fama/config/defaults.yaml`

### 1.4 启动 workflow

```bash
python scripts/workflow_llm_factors.py
```

可选覆盖参数（仅覆盖当次运行）：

```bash
python scripts/workflow_llm_factors.py \
  --config fama/config/defaults.yaml \
  --iterations 20 \
  --assets 000905.SH \
  --ric-threshold 0.08 \
  --corr-threshold 0.95 \
  --llm-self-corr-threshold 0.95 \
  --min-corr-obs 1000
```

---

## 2. 单轮流程（当前代码行为）

每一轮 `Iteration` 的执行顺序如下：

1. 解析配置与路径，合并 base 源，生成 `tmp/base_factor_cache_resolved.yaml`。
2. 创建 `PromptOrchestrator`：
   - 读取 base 源并计算 base 因子值到 `paths.factor_base_parquet`。
   - 按 `css.cluster_window_mode` 对 base 因子矩阵切窗做 CSS 聚类。
   - 按 `coe.ric_window_mode` + `coe.ric_asset_mode` 计算/加载 base RIC（`paths.factor_ric_base`）并构建 CoE 链。
3. 调 LLM 生成候选表达式，做 DSL 白名单校验后写入 LLM 缓存。
4. 仅对“本轮新增因子”计算因子值，输出 `paths.llm_factor_parquet`。
5. 计算训练集 RIC：输出 `paths.factor_ric_llm`。
6. 计算验证集 RIC：输出 `paths.factor_ric_llm_valid`。
7. 进入 `selection.run_selection_pipeline`（全量指标）：
   - 复杂度：`operator_count`、`nesting_depth`、`expression_size`
   - 训练相关性：`new_vs_base`、`new_vs_old_llm`、`new_vs_new_llm`
   - 验证相关性：同上三类
   - 汇总 `train_max_corr` / `valid_max_corr`（取 base 与 old_llm 的最大者）
8. 按 `selection.criteria` 做交集筛选（启用的规则必须全部通过）。
9. 写记忆库：
   - 成功：`paths.factor_success_cases`
   - 失败：`paths.factor_failure_cases`
10. 将通过筛选的因子追加到 `paths.llm_factor_library`。

说明：

- 主筛选是“交集规则”。
- 记忆库记录的是“全量评估结果”，不是只记录通过项。

---

## 3. 配置说明（defaults.yaml）

核心配置：`fama/config/defaults.yaml`

### 3.1 全局资产与时间窗

- `assets`：项目全局资产列表（RIC / CoE / selection 默认都从这里取）。
- `windows.train.start_date/end_date`：训练窗口。
- `windows.valid.start_date/end_date`：验证窗口。

### 3.2 CSS 与 CoE

- `css.cluster_window_mode`：`train|valid|full|custom`
- `css.cluster_start_date` / `css.cluster_end_date`：`custom` 模式下生效。

- `coe.ric_window_mode`：`train|valid|full|custom`
- `coe.ric_start_date` / `coe.ric_end_date`：`custom` 模式下生效。
- `coe.ric_asset_mode`：`global|custom`
- `coe.ric_assets`：`custom` 模式下生效。
- `coe.max_depth` / `coe.min_rankic` / `coe.prompt_chains`：CoE 链控制。

### 3.3 Selection（筛选）配置

- `selection.min_corr_obs`：相关性最小重叠样本数。
- `selection.require_full_asset_coverage`：RIC 是否要求覆盖全部资产。
- `selection.log_topk`：日志里展示 TopK 相关对。

- `selection.scope.asset_mode`：`global|custom`
- `selection.scope.assets`：`custom` 时使用。
- `selection.scope.train_window_mode`：`train|valid|full|custom`
- `selection.scope.valid_window_mode`：`train|valid|full|custom`
- `selection.scope.train_start_date/end_date`：`train_window_mode=custom` 时使用。
- `selection.scope.valid_start_date/end_date`：`valid_window_mode=custom` 时使用。

- `selection.criteria.*`：交集筛选规则。当前支持：
  - `train_min_abs_ric`
  - `train_max_abs_corr_base`
  - `train_max_abs_corr_old_llm`
  - `train_max_abs_corr_new_llm`
  - `valid_min_abs_ric`
  - `valid_max_abs_corr_base`
  - `valid_max_abs_corr_old_llm`
  - `valid_max_abs_corr_new_llm`
  - `max_operator_count`
  - `max_nesting_depth`

### 3.4 Base 源合并

- `base_catalog.selected_sources`：可选 `alpha101`、`alpha158`
- `base_catalog.include_llm_library_in_base`：是否把正式 LLM 库并入 base
- 运行时会合并到：`paths.base_factor_cache_resolved`

### 3.5 关键路径

- 行情：`paths.market_data`
- base 因子值：`paths.factor_base_parquet`
- base RIC：`paths.factor_ric_base`
- 本轮新因子值：`paths.llm_factor_parquet`
- 本轮训练/验证 RIC：`paths.factor_ric_llm` / `paths.factor_ric_llm_valid`
- 相关性输出（train/valid 三类）：`paths.corr_output_*`
- 正式入库：`paths.llm_factor_library`
- 记忆库：
  - `paths.factor_success_cases`
  - `paths.factor_failure_cases`

### 3.6 记忆库字段（核心）

记忆库由 `fama/memory/memory.py` 生成，成功与失败使用同一套字段。常用字段：

- 基础信息：`factor_id`、`round_id`、`batch_id`、`formula`、`formula_hash`
- 训练指标：`train_ic`、`train_ric`、`train_icir`、`train_max_corr`、`train_max_corr_factor_id`
- 验证指标：`valid_ic`、`valid_ric`、`valid_icir`、`valid_max_corr`、`valid_max_corr_factor_id`
- 复杂度：`operator_count`、`nesting_depth`、`expression_size`
- 决策结果：`final_status`、`failure_stage`、`failure_reason`

---

## 4. 目录与产物

```text
LLMQuant/
├── scripts/
│   ├── workflow_llm_factors.py
│   └── LLM_factors/
│       ├── dsl_LLM_factors_new.parquet
│       └── existing_llm_factors.parquet
├── fama/
│   ├── config/defaults.yaml
│   ├── mining/orchestrator.py
│   ├── selection/
│   └── memory/
├── utils/
│   ├── factor_collection_dsl.py
│   ├── ric_engine.py
│   ├── compute_correlation_new.py
│   ├── factor_catalog.py
│   ├── kun_backend.py
│   └── backtest_utils.py
├── data/
│   ├── factor_cache_new/
│   │   ├── alpha101.yaml
│   │   ├── alpha158.yaml
│   │   ├── LLM_factors.yaml
│   │   └── LLM_library.yaml
│   ├── base_factors/
│   └── memory/
│       ├── factor_success_cases.csv
│       └── factor_failure_cases.csv
├── factor_value_prepared/data/factors/
│   ├── dsl_factors_new.parquet
│   └── factor_ric.csv
└── tmp/
```

覆盖策略（默认）：

- `tmp/` 下多数文件按轮覆盖。
- `scripts/LLM_factors/dsl_LLM_factors_new.parquet` 按轮覆盖。
- `data/factor_cache_new/LLM_library.yaml` 追加。
- `data/memory/factor_success_cases.csv` / `data/memory/factor_failure_cases.csv` 追加。

---

## 5. 主要日志解读

你会在终端看到这些关键行：

- `Runtime params`：全局资产、selection 实际资产、训练阈值。
- `Criteria enabled`：当前启用/关闭的筛选规则。
- `Corr scope source`：train/valid 相关性窗口取值来源。
- `Window policy`（orchestrator）：CSS/CoE 的窗口与资产策略。
- `Metrics evaluated`：本轮候选、通过、失败数量。
- `Memory updated`：成功/失败记忆条数与目标文件。

---

## 6. 常见问题

### 6.1 `Skipping invalid expression ... 函数不在白名单`

代表 LLM 产物用了未允许算子（如 `TS_CORREL`），会被语法校验阶段过滤，不进入后续计算。

### 6.2 验证集 `rows=0` 的 RIC

常见原因是验证窗口内可用样本不足、或资产/时间无重叠。

### 6.3 为什么会看到 `new_vs_new_llm`

当本轮新因子数 `>=2` 时，selection 会计算新因子两两相关性用于去冗余。

### 6.4 `tmp/base_factor_cache_resolved.yaml` 是否每轮生成

每次 workflow 启动与 orchestrator 初始化都会按当前配置重建同一路径（覆盖写）。

---

## 7. 单模块调试

### 7.1 手动计算 DSL 因子值

```bash
python utils/factor_collection_dsl.py
```

默认行为：

- 优先读取 `tmp/base_factor_cache_resolved.yaml`。
- 若不存在会按 `base_catalog` 自动解析合并 base 源。
- 默认输出 `factor_value_prepared/data/factors/dsl_factors_new.parquet`。

### 7.2 手动计算相关性

```bash
python utils/compute_correlation_new.py \
  --llm-path scripts/LLM_factors/dsl_LLM_factors_new.parquet \
  --base-dir data/base_factors \
  --output-dir tmp
```

---

## 8. 开发约定

1. 路径、阈值、窗口、资产优先放 `defaults.yaml`。
2. `utils/` 放计算引擎，workflow 只做编排与落盘。
3. 因子长表统一列：`time/unique_id/factor_tag/value`。
4. 新功能优先保证：
   - 参数可控
   - 日志可观测
   - 模块可替换
