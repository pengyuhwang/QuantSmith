# LLMQuant

LLMQuant 是一个面向量化因子挖掘的闭环工程，当前主流程是：

1. 用可配置的 `base` 因子库构建上下文（CSS + CoE）。
2. 可选地读取近期 `round_memory.csv`，由检索 LLM 生成“记忆检索引导”并注入主 prompt。
3. 调用主 LLM 生成 DSL 因子表达式。
4. 计算新因子值与训练/验证指标（RIC + 相关性 + 复杂度）。
5. 走交集筛选规则，成功因子入正式库，全部因子进入成功/失败记忆库。
6. 生成本轮 `Round Packet`，并落盘到 `round_memory.csv`；若记忆模块开启，再由轮后复盘 LLM 写入结构化总结。

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
2. 若 `memory.enabled=true` 且已有 `paths.round_memory_csv`：
   - 读取最近若干轮 `round_memory.csv`
   - 调用 retrieval planner LLM，生成“记忆检索引导”
   - 可通过本地因子库查询工具补充佐证因子（返回因子名、表达式、解释、references）
   - 将检索结果注入主 prompt，并把检索输入保存到 `tmp/retrieval_planner_prompt.json`
3. 创建 `PromptOrchestrator`：
   - 读取 base 源并计算 base 因子值到 `paths.factor_base_parquet`。
   - 按 `css.cluster_window_mode` 对 base 因子矩阵切窗做 CSS 聚类。
   - 按 `coe.ric_window_mode` + `coe.ric_asset_mode` 计算/加载 base RIC（`paths.factor_ric_base`）并构建 CoE 链。
4. 调主 LLM 生成候选表达式，做 DSL 白名单校验后写入 LLM 缓存。
5. 仅对“本轮新增因子”计算因子值，输出 `paths.llm_factor_parquet`。
6. 计算训练集 RIC：输出 `paths.factor_ric_llm`。
7. 计算验证集 RIC：输出 `paths.factor_ric_llm_valid`。
8. 进入 `selection.run_selection_pipeline`（全量指标）：
   - 复杂度：`operator_count`、`nesting_depth`、`expression_size`
   - 训练相关性：`new_vs_base`、`new_vs_old_llm`、`new_vs_new_llm`
   - 验证相关性：同上三类
   - 汇总 `train_max_corr` / `valid_max_corr`（取 base 与 old_llm 的最大者）
9. 按 `selection.criteria` 做交集筛选（启用的规则必须全部通过）。
10. 写因子级记忆库：
   - 成功：`paths.factor_success_cases`
   - 失败：`paths.factor_failure_cases`
11. 生成本轮 `Round Packet`，总是追加到 `paths.round_memory_csv`：
   - 包含轮级计数、成功/失败模式分布、以及本轮全部因子卡片
   - 若 `memory.enabled=true`，再调用 round analyst LLM 生成 `round_reflection_json`
   - round analyst 的输入保存到 `tmp/round_analyst_prompt.json`
12. 将通过筛选的因子追加到 `paths.llm_factor_library`。

说明：

- 主筛选是“交集规则”。
- 记忆库记录的是“全量评估结果”，不是只记录通过项。
- `round_memory.csv` 是轮级账本；即使 `memory.enabled=false` 也会继续写入，只是 `round_reflection_json` 为空对象。

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

### 3.5 记忆模块

- `memory.enabled`
  - 是否启用两个 memory LLM：
    - retrieval planner（轮前）
    - round analyst（轮后）
- `memory.round_analyst.recent_context_rounds`
  - 轮后复盘时，附带最近多少轮的简短上下文
- `memory.retrieval_planner.recent_rounds`
  - 轮前检索读取最近多少轮 `round_memory.csv`
- `memory.retrieval_planner.top_pass_rounds`
  - 额外强调通过率较高的轮次
- `memory.retrieval_planner.top_duplicate_rounds`
  - 额外强调重复失败较多的轮次
- `memory.retrieval_planner.tool_max_calls`
  - retrieval planner 最多连续调用因子库查询工具的次数
- `memory.retrieval_planner.tool_result_limit`
  - 单次工具查询返回的最大因子数

说明：

- `memory.enabled=false` 时：
  - 不调用 retrieval planner / round analyst
  - 但仍然会写 `round_memory.csv`

### 3.6 关键路径

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
  - `paths.round_memory_csv`

运行时还会覆盖写两个调试文件（不走配置）：

- `tmp/retrieval_planner_prompt.json`
- `tmp/round_analyst_prompt.json`

### 3.7 因子级记忆库字段（核心）

记忆库由 `fama/memory/memory.py` 生成，成功与失败使用同一套字段。常用字段：

- 基础信息：`factor_id`、`round_id`、`batch_id`、`formula`、`formula_hash`
- 训练指标：`train_ic`、`train_ric`、`train_icir`、`train_max_corr`、`train_max_corr_factor_id`
- 验证指标：`valid_ic`、`valid_ric`、`valid_icir`、`valid_max_corr`、`valid_max_corr_factor_id`
- 复杂度：`operator_count`、`nesting_depth`、`expression_size`
- 决策结果：`final_status`、`failure_stage`、`failure_reason`

### 3.8 轮级记忆（`round_memory.csv`）

轮级记忆由 `fama/memory/round_memory.py` 生成，一轮一行。核心列：

- 基础计数：
  - `round_id`
  - `batch_id`
  - `generated_count`
  - `success_count`
  - `pass_rate`
  - `weak_signal_fail_count`
  - `stability_fail_count`
  - `duplicate_fail_count`
  - `complexity_fail_count`
- 分布与原始包：
  - `success_pattern_dist_json`
  - `failure_pattern_dist_json`
  - `round_packet_json`
- 轮后 LLM 总结：
  - `round_reflection_json`

其中：

- `round_packet_json` 包含本轮全部因子卡片，每张卡片都带：
  - `factor_name`
  - `expression`
  - `explanation`
  - `references`
  - train/valid 指标
  - corr 指标
  - 复杂度指标
  - `final_status`
  - `failure_reasons`
  - `pattern`
- `round_reflection_json` 由 round analyst LLM 生成，包含：
  - `round_overview`
  - `success_patterns`
  - `failure_patterns`
  - `stability_observations`
  - `next_round_guidance`

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
│   ├── graph/
│   │   ├── success_mainline_graph.py
│   │   └── round_memory_progress_graph.py
│   ├── mining/orchestrator.py
│   ├── selection/
│   └── memory/
│       ├── llm_agents.py
│       ├── memory.py
│       ├── patterns.py
│       └── round_memory.py
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
│       ├── factor_failure_cases.csv
│       ├── round_memory.csv
│       ├── round_generation_summary.png
│       └── round_cumulative_success.png
├── factor_value_prepared/data/factors/
│   ├── dsl_factors_new.parquet
│   └── factor_ric.csv
└── tmp/
    ├── retrieval_planner_prompt.json
    └── round_analyst_prompt.json
```

覆盖策略（默认）：

- `tmp/` 下多数文件按轮覆盖。
- `scripts/LLM_factors/dsl_LLM_factors_new.parquet` 按轮覆盖。
- `data/factor_cache_new/LLM_library.yaml` 追加。
- `data/memory/factor_success_cases.csv` / `data/memory/factor_failure_cases.csv` 追加。
- `data/memory/round_memory.csv` 追加。
- `tmp/retrieval_planner_prompt.json` / `tmp/round_analyst_prompt.json` 每轮覆盖。

---

## 5. 主要日志解读

你会在终端看到这些关键行：

- `Runtime params`：全局资产、selection 实际资产、训练阈值。
- `Criteria enabled`：当前启用/关闭的筛选规则。
- `Corr scope source`：train/valid 相关性窗口取值来源。
- `Window policy`（orchestrator）：CSS/CoE 的窗口与资产策略。
- `Metrics evaluated`：本轮候选、通过、失败数量。
- `Memory updated`：成功/失败记忆条数与目标文件。
- `Memory retrieval ready`：轮前检索完成，打印参考因子数量与注入行数。
- `Round memory updated`：轮级账本已写入；若记忆模块关闭，会显示 `without reflection`。

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

### 6.5 `memory.enabled=false` 时还会不会生成 `round_memory.csv`

会。

- 仍然会生成/追加 `round_memory.csv`
- 仍然会写入 `round_packet_json`
- 不会调用 retrieval planner / round analyst
- `round_reflection_json` 会是空对象

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

### 7.3 绘制轮级进展图

```bash
python fama/graph/round_memory_progress_graph.py
```

默认读取 `data/memory/round_memory.csv`，生成两张图：

- `data/memory/round_generation_summary.png`
  - 柱状图：每条记录的 `generated_count` / `success_count`
  - 折线图：每条记录的 `pass_rate`
- `data/memory/round_cumulative_success.png`
  - 累计成功数量曲线

说明：

- 横坐标按 `round_memory.csv` 的行顺序编号，不使用 `round_id`，避免手动中断导致的跳号影响展示。
- 默认每 5 个记录显示一个横轴标签。

---

## 8. 开发约定

1. 路径、阈值、窗口、资产优先放 `defaults.yaml`。
2. `utils/` 放计算引擎，workflow 只做编排与落盘。
3. 因子长表统一列：`time/unique_id/factor_tag/value`。
4. 新功能优先保证：
   - 参数可控
   - 日志可观测
   - 模块可替换
