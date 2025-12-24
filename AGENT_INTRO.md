# E³ ML-Master 智能体框架详解

## 一、项目概述

E³ ML-Master 是一个**长程且自演化**的智能体结构，通过蒙特卡洛树搜索（MCTS）和 LLM 驱动的代码生成，来解决机器学习竞赛任务。该框架采用简洁的双核心设计：**Envisioner（全局探索决策者）** 和 **Executor（并行执行器）**。

### 核心特点

- **长程探索能力**：通过 MCTS 树结构维护完整的探索历史
- **自演化机制**：基于全局最佳节点和记忆系统进行策略迭代
- **并行执行**：多个 Executor 同时执行不同策略，提高效率
- **记忆驱动**：全局 Memory 系统存储和检索所有探索经验
- **多轮微调**：Executor 支持 Multi-turn Refinement 进行代码优化

---

## 二、核心架构

### 2.1 Envisioner - 全局探索决策者

Envisioner 是整个系统的核心，负责：
- 维护全局 MCTS 搜索树
- 协调多个 Executor 并行执行
- 管理全局 Memory 系统
- 追踪全局最佳节点

### 2.2 Executor - 代码执行与优化

Executor 负责具体节点的代码执行和优化：
- 执行节点策略代码
- 使用 LLM 从执行输出中提取 metric
- 支持多轮微调（Multi-turn Refinement）
- 返回标量 reward 用于 MCTS 回溯

### 2.3 Memory - 全局记忆系统

Memory 提供上下文感知的记忆检索：
- 存储所有节点的探索历史
- 支持按节点关系、奖励值等多维度检索
- 为策略生成提供上下文

---

## 三、MCTS 算法详解

### 3.1 节点设计（MCTSNode）

每个节点代表一个策略，包含以下关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 节点唯一标识符（UUID） |
| `strategy` | Strategy | 策略对象（包含 plan 和 code） |
| `visits` | int | 节点被访问的总次数 |
| `expansion` | int | 节点扩展的总次数 |
| `max_expansions` | int | 最大扩展次数限制（默认 5） |
| `total_rewards` | float | 累积奖励值 |
| `value` | float | 平均奖励值（total_rewards / visits） |
| `improve_failure_depth` | int | 持续分数下降的次数 |
| `expected_child_count` | int | 预期子节点数量（用于并行控制） |
| `parent` | MCTSNode | 父节点引用 |
| `children` | List[MCTSNode] | 子节点列表 |

#### 关键方法

```python
# 判断节点是否已完全扩展
def is_fully_expanded_with_expected(self) -> bool:
    return self.expansion >= self.max_expansions

# 计算 UCT 值用于节点选择
def get_uct_value(self, exploration_constant: float = 1.414) -> float:
    if self.visits == 0:
        return float('inf')
    exploitation = self.value
    exploration = exploration_constant * (self.parent.visits / self.visits) ** 0.5
    return exploitation + exploration

# 更新节点统计信息
def update(self, reward: float, improve_threshold: float = 0.001):
    self.visits += 1
    self.total_rewards += reward
    self.value = self.total_rewards / self.visits
```

### 3.2 MCTS 四步循环

#### 第一步：Selection（选择）

**目的**：寻找最具潜力的"生长点"

**流程**：

```
1. 从 Root 节点开始
2. 判断当前节点状态：
   ├─ 如果 expansion < max_expansions → 停在该节点（准备扩展）
   └─ 如果 expansion >= max_expansions → 计算 UCT，选择最高值子节点
3. 递归向下，直到找到需要扩展的节点
```

**核心代码**（agent.py:1119-1159）：

```python
def selection(self, node: Optional[MCTSNode] = None) -> MCTSNode:
    with self.tree_lock:
        if node is None:
            node = self.root_node

        current = node

        # 向下遍历，直到找到未完全扩展的节点
        while current.is_fully_expanded_with_expected() and current.children:
            # 选择 UCT 值最高的子节点
            current = current.select_best_child(self.exploration_constant)
            if current is None:
                break

        return current
```

**选择公式**（UCT）：
```
UCT(node) = value + C × sqrt(parent_visits / visits)
```

其中：
- `value`：节点的平均奖励（利用项）
- `C`：探索常数（默认 1.414）
- `parent_visits`：父节点的访问次数
- `visits`：当前节点的访问次数

---

#### 第二步：Expansion & Task Dispatch（扩展与分发）

**目的**：基于 Memory 生成新的策略变体

**流程**：

```
1. 并发控制检查
   ├─ 检查节点是否已完全扩展
   └─ 检查是否有待处理的扩展（expected_child_count > len(children)）

2. 锁定资源
   └─ 提前设置 expected_child_count，防止并发重复扩展

3. 获取上下文
   └─ 调用 fetch_memory 获取相关历史记录

4. 生成新策略
   ├─ 使用 LLM（deepseek-reasoner）生成 2~3 个新策略
   ├─ 如果解析失败，使用 deepseek-chat 作为提取器重新格式化
   └─ 策略格式：<strategy><plan_content>...</plan_content><reasoning>...</reasoning></strategy>

5. 创建节点
   ├─ 为每个策略创建 MCTSNode
   ├─ 添加到父节点的 children 列表
   └─ 更新父节点的 expansion 计数
```

**核心代码**（agent.py:1161-1252）：

```python
def expansion_and_dispatch(self, node: MCTSNode, num_strategies: int = 3) -> List[MCTSNode]:
    # 1. 并发控制检查
    with self.tree_lock:
        if node.is_fully_expanded_with_expected():
            return []
        if node.expected_child_count > len(node.children):
            return []
        # 提前设置预期数量，锁定节点
        node.set_expected_children(len(node.children) + num_strategies)

    # 2. 准备上下文（在锁外执行）
    memory_context = self._fetch_memory(node)

    # 3. 生成新策略
    new_strategies = self._generate_strategies(
        task_description=self.task_description,
        memory_context=memory_context,
        num_strategies=num_strategies,
    )

    # 4. 创建新节点
    child_nodes = []
    with self.tree_lock:
        for strategy_plan in new_strategies:
            strategy = Strategy(
                id=uuid.uuid4().hex,
                plan=strategy_plan.get("plan_content", ""),
                metadata={
                    "parent_id": node.id,
                    "reasoning": strategy_plan.get("reasoning", ""),
                },
            )
            child_node = MCTSNode(strategy, parent=node)
            node.add_child(child_node)
            child_nodes.append(child_node)

        node.increment_expansion()

    return child_nodes
```

**策略生成提示词**：

```
Task: {task_description}

You are using Monte Carlo Tree Search (MCTS) as your exploration structure.

Context from Memory:
{memory_context}

Please provide 2-3 different strategies to solve this task.

**IMPORTANT OUTPUT FORMAT:**
<strategy>
<plan_content>
[详细的行动计划：模型架构、特征工程、预处理、验证策略等]
</plan_content>
<reasoning>
[为什么这个方法可能有效：优势、潜在风险等]
</reasoning>
</strategy>
```

---

#### 第三步：Simulation / Evaluation（模拟与执行）

**目的**：并行执行策略并获取评估结果

**流程**：

```
1. 线程池执行
   └─ 使用 ThreadPoolExecutor 并行执行多个节点

2. Executor 执行（详见下一节）
   ├─ 如果节点没有代码，先生成初始代码
   ├─ 执行代码并获取输出
   ├─ 使用 LLM 从输出中提取 metric
   ├─ 计算 reward
   └─ 可选：进行多轮微调

3. 收集结果
   └─ 返回节点 ID -> ExecutionResult 的映射
```

**核心代码**（agent.py:1254-1302）：

```python
def simulation(self, nodes: List[MCTSNode]) -> Dict[str, ExecutionResult]:
    results = {}

    with ThreadPoolExecutor(max_workers=self.max_executor_count) as executor:
        future_to_node = {
            executor.submit(self._execute_single_node, node): node
            for node in nodes
        }

        for future in future_to_node:
            node = future_to_node[future]
            try:
                result = future.result(timeout=1800)
                results[node.id] = result
            except Exception as e:
                results[node.id] = ExecutionResult(
                    success=False,
                    reward=-1.0,
                    summary=f"Execution failed: {str(e)}",
                )

    return results
```

---

#### 第四步：Backpropagation（回溯更新）

**目的**：将执行结果传播回整个搜索树

**流程**：

```
1. 更新全局最佳节点
   ├─ 比较 metric 与当前最佳
   ├─ 根据 lower_is_better 判断改进方向
   └─ 更新 best_metric 和 best_node

2. 单点更新
   └─ 更新当前节点的 visits 和 total_rewards

3. 路径更新
   └─ 沿着 parent 指针向上，更新所有祖先节点

4. 记录到 Memory
   └─ 添加探索历史到全局记忆系统
```

**核心代码**（agent.py:1304-1393）：

```python
def backpropagation(self, nodes: List[MCTSNode], results: Dict[str, ExecutionResult]):
    for node in nodes:
        if node.id not in results:
            continue

        result = results[node.id]
        reward = result.reward
        metric = result.metric

        # 1. 更新全局最佳节点
        if not result.is_buggy and metric is not None:
            with self.best_lock:
                if self.best_metric is None:
                    self.best_metric = metric
                    self.best_node = node
                else:
                    lower_is_better = result.lower_is_better or False
                    if lower_is_better:
                        if metric < self.best_metric:
                            self.best_metric = metric
                            self.best_node = node
                    else:
                        if metric > self.best_metric:
                            self.best_metric = metric
                            self.best_node = node

        # 2. 回溯更新路径
        current = node
        with self.tree_lock:
            while current is not None:
                current.update(reward)
                current = current.parent

        # 3. 记录到 Memory
        self.memory.add_entry(
            node_id=node.id,
            strategy_id=node.strategy.id,
            plan=node.strategy.plan,
            code=node.strategy.code,
            reward=reward,
            summary=result.summary,
            metric=metric,
            is_buggy=result.is_buggy,
            parent_node_id=node.parent.id if node.parent else None,
        )
```

---

### 3.3 MCTS 完整循环

**完整流程**（agent.py:1397-1432）：

```python
def mcts_step(self, budget: int = 10):
    for i in range(budget):
        # 1. Selection
        selected_node = self.selection()

        # 2. Expansion & Task Dispatch
        new_nodes = self.expansion_and_dispatch(selected_node, num_strategies=3)

        # 3. Simulation / Evaluation
        results = self.simulation(new_nodes)

        # 4. Backpropagation
        self.backpropagation(new_nodes, results)
```

---

## 四、Executor 运行细节

### 4.1 Executor 核心职责

Executor 负责执行节点的代码并返回评估结果：

```
输入：MCTSNode（包含策略 plan 和可选的 code）
输出：ExecutionResult（包含 reward、metric、summary 等）
```

### 4.2 Reward 计算机制

Reward 是一个标量值，用于 MCTS 的 UCT 计算：

```python
def _calculate_reward(self, is_buggy: bool, metric: Optional[float], lower_is_better: bool) -> float:
    # 有 bug 或无 metric：奖励 -1
    if is_buggy or metric is None:
        return -1.0

    # 无 bug 且有 metric：基础奖励 +1
    reward = 1.0

    # 如果比当前最佳更好，额外 +1
    if self.best_metric is not None:
        if lower_is_better:
            improvement = self.best_metric - metric
        else:
            improvement = metric - self.best_metric

        if improvement > 0:
            reward += 1

    return reward
```

**Reward 分级**：
- `-1.0`：有 bug 或无有效 metric
- `1.0`：成功执行，但未改进
- `2.0`：成功执行且改进了最佳结果

### 4.3 LLM Metric 提取

Executor 使用 LLM（Function Calling）从代码执行输出中提取结构化信息：

**Function Spec**（agent.py:53-95）：

```python
REVIEW_FUNC_SPEC = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if execution failed or has bugs"
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if submission.csv was saved in ./submission/"
            },
            "summary": {
                "type": "string",
                "description": "short summary (2-3 sentences) of findings"
            },
            "metric": {
                "type": "number",
                "description": "validation metric value, or null if failed"
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if lower metric is better (e.g., MSE)"
            },
        },
        "required": ["is_bug", "has_csv_submission", "summary", "metric", "lower_is_better"],
    },
)
```

**提取 Prompt**（agent.py:235-246）：

```python
prompt = {
    "Introduction": "You are a Kaggle grandmaster attending a competition...",
    "Task description": self.task_description,
    "Implementation": node.strategy.code,
    "Execution output": node.term_out,
}
```

### 4.4 Multi-turn Refinement（多轮微调）

Executor 支持在执行代码后进行多轮迭代优化：

#### 启动条件

```python
Executor(
    interpreter=interpreter,
    task_description=task_description,
    model_name="deepseek-chat",
    enable_refinement=True,      # 启用多轮微调
    max_refinement_turns=3,       # 最多 3 轮
)
```

#### 完整流程

```
1. 检查节点是否有代码
   └─ 如果没有，先生成初始代码

2. 执行初始代码
   └─ 调用 _execute_once 获取初始结果

3. 多轮微调循环（最多 max_refinement_turns 轮）
   ├─ 构建 refinement messages（包含任务、代码、执行结果）
   ├─ 调用 LLM，提供工具：
   │   ├─ run_python_code：执行代码并获取输出
   │   ├─ web_search：搜索 ML 最佳实践
   │   └─ web_parse：解析网页内容
   ├─ LLM 可以调用工具进行实验
   ├─ 每次执行后提取 metric，记录最佳结果
   └─ 如果 LLM 选择停止或达到最大轮数，退出循环

4. 生成最终代码
   ├─ 基于所有探索，让 LLM 生成最终最佳代码
   ├─ 执行验证
   └─ 生成总结

5. 返回最佳结果
```

#### Refinement Tools（agent.py:179-223）

```python
refinement_tools = [
    FunctionSpec(
        name="run_python_code",
        description="Execute Python code locally and return raw console output",
        json_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    ),
    FunctionSpec(
        name="web_search",
        description="Search web for ML best practices, libraries, or techniques",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    ),
    FunctionSpec(
        name="web_parse",
        description="Parse web URLs for detailed information",
        json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to parse"}
            },
            "required": ["url"],
        },
    ),
]
```

#### Refinement System Prompt（agent.py:939-976）

```
You are an expert machine learning engineer performing multi-turn code refinement.

**Task Description:** {task_description}
**Current Code:** {node.strategy.code}
**Initial Execution Result:**
- Metric: {initial_result.metric}
- Summary: {initial_result.summary}

Your goal is to improve the code performance through iterative refinement.

**Important Constraints:**
- Focus on hyperparameter tuning and minor code improvements
- Do NOT make major architectural changes
- Do NOT change data processing pipelines
- Keep the code executable and complete
- Each improvement should be incremental

Use tools iteratively to find the best configuration.
```

### 4.5 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     Executor.execute()                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 节点有代码吗？  │
                    └─────────────────┘
                         │          │
                        否          是
                         │          │
                         ▼          │
              ┌─────────────────┐  │
              │ 生成初始代码    │  │
              └─────────────────┘  │
                         │          │
                         └────┬─────┘
                              ▼
                    ┌─────────────────┐
                    │ _execute_once() │
                    │ (执行代码)      │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 启用 Refinement?│
                    └─────────────────┘
                         │          │
                        否          是
                         │          │
                         ▼          ▼
              ┌──────────┐   ┌──────────────────┐
              │ 返回结果  │   │_multi_turn_refine│
              └──────────┘   └──────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐  ┌─────────────┐  ┌─────────────┐
            │多轮工具调用│  │生成最终代码 │  │生成总结     │
            └───────────┘  └─────────────┘  └─────────────┘
                    │               │               │
                    └───────────────┴───────────────┘
                                    ▼
                              ┌──────────┐
                              │返回最佳结果│
                              └──────────┘
```

---

## 五、Memory 全局记忆系统

### 5.1 MemoryEntry 结构

```python
@dataclass
class MemoryEntry:
    timestamp: datetime              # 时间戳
    node_id: str                     # 节点 ID
    strategy_id: str                 # 策略 ID
    plan: str                        # 策略计划
    code: Optional[str]              # 执行代码
    reward: float                    # 奖励值
    summary: str                     # 探索摘要
    metric: Optional[float]          # LLM 提取的 metric
    is_buggy: bool                   # 是否有 bug
    tags: List[str]                  # 标签
    parent_node_id: Optional[str]    # 父节点 ID
    children_node_ids: List[str]     # 子节点 ID 列表
    metadata: Dict[str, Any]         # 额外元数据
```

### 5.2 记忆检索：fetch_context_for_expansion

为核心扩展操作提供上下文感知的记忆检索：

**检索策略**（memory.py:234-336）：

```
1. 当前节点的邻居和祖先（突出显示）
   ├─ 祖先节点历史（从根到当前节点的路径）
   ├─ 父节点最近 3 条记录
   ├─ 兄弟节点（最多 5 个）
   └─ 子节点（最多 5 个）

2. 全局最佳策略（Top 10）
   └─ 按奖励排序的成功策略

3. 最近的探索（最近 20 条）
   └─ 全局最近执行记录

4. 失败的探索
   └─ 避免 repeat 失败

5. 有 Bug 的探索
   └─ 常见错误预警

6. 统计信息
   ├─ 总探索次数
   ├─ 成功率
   ├─ Bug 率
   └─ 最佳奖励
```

**输出格式示例**：

```
============================================================
=== CURRENT NODE CONTEXT (Neighbors & Ancestors) ===
============================================================

--- Ancestors (Path from Root) ---
→ [SUCCESS] Node abc12345 (reward=2.00, metric=0.9234):
  Plan: Use XGBoost with feature engineering...
  Summary: Achieved good accuracy with gradient boosting...
  Children: 3

--- Sibling Nodes (2) ---
  • [FAILURE] Node def67890 (reward=-1.00):
  Summary: Failed due to data preprocessing error...

=== Best Successful Strategies (Global) ===
1. [SUCCESS] Node xyz98765 (reward=2.00, metric=0.9456):
  Summary: Best model with tuned hyperparameters...

=== Recent Explorations (Global) ===
[SUCCESS] Node lmn34567 (reward=1.00, metric=0.9123):
  Summary: Moderate performance with random forest...

=== Statistics ===
Total explorations: 156
Success rate: 68.5%
Bug rate: 15.2%
Best reward: 2.00
```

### 5.3 索引结构

Memory 维护多个索引以支持高效检索：

```python
# 核心存储
self.entries: List[MemoryEntry]              # 所有记忆条目
self.recent_entries: deque                   # 最近记忆（快速访问）

# 索引结构
self.entries_by_node: Dict[str, List[MemoryEntry]]    # 节点 -> 条目
self.entries_by_tag: Dict[str, List[MemoryEntry]]     # 标签 -> 条目

# 节点关系索引
self.node_to_parent: Dict[str, Optional[str]]          # 节点 -> 父节点
self.node_to_children: Dict[str, Set[str]]             # 节点 -> 子节点集合

# 最佳/最差记录
self.best_entries: List[Tuple[float, MemoryEntry]]     # Top 100 成功
self.worst_entries: List[Tuple[float, MemoryEntry]]    # Top 100 失败
```

---

## 六、完整数据流与执行流程

### 6.1 初始化流程

```
┌─────────────────────────────────────────────────────────────┐
│                       系统初始化                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 创建 Envisioner │
                    │ - 加载系统提示词 │
                    │ - 初始化 Memory │
                    │ - 创建 LLM 客户端│
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │initialize_root()│
                    │ - 让 LLM 生成   │
                    │   初始策略       │
                    │ - 创建根节点     │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  MCTS 搜索循环  │
                    │  (mcts_step)    │
                    └─────────────────┘
```

### 6.2 完整 MCTS 迭代流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCTS Iteration                              │
└─────────────────────────────────────────────────────────────────────┘

   Selection                Expansion                 Simulation
┌──────────────┐       ┌──────────────┐         ┌──────────────┐
│              │       │              │         │              │
│  Root Node   │──────▶│ Selected Node│───────▶│  New Nodes   │
│              │       │              │         │  (2-3 nodes) │
│              │       │ - Fetch      │         │              │
│              │       │   Memory     │         │ - Parallel   │
│              │       │ - Generate   │         │   Execution  │
│              │       │   Strategies │         │ - LLM Extract│
│              │       │ - Create     │         │   Metric     │
│              │       │   Children   │         │              │
└──────────────┘       └──────────────┘         └──────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ ExecutionResult  │
                                                │ - reward         │
                                                │ - metric         │
                                                │ - summary        │
                                                │ - is_buggy       │
                                                └──────────────────┘
                                                          │
                                                          ▼
                                      ┌──────────────────────────────────┐
                                      │       Backpropagation             │
                                      │                                  │
                                      │  ┌────────────────────────────┐  │
                                      │  │ Update Global Best         │  │
                                      │  └────────────────────────────┘  │
                                      │                │                 │
                                      │  ┌────────────────────────────┐  │
                                      │  │ Update Node Statistics     │  │
                                      │  │ - visits += 1              │  │
                                      │  │ - total_rewards += reward  │  │
                                      │  │ - value = avg              │  │
                                      │  └────────────────────────────┘  │
                                      │                │                 │
                                      │  ┌────────────────────────────┐  │
                                      │  │ Update All Ancestors       │  │
                                      │  │ (recursive via parent)     │  │
                                      │  └────────────────────────────┘  │
                                      │                │                 │
                                      │  ┌────────────────────────────┐  │
                                      │  │ Record to Memory           │  │
                                      │  └────────────────────────────┘  │
                                      └──────────────────────────────────┘
```

### 6.3 线程模型

```
┌─────────────────────────────────────────────────────────────┐
│                    主线程 (Envisioner)                       │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │Selection│───▶│Expansion│───▶│Simulation│───▶│Backprop │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       │                              │                      │
│       │                              ▼                      │
│                       ┌──────────────────────────┐          │
│                       │   ThreadPoolExecutor     │          │
│                       │   (max_workers = 3)      │          │
│                       └──────────────────────────┘          │
│                                    │                         │
│              ┌─────────────────────┼─────────────────────┐  │
│              │                     │                     │  │
│              ▼                     ▼                     ▼  │
│       ┌──────────┐          ┌──────────┐         ┌──────────┐│
│       │Executor 1│          │Executor 2│         │Executor 3││
│       │Thread    │          │Thread    │         │Thread    ││
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                    Executor 工作流                       │ │
│ │  1. 生成初始代码（如果需要）                             │ │
│ │  2. 执行代码（Interpreter）                              │ │
│ │  3. LLM 提取 metric                                      │ │
│ │  4. Multi-turn Refinement（可选）                       │ │
│ │     - 工具调用（run_python_code, web_search, web_parse）│ │
│ │     - 迭代优化                                           │ │
│ │     - 生成最终代码                                       │ │
│ │  5. 返回 ExecutionResult                                 │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、关键算法与公式

### 7.1 UCT（Upper Confidence Bound for Trees）

```
UCT(node) = value + C × sqrt(parent_visits / visits)
```

- `value`：节点的平均奖励（利用项，Exploitation）
- `C`：探索常数（默认 √2 ≈ 1.414）
- `parent_visits`：父节点的访问次数
- `visits`：当前节点的访问次数

### 7.2 Reward 计算

```python
# 基础奖励
reward = -1.0    # 有 bug 或无 metric
reward = 1.0     # 成功执行
reward = 2.0     # 成功且改进

# 改进判断
if lower_is_better:
    improvement = best_metric - metric  # 越小越好
else:
    improvement = metric - best_metric  # 越大越好

if improvement > 0:
    reward += 1  # 额外奖励
```

### 7.3 节点完全扩展判断

```python
is_fully_expanded = (expansion >= max_expansions)
```

- `expansion`：当前扩展次数
- `max_expansions`：最大扩展次数（默认 5）

### 7.4 并发控制

```python
# 提前锁定节点
if node.expected_child_count > len(node.children):
    return []  # 已有待处理的扩展

node.set_expected_children(len(node.children) + num_strategies)
```

---

## 八、配置与超参数

### 8.1 Envisioner 配置

```python
Envisioner(
    interpreter=interpreter,
    memory=memory,
    task_description=task_description,
    model_name="deepseek-reasoner",        # 策略生成模型
    feedback_model_name="deepseek-chat",   # metric 提取模型
    exploration_constant=1.414,            # UCT 探索常数
    max_executor_count=3,                  # 最大并行执行器数
    max_node_expansions=5,                 # 节点最大扩展次数
)
```

### 8.2 Executor 配置

```python
Executor(
    interpreter=interpreter,
    task_description=task_description,
    model_name="deepseek-chat",            # metric 提取模型
    best_metric=best_metric,
    best_node=best_node,
    enable_refinement=False,               # 是否启用多轮微调
    max_refinement_turns=3,                # 最大微调轮数
)
```

### 8.3 Memory 配置

```python
Memory(
    capacity=5000,                         # 最大记忆容量
    save_path="memory.pkl",                # 持久化路径
)
```

---

## 九、总结

### 9.1 核心设计思想

1. **分离关注点**：Envisioner 负责全局策略，Executor 负责局部执行
2. **记忆驱动**：Memory 系统提供上下文感知的探索历史
3. **并行加速**：多个 Executor 同时执行，提高探索效率
4. **自适应探索**：MCTS 的 UCT 算法平衡探索与利用
5. **多轮优化**：Refinement 机制允许代码迭代改进

### 9.2 与传统 MCTS 的区别

| 特性 | 传统 MCTS | E³ ML-Master |
|------|-----------|--------------|
| 状态空间 | 游戏状态（离散） | ML 策略空间（连续） |
| 动作 | 游戏动作 | LLM 生成的策略 |
| 模拟 | 随机模拟 | 实际代码执行 + LLM 评估 |
| 回溯 | 胜负结果 | 标量 reward（基于 metric） |
| 并行 | 通常串行 | 多 Executor 并行 |

### 9.3 适用场景

- 机器学习竞赛（Kaggle、MLE-Bench）
- 代码优化任务
- 超参数调优
- 特征工程探索

---

*文档版本：v1.0*
*最后更新：2024*
