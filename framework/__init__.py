"""
Framework 2 - E^3 ML-Master

一个**长程且自演化**的智能体结构，通过简洁的智能体设计和工作流来实现长程探索任务上的提升。

核心分工：Envisioner & Executor

- Envisioner: 一个全局的 Agent，代表全局探索的统领者和决策者（一个强推理模型）
  - 维护一个全局的蒙特卡洛树搜索作为全局的树探索结构
  - 维护一个全局更新的 Memory 保证智能体的策略始终不发生偏移

- Executor: 多个并行执行的执行器（使用 Interpreter）
  - execute() 函数返回 Logger 信息、标量奖励和摘要

MCTS 探索循环：
1. Selection（选择）—— 使用 UCT 算法选择最具潜力的节点
2. Expansion & Task Dispatch（扩展与分发）—— Envisioner 生成 2-3 个新策略
3. Simulation / Evaluation（模拟与执行）—— Executor 并行执行并评测
4. Backpropagation（回溯更新）—— 全局同步更新搜索树
"""

from .node import MCTSNode, Strategy
from .memory import Memory, MemoryEntry
from .agent import Envisioner, Executor, ExecutorTask
from .utils import (
    create_pattern_extractor,
    extract_code_from_response,
    extract_summary_from_response,
    parse_strategy_response,
    calculate_uct,
    format_plan_prompt,
)

__all__ = [
    # Node
    "MCTSNode",
    "Strategy",
    # Memory
    "Memory",
    "MemoryEntry",
    # Agent
    "Envisioner",
    "Executor",
    "ExecutorTask",
    # Utils
    "create_pattern_extractor",
    "extract_code_from_response",
    "extract_summary_from_response",
    "parse_strategy_response",
    "calculate_uct",
    "format_plan_prompt",
]

__version__ = "0.1.0"
