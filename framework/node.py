"""
Core Node Implementation for Framework 2

节点设计：
每一个节点代表一个策略，主要包含下面的字段：
- code: 最终智能体提交的代码
- visits: 节点被访问的总次数
- expansion: 节点扩展的总次数
- total_rewards: 节点的奖励
- improve_failure_depth: 从这个节点向上出发不断回溯，判断持续分数下降的次数
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


def trim_long_string(string, threshold=5100, k=2500):
    """截断过长的字符串"""
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


@dataclass
class Strategy:
    """Represents a strategy with plan and code."""
    id: str
    plan: str
    code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "plan": self.plan,
            "code": self.code,
            "metadata": self.metadata
        }


class MCTSNode:
    """
    MCTS 搜索树节点实现

    节点字段：
    - code: 最终智能体提交的代码
    - visits: 节点被访问的总次数
    - expansion: 节点扩展的总次数
    - total_rewards: 节点的累积奖励
    - improve_failure_depth: 从这个节点向上出发不断回溯，判断持续分数下降的次数
    """

    def __init__(
        self,
        strategy: Strategy,
        parent: Optional['MCTSNode'] = None,
        max_expansions: int = 5
    ):
        self.id = uuid.uuid4().hex
        self.strategy = strategy
        self.parent = parent
        self.children: List['MCTSNode'] = []

        self.visits: int = 0          # 节点被访问的总次数
        self.expansion: int = 0       # 节点扩展的总次数
        self.total_rewards: float = 0.0  # 节点的累积奖励
        self.improve_failure_depth: int = 0  # 持续分数下降的次数

        # 最大扩展次数限制
        self.max_expansions: int = max_expansions

        # 预期子节点数量（用于并行环境下的扩展控制）
        self.expected_child_count: int = 0

        # 计算字段
        self.value: float = 0.0  # 平均奖励值

        # 状态标志
        self.is_terminal: bool = False
        self.execution_result: Optional[Any] = None

        # 改进追踪
        self.last_reward: float = 0.0

        # 代码执行结果字段（与 Interpreter 兼容）
        self._term_out: List[str] = []  # 终端输出
        self.exec_time: float = 0.0  # 执行时间
        self.exc_type: Optional[str] = None  # 异常类型
        self.exc_info: Optional[Dict] = None  # 异常信息
        self.exc_stack: Optional[List] = None  # 异常栈

        self.creation_time = time.time()

    @property
    def code(self) -> Optional[str]:
        """获取节点的代码"""
        return self.strategy.code

    @property
    def term_out(self) -> str:
        """获取终端输出（截断后）"""
        return trim_long_string("".join(self._term_out))

    def absorb_exec_result(self, exec_result):
        """
        吸收代码执行结果（与 Interpreter 的 ExecutionResult 兼容）

        Args:
            exec_result: Interpreter 返回的 ExecutionResult 对象
        """
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def add_child(self, child: 'MCTSNode'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)

    def get_path_to_root(self) -> List['MCTSNode']:
        """获取从当前节点到根节点的路径"""
        path = [self]
        current = self
        while current.parent:
            path.append(current.parent)
            current = current.parent
        return path

    def is_fully_expanded_with_expected(self) -> bool:
        """
        判断节点是否已完全扩展

        条件：扩展次数 >= max_expansions（最大扩展次数限制）

        根据 README.md 设计：
        - 获得当前节点的扩展次数（初始时为 0）
        - 如果扩展次数 < 最大扩展次数，则进行扩展
        """
        return self.expansion >= self.max_expansions

    def can_expand(self) -> bool:
        """
        判断是否可以继续扩展

        Returns:
            是否可以扩展（未达到最大扩展次数且非终端节点）
        """
        if self.is_terminal:
            return False

        return not self.is_fully_expanded_with_expected()

    def get_uct_value(self, exploration_constant: float = 1.414) -> float:
        """
        计算 UCT 值用于节点选择

        UCT = value + C * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: 探索常数

        Returns:
            UCT 值
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.value

        exploitation = self.value
        exploration = exploration_constant * (self.parent.visits / self.visits) ** 0.5

        return exploitation + exploration

    def update(self, reward: float, improve_threshold: float = 0.001):
        """
        更新节点统计信息

        Args:
            reward: 本次奖励
            improve_threshold: 改进阈值，用于判断 improve_failure_depth
        """
        self.visits += 1
        self.total_rewards += reward
        self.value = self.total_rewards / self.visits

        # 更新 improve_failure_depth
        # 如果改进幅度 < 阈值，则增加失败深度
        if self.visits > 1:
            improvement = reward - self.last_reward
            if improvement < improve_threshold:
                self.improve_failure_depth += 1
            else:
                # 改进成功，重置失败深度
                self.improve_failure_depth = 0

        self.last_reward = reward

    def increment_expansion(self):
        """增加扩展次数"""
        self.expansion += 1

    def set_expected_children(self, count: int):
        """设置预期的子节点数量（用于并行环境）"""
        self.expected_child_count = count

    def select_best_child(self, exploration_constant: float = 1.414) -> Optional['MCTSNode']:
        """
        使用 UCT 选择最佳子节点

        Args:
            exploration_constant: 探索常数

        Returns:
            最佳子节点
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.get_uct_value(exploration_constant))

    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典"""
        return {
            "id": self.id,
            "strategy": self.strategy.to_dict(),
            "visits": self.visits,
            "expansion": self.expansion,
            "max_expansions": self.max_expansions,
            "total_rewards": self.total_rewards,
            "value": self.value,
            "improve_failure_depth": self.improve_failure_depth,
            "expected_child_count": self.expected_child_count,
            "is_terminal": self.is_terminal,
            "children_count": len(self.children),
            "creation_time": self.creation_time
        }
