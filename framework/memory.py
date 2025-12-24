"""
Memory Implementation for Framework 2

全局记忆系统 - 智能体的长期记忆存储和检索
"""

import json
import logging
import pickle
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from rich.console import Console

logger = logging.getLogger(__name__)

# Console for colored output
console = Console()


def print_memory_header(title: str, color: str = "blue"):
    """Print memory-related header"""
    console.print(f"\n[bold {color}]{'=' * 60}[/bold {color}]")
    console.print(f"[bold {color}]{title}[/bold {color}]")
    console.print(f"[bold {color}]{'=' * 60}[/bold {color}]")


@dataclass
class MemoryEntry:
    """单条记忆记录"""
    timestamp: datetime
    node_id: str
    strategy_id: str
    plan: str
    code: Optional[str]
    reward: float
    summary: str  # 探索摘要
    metric: Optional[float] = None  # LLM 提取的 metric
    is_buggy: bool = False
    tags: List[str] = field(default_factory=list)
    parent_node_id: Optional[str] = None  # 父节点 ID
    children_node_ids: List[str] = field(default_factory=list)  # 子节点 ID 列表
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "strategy_id": self.strategy_id,
            "plan": self.plan[:300] + "..." if len(self.plan) > 300 else self.plan,
            "has_code": self.code is not None,
            "code_length": len(self.code) if self.code else 0,
            "reward": self.reward,
            "summary": self.summary,
            "metric": self.metric,
            "is_buggy": self.is_buggy,
            "tags": self.tags,
            "parent_node_id": self.parent_node_id,
            "children_count": len(self.children_node_ids),
        }

    def to_prompt_format(self, detailed: bool = False) -> str:
        """转换为适合 LLM 提示词的格式"""
        buggy_str = " [BUG]" if self.is_buggy else ""
        metric_str = f", metric={self.metric:.4f}" if self.metric is not None else ""

        # 根据 reward 显示状态
        if self.is_buggy:
            status = "BUG"
        elif self.reward > 0:
            status = "SUCCESS"
        elif self.reward < 0:
            status = "FAILURE"
        else:
            status = "NEUTRAL"

        if detailed:
            return (
                f"[{status}]{buggy_str} Node {self.node_id[:8]} "
                f"(reward={self.reward:.2f}{metric_str}):\n"
                f"  Plan: {self.plan[:200]}...\n"
                f"  Summary: {self.summary}\n"
                f"  Children: {len(self.children_node_ids)}\n"
            )
        else:
            return (
                f"[{status}]{buggy_str} Node {self.node_id[:8]} "
                f"(reward={self.reward:.2f}{metric_str}):\n"
                f"  Summary: {self.summary[:150]}...\n"
            )


class Memory:
    """
    Memory - 全局记忆系统

    功能：
    1. 记录所有节点的探索历史（包含 summary）
    2. 支持按奖励值、时间等多维度检索
    3. 突出显示当前节点的邻居和祖先节点
    4. 为智能体提供上下文感知的记忆检索
    5. 支持持久化存储和加载
    """

    def __init__(
        self,
        capacity: int = 5000,
        save_path: Optional[str] = None
    ):
        self.capacity = capacity
        self.save_path = save_path

        # 核心存储
        self.entries: List[MemoryEntry] = []  # 所有记忆条目
        self.recent_entries: deque = deque(maxlen=500)  # 最近记忆（快速访问）

        # 索引结构
        self.entries_by_node: Dict[str, List[MemoryEntry]] = {}
        self.entries_by_tag: Dict[str, List[MemoryEntry]] = {}

        # 节点关系索引
        self.node_to_parent: Dict[str, Optional[str]] = {}  # 节点 -> 父节点
        self.node_to_children: Dict[str, Set[str]] = {}  # 节点 -> 子节点集合

        # 最佳/最差记录
        self.best_entries: List[Tuple[float, MemoryEntry]] = []
        self.worst_entries: List[Tuple[float, MemoryEntry]] = []

        # 统计信息
        self.total_entries = 0
        self.best_reward: float = float('-inf')
        self.worst_reward: float = float('inf')
        self.bug_count = 0
        self.success_count = 0

        # 线程安全
        self.lock = threading.RLock()

        # 打印初始化信息
        console.print(f"[bold blue]Memory System Initialized[/bold blue]")
        console.print(f"[blue]  Capacity: {capacity} entries[/blue]")
        if save_path:
            console.print(f"[blue]  Save path: {save_path}[/blue]")

        # 加载已有记忆
        if save_path:
            self._load()

    def add_entry(
        self,
        node_id: str,
        strategy_id: str,
        plan: str,
        code: Optional[str],
        reward: float,
        summary: str,
        metric: Optional[float] = None,
        is_buggy: bool = False,
        tags: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
        children_node_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加记忆条目

        Args:
            node_id: 节点 ID
            strategy_id: 策略 ID
            plan: 策略计划
            code: 执行代码
            reward: 奖励值
            summary: 探索摘要
            metric: LLM 提取的 metric
            is_buggy: 是否有 bug
            tags: 标签列表
            parent_node_id: 父节点 ID
            children_node_ids: 子节点 ID 列表
            metadata: 额外元数据
        """
        with self.lock:
            entry = MemoryEntry(
                timestamp=datetime.now(),
                node_id=node_id,
                strategy_id=strategy_id,
                plan=plan,
                code=code,
                reward=reward,
                summary=summary,
                metric=metric,
                is_buggy=is_buggy,
                tags=tags or [],
                parent_node_id=parent_node_id,
                children_node_ids=children_node_ids or [],
                metadata=metadata or {}
            )

            # 添加到主存储
            self.entries.append(entry)
            self.recent_entries.append(entry)
            self.total_entries += 1

            # 更新统计
            if is_buggy:
                self.bug_count += 1
            elif reward > 0:
                self.success_count += 1

            # 更新节点关系索引
            self.node_to_parent[node_id] = parent_node_id
            if node_id not in self.node_to_children:
                self.node_to_children[node_id] = set()
            if children_node_ids:
                for child_id in children_node_ids:
                    self.node_to_children.setdefault(child_id, set()).add(node_id)

            # 更新索引
            if node_id not in self.entries_by_node:
                self.entries_by_node[node_id] = []
            self.entries_by_node[node_id].append(entry)

            for tag in entry.tags:
                if tag not in self.entries_by_tag:
                    self.entries_by_tag[tag] = []
                self.entries_by_tag[tag].append(entry)

            # 更新最佳/最差列表
            self.best_entries.append((reward, entry))
            self.best_entries.sort(key=lambda x: x[0], reverse=True)
            self.best_entries = self.best_entries[:100]

            self.worst_entries.append((reward, entry))
            self.worst_entries.sort(key=lambda x: x[0])
            self.worst_entries = self.worst_entries[:100]

            # 更新统计
            if reward > self.best_reward:
                self.best_reward = reward
                logger.info(f"New best reward: {reward:.4f}")
            if reward < self.worst_reward:
                self.worst_reward = reward

            # 容量管理
            if len(self.entries) > self.capacity:
                self._prune_entries()

            # 打印添加日志（带颜色）
            status_color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
            status_symbol = "✓" if not is_buggy and reward > 0 else "✗" if is_buggy else "○"
            console.print(f"[{status_color}]  {status_symbol} Memory: Node {node_id[:8]} | Reward: {reward:.3f} | Bug: {is_buggy}[/{status_color}]")

            logger.debug(f"Added memory entry for node {node_id[:8]}, reward: {reward:.4f}")

    def fetch_context_for_expansion(
        self,
        current_node: Optional[str] = None,
    ) -> str:
        """
        为智能体扩展节点时获取上下文记忆

        Args:
            current_node: 当前节点 ID（可选，用于突出邻居和祖先）
            max_entries: 最大返回条目数

        Returns:
            格式化的记忆上下文字符串
        """
        with self.lock:
            context_parts = []

            # 获取当前节点的祖先节点
            ancestor_ids = set()
            if current_node:
                ancestor_ids = self._get_ancestor_ids(current_node)

            # === 第一部分：当前节点的邻居和祖先（突出显示）===
            if current_node and (ancestor_ids or self.node_to_parent.get(current_node)):
                context_parts.append("=" * 60)
                context_parts.append("=== CURRENT NODE CONTEXT (Neighbors & Ancestors) ===")
                context_parts.append("=" * 60)

                # 祖先节点历史
                if ancestor_ids:
                    context_parts.append("\n--- Ancestors (Path from Root) ---")
                    for ancestor_id in reversed(list(ancestor_ids)):
                        ancestor_entries = self.entries_by_node.get(ancestor_id, [])
                        if ancestor_entries:
                            entry = ancestor_entries[-1]  # 最新的条目
                            context_parts.append(f"→ {entry.to_prompt_format(detailed=True)}")

                # 父节点
                parent_id = self.node_to_parent.get(current_node)
                if parent_id:
                    parent_entries = self.entries_by_node.get(parent_id, [])
                    if parent_entries:
                        context_parts.append(f"\n--- Parent Node {parent_id[:8]} ---")
                        for entry in parent_entries[-3:]:  # 最近3条
                            context_parts.append(f"  {entry.to_prompt_format()}")

                # 兄弟节点（同一父节点的其他子节点）
                siblings = self._get_sibling_ids(current_node)
                if siblings:
                    context_parts.append(f"\n--- Sibling Nodes ({len(siblings)}) ---")
                    for sibling_id in list(siblings)[:5]:  # 最多显示5个
                        sibling_entries = self.entries_by_node.get(sibling_id, [])
                        if sibling_entries:
                            entry = sibling_entries[-1]
                            context_parts.append(f"  • {entry.to_prompt_format()}")

                # 子节点
                children = self.node_to_children.get(current_node, set())
                if children:
                    context_parts.append(f"\n--- Children Nodes ({len(children)}) ---")
                    for child_id in list(children)[:5]:  # 最多显示5个
                        child_entries = self.entries_by_node.get(child_id, [])
                        if child_entries:
                            entry = child_entries[-1]
                            context_parts.append(f"  • {entry.to_prompt_format()}")

                context_parts.append("=" * 60)

            # === 第二部分：全局最佳策略 ===
            context_parts.append("\n=== Best Successful Strategies (Global) ===")
            for i, (reward, entry) in enumerate(self.best_entries[:10]):
                if not entry.is_buggy and reward > 0:
                    context_parts.append(f"{i+1}. {entry.to_prompt_format()}")

            # === 第三部分：最近的探索 ===
            if self.recent_entries:
                recent = list(self.recent_entries)[-20:]
                context_parts.append("\n=== Recent Explorations (Global) ===")
                for entry in recent:
                    context_parts.append(entry.to_prompt_format())

            # === 第四部分：失败的探索（避免重复）===
            worst_entries = [e for r, e in self.worst_entries if r < 0]
            if worst_entries:
                context_parts.append("\n=== Lessons from Failures ===")
                for entry in worst_entries[:10]:
                    context_parts.append(f"- {entry.summary[:200]}... (reward: {entry.reward:.2f})")

            # === 第五部分：有 Bug 的探索 ===
            buggy_entries = [e for e in self.entries if e.is_buggy]
            if buggy_entries:
                context_parts.append("\n=== Common Bugs to Avoid ===")
                for entry in buggy_entries[:10]:
                    context_parts.append(f"- {entry.summary[:200]}...")

            # === 第六部分：统计信息 ===
            context_parts.append(f"\n=== Statistics ===")
            context_parts.append(f"Total explorations: {self.total_entries}")
            context_parts.append(f"Success rate: {self.success_count / max(self.total_entries, 1) * 100:.1f}%")
            context_parts.append(f"Bug rate: {self.bug_count / max(self.total_entries, 1) * 100:.1f}%")
            context_parts.append(f"Best reward: {self.best_reward:.2f}")

            return "\n".join(context_parts) if context_parts else "No memory available yet."

    def _get_ancestor_ids(self, node_id: str) -> Set[str]:
        """获取节点的所有祖先节点 ID"""
        ancestors = set()
        current = node_id
        while current in self.node_to_parent:
            parent = self.node_to_parent[current]
            if parent is None:
                break
            ancestors.add(parent)
            current = parent
        return ancestors

    def _get_sibling_ids(self, node_id: str) -> Set[str]:
        """获取兄弟节点 ID"""
        parent = self.node_to_parent.get(node_id)
        if not parent:
            return set()

        siblings = set()
        for other_id, other_parent in self.node_to_parent.items():
            if other_parent == parent and other_id != node_id:
                siblings.add(other_id)
        return siblings

    def get_best_strategies(
        self,
        count: int = 10,
        min_reward: Optional[float] = None
    ) -> List[MemoryEntry]:
        """
        获取最佳策略

        Args:
            count: 返回数量
            min_reward: 最低奖励阈值

        Returns:
            最佳策略列表
        """
        with self.lock:
            entries = [entry for reward, entry in self.best_entries]

            if min_reward is not None:
                entries = [e for e in entries if e.reward >= min_reward]

            return entries[:count]

    def get_node_history(
        self,
        node_id: str,
        include_ancestors: bool = True
    ) -> List[MemoryEntry]:
        """
        获取节点的历史记录

        Args:
            node_id: 节点 ID
            include_ancestors: 是否包含祖先节点的历史

        Returns:
            历史记录列表
        """
        with self.lock:
            entries = self.entries_by_node.get(node_id, []).copy()
            return entries

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "total_entries": len(self.entries),
                "total_executions": self.total_entries,
                "best_reward": self.best_reward,
                "worst_reward": self.worst_reward,
                "average_reward": sum(e.reward for e in self.entries) / len(self.entries) if self.entries else 0.0,
                "unique_nodes": len(self.entries_by_node),
                "success_count": self.success_count,
                "bug_count": self.bug_count,
            }

    def _prune_entries(self):
        """清理条目，保留高价值的记录"""
        # 按奖励排序（成功 > 中性 > 失败 > bug）
        def score_entry(entry: MemoryEntry) -> float:
            if entry.is_buggy:
                return entry.reward - 2.0  # bug 惩罚
            return entry.reward

        self.entries.sort(key=score_entry, reverse=True)
        self.entries = self.entries[:self.capacity]

        # 重建索引
        self.entries_by_node.clear()
        self.entries_by_tag.clear()

        for entry in self.entries:
            if entry.node_id not in self.entries_by_node:
                self.entries_by_node[entry.node_id] = []
            self.entries_by_node[entry.node_id].append(entry)

            for tag in entry.tags:
                if tag not in self.entries_by_tag:
                    self.entries_by_tag[tag] = []
                self.entries_by_tag[tag].append(entry)

    def save(self):
        """保存到磁盘"""
        if not self.save_path:
            return

        with self.lock:
            try:
                data = {
                    "entries": [e.to_dict() for e in self.entries],
                    "statistics": {
                        "total_entries": self.total_entries,
                        "best_reward": self.best_reward,
                        "worst_reward": self.worst_reward,
                        "success_count": self.success_count,
                        "bug_count": self.bug_count
                    }
                }

                with open(self.save_path, 'wb') as f:
                    pickle.dump(data, f)

                logger.info(f"Memory saved to {self.save_path}")

            except Exception as e:
                logger.error(f"Failed to save Memory: {e}")

    def _load(self):
        """从磁盘加载"""
        if not self.save_path:
            return

        with self.lock:
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)

                # 重建条目
                self.entries = []
                for entry_data in data.get("entries", []):
                    entry = MemoryEntry(
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        node_id=entry_data["node_id"],
                        strategy_id=entry_data["strategy_id"],
                        plan=entry_data["plan"],
                        code=None,  # code 不保存，节省空间
                        reward=entry_data["reward"],
                        summary=entry_data["summary"],
                        metric=entry_data.get("metric"),
                        is_buggy=entry_data.get("is_buggy", False),
                        tags=entry_data.get("tags", []),
                        parent_node_id=entry_data.get("parent_node_id"),
                        children_node_ids=[],
                        metadata=entry_data.get("metadata", {})
                    )
                    self.entries.append(entry)

                # 重建索引
                self.entries_by_node.clear()
                self.entries_by_tag.clear()
                self.node_to_parent.clear()
                self.node_to_children.clear()

                for entry in self.entries:
                    if entry.node_id not in self.entries_by_node:
                        self.entries_by_node[entry.node_id] = []
                    self.entries_by_node[entry.node_id].append(entry)

                    if entry.parent_node_id:
                        self.node_to_parent[entry.node_id] = entry.parent_node_id

                    for tag in entry.tags:
                        if tag not in self.entries_by_tag:
                            self.entries_by_tag[tag] = []
                        self.entries_by_tag[tag].append(entry)

                # 重建最佳/最差列表
                self.best_entries = [(e.reward, e) for e in self.entries if e.reward > 0]
                self.best_entries.sort(key=lambda x: x[0], reverse=True)
                self.best_entries = self.best_entries[:100]

                self.worst_entries = [(e.reward, e) for e in self.entries if e.reward < 0]
                self.worst_entries.sort(key=lambda x: x[0])
                self.worst_entries = self.worst_entries[:100]

                # 加载统计信息
                stats = data.get("statistics", {})
                self.total_entries = stats.get("total_entries", 0)
                self.best_reward = stats.get("best_reward", float('-inf'))
                self.worst_reward = stats.get("worst_reward", float('inf'))
                self.success_count = stats.get("success_count", 0)
                self.bug_count = stats.get("bug_count", 0)

                logger.info(f"Memory loaded from {self.save_path} ({len(self.entries)} entries)")

            except FileNotFoundError:
                logger.info(f"No existing Memory found at {self.save_path}, starting fresh")
            except Exception as e:
                logger.error(f"Failed to load Memory: {e}")
