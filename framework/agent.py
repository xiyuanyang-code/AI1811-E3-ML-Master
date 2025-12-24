import logging
import time
import uuid
import os
import json
import sys
import threading

sys.path.append(os.getcwd())


from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from dataclasses import dataclass, field
from openai import OpenAI
from rich.console import Console

from .node import MCTSNode, Strategy
from .memory import Memory
from .utils import (
    parse_strategy_response,
    format_plan_prompt,
    extract_code_from_response,
    web_parse,
    web_search,
)
from interpreter.interpreter_parallel import Interpreter, ExecutionResult
from backend import FunctionSpec, query

logger = logging.getLogger(__name__)

# Console for colored output
console = Console()


def print_llm_output(title: str, content: str, max_length: int = 1000):
    """Print LLM output in green color to console for monitoring"""
    if content:
        # Truncate if too long
        display_content = content if len(content) <= max_length else content[:max_length] + "\n... [truncated] ..."
        console.print(f"\n[bold green]{'=' * 60}[/bold green]")
        console.print(f"[bold green]{title}[/bold green]")
        console.print(f"[bold green]{'=' * 60}[/bold green]")
        console.print(f"[green]{display_content}[/green]")
        console.print(f"[bold green]{'=' * 60}[/bold green]\n")
    else:
        console.print(f"\n[bold yellow]LLM Output ({title}): [Empty or None][/bold yellow]\n")


# ==================== Function Spec for LLM ====================

REVIEW_FUNC_SPEC = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a CSV file named either `submission.csv` or matching the pattern `submission_<hash>.csv` in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


# ==================== Data Classes ====================


@dataclass
class ExecutionResult:
    """执行结果数据类"""

    success: bool
    reward: float  # 标量奖励（基于 LLM 提取的 metric）
    summary: str  # 执行摘要
    logger_info: List[str]  # Logger 信息
    metric: Optional[float] = None  # LLM 提取的实际评分
    is_buggy: bool = False  # 是否有 bug
    lower_is_better: Optional[bool] = None  # 指标是否越小越好
    output: Any = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorTask:
    """执行器任务"""

    node: MCTSNode
    callback: Optional[Callable[[MCTSNode, ExecutionResult], None]] = None
    priority: int = 0


# ==================== Executor ====================


class Executor:
    """
    执行器（Executor）

    负责执行具体节点的代码并返回结果。
    使用 Interpreter 进行代码执行，使用 LLM 从 term_out 中提取 metric 作为 reward。

    支持多轮微调（Multi-turn Refinement）功能，通过 Tool Calling 优化代码。
    """

    def __init__(
        self,
        interpreter: Interpreter,
        task_description: str,
        model_name: str = "deepseek-chat",
        best_metric: Optional[float] = None,
        best_node: Optional[MCTSNode] = None,
        executor_id: int = 0,
        enable_refinement: bool = False,
        max_refinement_turns: int = 3,
    ):
        """
        Args:
            interpreter: Interpreter 实例
            llm_client: LLM 客户端（OpenAI）
            task_description: 任务描述
            model_name: 用于提取 metric 的模型名称
            best_metric: 当前全局最佳 metric 值
            best_node: 当前全局最佳节点
            executor_id: 执行器 ID
            enable_refinement: 是否启用多轮微调
            max_refinement_turns: 最大微调轮数
        """
        self.interpreter = interpreter
        self.llm_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )
        self.task_description = task_description
        self.model_name = model_name
        self.best_metric = best_metric
        self.best_node = best_node
        self.executor_id = executor_id
        self.is_running = False
        self.enable_refinement = enable_refinement
        self.max_refinement_turns = max_refinement_turns
        self.web_search_fn = None  # 用户提供的 web_search 函数

        # Multi-turn refinement 工具定义
        self.refinement_tools = [
            FunctionSpec(
                name="run_python_code",
                description="Execute Python code locally and return the raw console output",
                json_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            ),
            FunctionSpec(
                name="web_search",
                description="Search the web for information about machine learning best practices, libraries, or techniques",
                json_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            FunctionSpec(
                name="web_parse",
                description="Search the web for parsing the relevant web urls for more **detailed informations**",
                json_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Search url"},
                    },
                    "required": ["query"],
                },
            ),
        ]

        # 工具函数映射
        self.tool_functions = {
            "run_python_code": self._tool_run_python_code,
            "web_search": web_search,
            "web_parse": web_parse,
        }

    def _parse_exec_result_with_llm(self, node: MCTSNode) -> Dict[str, Any]:
        """
        使用 LLM 从执行结果中提取 metric

        Args:
            node: 已执行的节点（包含 term_out）

        Returns:
            LLM 返回的解析结果（is_bug, has_csv_submission, summary, metric, lower_is_better）
        """
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )

        prompt = {
            "Introduction": introduction,
            "Task description": self.task_description,
            "Implementation": self._wrap_code(node.strategy.code or ""),
            "Execution output": self._wrap_code(node.term_out, lang=""),
        }

        try:
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=REVIEW_FUNC_SPEC,
                model=self.model_name,
                temperature=0.0,
            )

            # 打印 LLM 输出到控制台（绿色）
            response_str = json.dumps(response, indent=2, ensure_ascii=False)
            print_llm_output(f"LLM Execution Result Parsing (node {node.id[:8]})", response_str, max_length=1500)

            if not isinstance(response, dict):
                logger.error(f"LLM returned non-dict response: {type(response)}")
                return self._get_default_error_response()

            # 验证返回值
            if not isinstance(response.get("metric"), float):
                response["metric"] = None

            return response

        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return self._get_default_error_response()

    def _wrap_code(self, code: str, lang: str = "python") -> str:
        """包装代码为 markdown 格式"""
        return f"```{lang}\n{code}\n```"

    def _get_default_error_response(self) -> Dict[str, Any]:
        """返回默认错误响应"""
        return {
            "is_bug": True,
            "has_csv_submission": False,
            "summary": "Failed to parse execution result with LLM",
            "metric": None,
            "lower_is_better": True,
        }

    def _calculate_reward(
        self, is_buggy: bool, metric: Optional[float], lower_is_better: bool
    ) -> float:
        """
        基于 metric 计算 reward（仿照原代码逻辑）

        Args:
            is_buggy: 是否有 bug
            metric: 指标值
            lower_is_better: 指标是否越小越好

        Returns:
            reward 值
        """
        # 有 bug 或无 metric：奖励 -1
        if is_buggy or metric is None:
            return -1.0

        # 无 bug 且有 metric：基础奖励 +1
        reward = 1.0

        # 如果比当前最佳更好，额外 +1
        if self.best_metric is not None:
            # 根据 lower_is_better 判断改进方向
            if lower_is_better:
                # 越小越好：metric < best_metric 表示改进
                improvement = self.best_metric - metric
            else:
                # 越大越好：metric > best_metric 表示改进
                improvement = metric - self.best_metric

            if improvement > 0:
                logger.info(
                    f"Node metric {metric:.4f} is better than best {self.best_metric:.4f}! "
                    f"(improvement: {improvement:.4f})"
                )
                reward += 1

        return reward

    def execute(self, node: MCTSNode) -> ExecutionResult:
        """
        执行节点的代码并使用 LLM 提取 metric

        Args:
            node: 要执行的 MCTS 节点

        Returns:
            ExecutionResult 包含基于 LLM 提取 metric 的 reward
        """
        if not node.strategy.code:
            return ExecutionResult(
                success=False,
                reward=-1.0,
                summary="No code to execute",
                logger_info=["Error: Node has no code"],
                is_buggy=True,
            )

        logger.info(f"Executor {self.executor_id} executing node {node.id[:8]}")

        try:
            # 第一步：使用 Interpreter 执行代码
            exec_result = self.interpreter.run(
                code=node.strategy.code, id=node.id, reset_session=True
            )

            # 吸收执行结果到节点
            node.absorb_exec_result(exec_result)

            # 提取 logger_info
            logger_info = exec_result.term_out

            # 判断是否有异常
            has_exception = exec_result.exc_type is not None

            # 第二步：使用 LLM 从 term_out 中提取 metric
            llm_response = self._parse_exec_result_with_llm(node)

            is_buggy = (
                llm_response["is_bug"]
                or has_exception
                or llm_response["metric"] is None
                or not llm_response["has_csv_submission"]
            )

            metric_value = llm_response["metric"]
            lower_is_better = llm_response["lower_is_better"]
            summary = llm_response["summary"]

            # 第三步：基于 metric 计算 reward
            reward = self._calculate_reward(
                is_buggy=is_buggy, metric=metric_value, lower_is_better=lower_is_better
            )

            logger.info(
                f"Node {node.id[:8]}: is_buggy={is_buggy}, "
                f"metric={metric_value}, reward={reward:.2f}"
            )

            return ExecutionResult(
                success=not is_buggy,
                reward=reward,
                summary=summary,
                logger_info=logger_info,
                metric=metric_value,
                is_buggy=is_buggy,
                lower_is_better=lower_is_better,
                execution_time=exec_result.exec_time,
                error_message=exec_result.exc_type,
                metadata={"exc_info": exec_result.exc_info},
            )

        except Exception as e:
            logger.error(
                f"Executor {self.executor_id} failed for node {node.id[:8]}: {e}"
            )
            return ExecutionResult(
                success=False,
                reward=-1.0,
                summary=f"Execution error: {str(e)}",
                logger_info=[f"Error: {str(e)}"],
                is_buggy=True,
            )

    # ==================== Multi-turn Refinement ====================

    def execute(self, node: MCTSNode) -> ExecutionResult:
        """
        执行节点 - Multi-turn Refinement 过程

        完整流程：
        1. 如果节点没有 code，先让 LLM 生成初始代码
        2. 执行代码并获取结果
        3. 如果启用 refinement，进行多轮微调
        4. 返回最佳结果

        Args:
            node: 要执行的 MCTS 节点

        Returns:
            ExecutionResult 包含最佳执行结果
        """
        if not node.strategy.code:
            logger.info(f"Node {node.id[:8]} has no code, generating initial code...")
            initial_code = self._generate_initial_code(node)
            if not initial_code:
                return ExecutionResult(
                    success=False,
                    reward=-1.0,
                    summary="Failed to generate initial code",
                    logger_info=["Error: Failed to generate initial code"],
                    is_buggy=True,
                )
            node.strategy.code = initial_code

        logger.info(f"Executor {self.executor_id} executing node {node.id[:8]}")
        initial_result = self._execute_once(node)
        logger.info(f"Starting multi-turn refinement for node {node.id[:8]}")

        try:
            refined_result = self._multi_turn_refine(node, initial_result)
            return refined_result
        except Exception as e:
            logger.error(f"Refinement failed for node {node.id[:8]}: {e}")
            return initial_result

    def _generate_initial_code(self, node: MCTSNode) -> Optional[str]:
        """
        生成初始代码

        Args:
            node: 节点（包含 plan）

        Returns:
            生成的代码字符串
        """
        try:
            system_prompt = f"""You are an expert machine learning engineer.

**Task Description:**
{self.task_description}

**Strategy Plan:**
{node.strategy.plan}

Generate complete, executable Python code to solve this task. The code should:
1. Load and preprocess the data
2. Build and train a machine learning model
3. Evaluate on validation set
4. Save predictions to `./submission/submission.csv`

Return your code in a <code> tag."""

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please generate the initial code."},
                ],
                temperature=0.7,
                max_tokens=3000,
            )

            content = response.choices[0].message.content

            # 打印 LLM 输出到控制台（绿色）
            print_llm_output(f"LLM Initial Code Generation (node {node.id[:8]})", content, max_length=3000)

            # 提取代码

            code = extract_code_from_response(content)

            if code:
                logger.info(f"Generated initial code for node {node.id[:8]}")
                return code
            else:
                # 如果没有 <code> 标签，尝试提取 ```python 代码块
                import re

                pattern = r"```python\s*\n(.*?)\n```"
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    return matches[0]

                logger.warning(
                    f"Failed to extract code from LLM response for node {node.id[:8]}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to generate initial code: {e}")
            return None

    def _execute_once(self, node: MCTSNode) -> ExecutionResult:
        """
        执行一次代码

        Args:
            node: 要执行的节点

        Returns:
            执行结果
        """
        if not node.strategy.code:
            return ExecutionResult(
                success=False,
                reward=-1.0,
                summary="No code to execute",
                logger_info=["Error: Node has no code"],
                is_buggy=True,
            )

        try:
            # 使用 Interpreter 执行代码
            exec_result = self.interpreter.run(
                code=node.strategy.code, id=node.id, reset_session=True
            )

            # 吸收执行结果到节点
            node.absorb_exec_result(exec_result)

            # 提取 logger_info
            logger_info = exec_result.term_out

            # 判断是否有异常
            has_exception = exec_result.exc_type is not None

            # 使用 LLM 从 term_out 中提取 metric
            llm_response = self._parse_exec_result_with_llm(node)

            is_buggy = (
                llm_response["is_bug"]
                or has_exception
                or llm_response["metric"] is None
                or not llm_response["has_csv_submission"]
            )

            metric_value = llm_response["metric"]
            lower_is_better = llm_response["lower_is_better"]
            summary = llm_response["summary"]

            # 计算 reward
            reward = self._calculate_reward(
                is_buggy=is_buggy, metric=metric_value, lower_is_better=lower_is_better
            )

            logger.info(
                f"Node {node.id[:8]}: is_buggy={is_buggy}, "
                f"metric={metric_value}, reward={reward:.2f}"
            )

            return ExecutionResult(
                success=not is_buggy,
                reward=reward,
                summary=summary,
                logger_info=logger_info,
                metric=metric_value,
                is_buggy=is_buggy,
                lower_is_better=lower_is_better,
                execution_time=exec_result.exec_time,
                error_message=exec_result.exc_type,
                metadata={"exc_info": exec_result.exc_info},
            )

        except Exception as e:
            logger.error(
                f"Executor {self.executor_id} failed for node {node.id[:8]}: {e}"
            )
            return ExecutionResult(
                success=False,
                reward=-1.0,
                summary=f"Execution error: {str(e)}",
                logger_info=[f"Error: {str(e)}"],
                is_buggy=True,
            )

    def _multi_turn_refine(
        self, node: MCTSNode, initial_result: ExecutionResult
    ) -> ExecutionResult:
        """
        多轮微调实现

        Args:
            node: 当前节点
            initial_result: 初始执行结果

        Returns:
            微调后的执行结果
        """
        current_code = node.strategy.code
        best_metric = initial_result.metric
        best_code = current_code
        best_result = initial_result

        # 构建初始消息
        messages = self._build_refinement_messages(node, initial_result)

        # 多轮对话
        for turn in range(self.max_refinement_turns):
            logger.info(f"Refinement turn {turn + 1}/{self.max_refinement_turns}")

            try:
                # 调用 LLM
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.refinement_tools,
                    temperature=0.7,
                )

                assistant_message = completion.choices[0].message

                # 打印 LLM 输出到控制台（绿色）
                if assistant_message.content:
                    print_llm_output(
                        f"LLM Refinement Turn {turn + 1}/{self.max_refinement_turns} (node {node.id[:8]})",
                        assistant_message.content,
                        max_length=1500
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                    }
                )

                # 检查是否停止
                if completion.choices[0].finish_reason == "stop":
                    logger.info(f"Model chose to stop at turn {turn + 1}")
                    break

                # 执行工具调用
                if (
                    completion.choices[0].finish_reason == "tool_calls"
                    and assistant_message.tool_calls
                ):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            "tool_calls": assistant_message.tool_calls,
                        }
                    )

                    for tool_call in assistant_message.tool_calls:
                        try:
                            call_args = json.loads(tool_call.function.arguments)

                            # 根据工具类型调用
                            if tool_call.function.name == "run_python_code":
                                result = self.tool_functions["run_python_code"](
                                    code=call_args.get("code", current_code), node=node
                                )

                                # 添加工具响应
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(
                                            result, ensure_ascii=False
                                        ),
                                    }
                                )

                                # 如果执行成功，更新当前代码
                                if result.get("success"):
                                    new_code = call_args.get("code", current_code)
                                    if new_code != current_code:
                                        current_code = new_code

                                        # 使用 LLM 提取 metric
                                        temp_node = MCTSNode(
                                            strategy=type(
                                                "Strategy",
                                                (),
                                                {
                                                    "code": new_code,
                                                    "plan": node.strategy.plan,
                                                    "metadata": {},
                                                },
                                            )(),
                                            parent=node.parent,
                                        )
                                        temp_node._term_out = result["output"].split(
                                            "\n"
                                        )

                                        llm_response = self._parse_exec_result_with_llm(
                                            temp_node
                                        )
                                        new_metric = llm_response.get("metric")

                                        if new_metric is not None:
                                            # 判断是否有改进
                                            improvement = self._calculate_improvement(
                                                new_metric,
                                                best_metric,
                                                initial_result.lower_is_better,
                                            )

                                            if improvement > 0:
                                                logger.info(
                                                    f"Improvement found: {best_metric:.6f} -> {new_metric:.6f} "
                                                    f"({improvement:+.6f})"
                                                )
                                                best_code = new_code
                                                best_metric = new_metric

                                                # 更新 best_result
                                                best_result = ExecutionResult(
                                                    success=True,
                                                    reward=self._calculate_reward(
                                                        is_buggy=False,
                                                        metric=new_metric,
                                                        lower_is_better=initial_result.lower_is_better,
                                                    ),
                                                    summary=f"Refined at turn {turn + 1}",
                                                    logger_info=result["output"].split(
                                                        "\n"
                                                    ),
                                                    metric=new_metric,
                                                    is_buggy=False,
                                                    lower_is_better=initial_result.lower_is_better,
                                                    metadata={
                                                        "refinement_turn": turn + 1,
                                                        "improvement": improvement,
                                                    },
                                                )

                            elif tool_call.function.name == "web_search":
                                result = self.tool_functions["web_search"](**call_args)

                                # 添加工具响应
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(
                                            result, ensure_ascii=False
                                        ),
                                    }
                                )

                        except Exception as e:
                            logger.error(
                                f"Tool call {tool_call.function.name} failed: {e}"
                            )
                            error_result = {"error": str(e), "success": False}
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(
                                        error_result, ensure_ascii=False
                                    ),
                                }
                            )

            except Exception as e:
                logger.error(f"Turn {turn + 1} failed: {e}")
                break

        # === 最终代码生成和总结步骤 ===
        logger.info("Generating final submission code and summary...")

        # 构建最终代码生成的提示词
        final_prompt = f"""Based on all the explorations and refinements above, please generate the FINAL submission code.

**Current Best Code:**
```python
{best_code}
```

**Best Metric Achieved:** {best_metric:.6f}

**Task:** {self.task_description}

Please provide your final, polished code that incorporates all the improvements discovered during refinement.
- This should be the BEST version of the code
- Include any hyperparameter tuning that worked well
- Make sure the code is complete and executable
- Return ONLY the code in <code> tags, no additional explanation needed

Format your response as:
<code>
# your final code here
</code>"""

        # 添加到消息历史
        messages.append({"role": "user", "content": final_prompt})

        try:
            # 调用 LLM 生成最终代码（不使用工具）
            final_response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=8000,
            )

            content = final_response.choices[0].message.content

            # 打印 LLM 输出到控制台（绿色）
            print_llm_output(f"LLM Final Code Generation (node {node.id[:8]})", content, max_length=3000)

            final_code = extract_code_from_response(content)

            if final_code:
                logger.info("Final submission code generated successfully")

                # 执行最终代码验证
                temp_node = MCTSNode(
                    strategy=type("Strategy", (), {
                        "code": final_code,
                        "plan": node.strategy.plan,
                        "metadata": {},
                    })(),
                    parent=node.parent,
                )

                validation_result = self._execute_once(temp_node)

                # === 让 LLM 基于 messages 生成总结 ===
                logger.info("Generating refinement summary...")

                summary_prompt = f"""Based on all the explorations and refinements during this multi-turn refinement process, please provide a comprehensive summary.

**Initial Metric:** {initial_result.metric:.6f}
**Best Metric Achieved:** {best_metric:.6f}
**Improvement:** {best_metric - initial_result.metric:.6f}
**Number of Refinement Turns:** {self.max_refinement_turns}

Please summarize:
1. What approaches were tried during refinement
2. What hyperparameters or techniques worked best
3. What the key improvements were
4. Any notable findings or insights

Keep the summary concise (2-3 paragraphs)."""

                summary_messages = messages + [{"role": "user", "content": summary_prompt}]
                summary_response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=summary_messages,
                    temperature=0.3,
                    max_tokens=1000,
                )

                llm_summary = summary_response.choices[0].message.content.strip()

                # 打印 LLM 输出到控制台（绿色）
                print_llm_output(f"LLM Refinement Summary (node {node.id[:8]})", llm_summary, max_length=1000)

                # 合并 summary
                combined_summary = f"""=== LLM Refinement Summary ===
{llm_summary}

=== Execution Result ===
{validation_result.summary}
"""

                # 更新 best_code 和 best_result
                best_code = final_code
                best_result = ExecutionResult(
                    success=validation_result.success,
                    reward=validation_result.reward,
                    summary=combined_summary,
                    logger_info=validation_result.logger_info,
                    metric=validation_result.metric,
                    is_buggy=validation_result.is_buggy,
                    lower_is_better=validation_result.lower_is_better,
                    metadata={
                        "final_submission": True,
                        "original_metric": best_metric,
                        "llm_summary": llm_summary,
                    },
                )

                logger.info(
                    f"Final code metric: {validation_result.metric:.6f}, "
                    f"reward: {validation_result.reward:.2f}"
                )
                logger.info(f"Refinement summary:\n{llm_summary}")
            else:
                logger.warning("Failed to extract final code from LLM response")
                # 使用最佳代码
                best_code = best_code

        except Exception as e:
            logger.error(f"Final code generation failed: {e}")
            # 使用最佳代码
            best_code = best_code

        # 如果找到了更好的代码，更新节点
        if best_code != node.strategy.code:
            node.strategy.code = best_code
            logger.info(
                f"Refinement completed: metric improved from {initial_result.metric:.6f} "
                f"to {best_metric:.6f}"
            )

        return best_result

    def _build_refinement_messages(
        self, node: MCTSNode, initial_result: ExecutionResult
    ) -> List[Dict]:
        """构建多轮微调的初始消息"""
        system_prompt = f"""You are an expert machine learning engineer performing multi-turn code refinement.

**Task Description:**
{self.task_description}

**Current Code:**
```python
{node.strategy.code}
```

**Current Strategy:**
{node.strategy.plan}

**Initial Execution Result:**
- Metric: {initial_result.metric}
- Summary: {initial_result.summary}
- Success: {initial_result.success}

Your goal is to improve the code performance through iterative refinement. You have access to the following tools:

1. `run_python_code`: Execute Python code locally and get the raw console output
   - Use this to test different code variations
   - Analyze the output to extract validation metrics
   - Compare results to find improvements

2. `web_search`: Search the web for ML best practices, library documentation, or techniques
   - Use this when you need information about optimal hyperparameters
   - Search for solutions to specific problems you encounter

**Important Constraints:**
- Focus on hyperparameter tuning and minor code improvements
- Do NOT make major architectural changes
- Do NOT change data processing pipelines
- Keep the code executable and complete
- Each improvement should be incremental
- Extract metrics from the console output to track improvements

Use tools iteratively to find the best configuration."""

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Please analyze the current code and suggest improvements to improve the metric.",
            },
        ]

    def _calculate_improvement(
        self, new_metric: float, old_metric: float, lower_is_better: bool
    ) -> float:
        """计算改进幅度"""
        if lower_is_better:
            return old_metric - new_metric  # 越小越好，所以 old - new
        else:
            return new_metric - old_metric  # 越大越好，所以 new - old

    # ==================== Tool Functions ====================

    def _tool_run_python_code(
        self, code: str, node: MCTSNode, **kwargs
    ) -> Dict[str, Any]:
        """
        工具函数：执行 Python 代码并返回原始控制台输出

        Args:
            code: 要执行的代码
            node: 当前节点

        Returns:
            执行结果字典，包含原始控制台输出
        """
        try:
            exec_id = f"refinement_{node.id[:8]}_{int(time.time())}"

            # 执行代码
            exec_result = self.interpreter.run(
                code=code, id=exec_id, reset_session=True
            )

            # 返回原始控制台输出
            output = "\n".join(exec_result.term_out)
            if exec_result.exc_type:
                output += f"\n\nError: {exec_result.exc_type}"
                if exec_result.exc_info:
                    output += f"\n{exec_result.exc_info}"

            return {
                "success": exec_result.exc_type is None,
                "output": output,
                "error": exec_result.exc_type,
                "exec_time": exec_result.exec_time,
            }

        except Exception as e:
            logger.error(f"run_python_code failed: {e}")
            return {"success": False, "output": str(e), "error": str(e)}


class Envisioner:
    """
    Envisioner - 全局探索的统领者和决策者

    核心功能：
    1. 维护一个全局的蒙特卡洛树搜索作为全局的树探索结构
    2. 维护一个全局更新的 Logger Pool 保证智能体的策略始终不发生偏移

    MCTS 探索循环：
    - Selection: 使用 UCT 算法选择需要扩展的节点
    - Expansion: 生成 2-3 个新的策略节点
    - Simulation: 并行执行这些策略（使用 LLM 从 term_out 提取 metric）
    - Backpropagation: 回溯更新搜索树
    """

    def __init__(
        self,
        interpreter,
        memory: Memory,
        task_description: str = "",
        model_name: str = "deepseek-reasoner",
        feedback_model_name: str = "deepseek-chat",
        exploration_constant: float = 1.414,
        max_executor_count: int = 3,
        max_node_expansions: int = 5,
        system_prompt_path: str = "prompt/system_prompt.md",
    ):
        self.interpreter = interpreter
        self.memory = memory
        self.task_description = task_description
        self.model_name = model_name  # 用于生成策略的模型
        self.feedback_model_name = feedback_model_name  # 用于提取 metric 的模型
        self.exploration_constant = exploration_constant
        self.max_executor_count = max_executor_count
        self.max_node_expansions = max_node_expansions  # 节点最大扩展次数

        # 初始化 LLM 客户端
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )

        # 加载系统提示词
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        # MCTS 搜索树
        self.root_node: Optional[MCTSNode] = None

        # 线程安全
        self.tree_lock = threading.Lock()

        # 执行器线程池
        self.executor_pool: List[Executor] = []
        self.executor_threads: List[threading.Thread] = []
        self.task_queue: Queue[ExecutorTask] = Queue()
        self.active_tasks: Dict[str, Future] = {}
        self.is_running = False

        # 统计信息
        self.stats = {
            "selections": 0,
            "expansions": 0,
            "simulations": 0,
            "backpropagations": 0,
        }

        # 全局最佳节点跟踪
        self.best_metric: Optional[float] = None
        self.best_node: Optional[MCTSNode] = None
        self.best_lock = threading.Lock()

    def _load_system_prompt(self, path: str) -> str:
        """加载系统提示词"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"System prompt file not found: {path}")
            return "You are an expert problem solver."

    # ==================== MCTS Steps ====================

    def selection(self, node: Optional[MCTSNode] = None) -> MCTSNode:
        """
        第一步：Selection（选择）—— 寻找最具潜力的"生长点"

        流程：
        1. 从 Root 节点开始
        2. 判断逻辑：
           - 如果当前节点尚未达到最大扩展次数，则停在该节点
           - 如果当前节点已经充分扩展，则计算其所有子节点的 UCT 值
        3. 递归向下：选择 UCT 最高的子节点，直到找到需要扩展的节点

        Args:
            node: 起始节点（默认为根节点）

        Returns:
            需要扩展的节点
        """
        self.stats["selections"] += 1

        with self.tree_lock:
            if node is None:
                node = self.root_node

            if node is None:
                raise ValueError("No root node available for selection")

            current = node

            # 向下遍历，直到找到未完全扩展的节点
            while current.is_fully_expanded_with_expected() and current.children:
                # 选择 UCT 值最高的子节点
                current = current.select_best_child(self.exploration_constant)
                if current is None:
                    break

            logger.debug(
                f"Selection: selected node {current.id[:8]}, "
                f"visits={current.visits}, expansion={current.expansion}"
            )

            return current

    def expansion_and_dispatch(
        self, node: MCTSNode, num_strategies: int = 3
    ) -> List[MCTSNode]:
        """
        第二步：Expansion & Task Dispatch（扩展与分发）—— Envisioner 发力

        流程：
        1. Context 准备：Envisioner 调用 fetch_memory 获取具体的智能体记忆
        2. 生成变体：Envisioner 基于 Memory 一次性生成 2~3 个新策略节点
        3. 锁定资源：
           - 将这些新节点加入当前节点的 children 列表
           - 更新当前节点的 visits 和 expansion
           - 更新 expected_child_count，防止并行环境下重复扩展

        Args:
            node: 要扩展的节点
            num_strategies: 要生成的策略数量

        Returns:
            新生成的子节点列表
        """
        self.stats["expansions"] += 1

        # 检查是否可以扩展，并提前锁定节点
        with self.tree_lock:
            if node.is_fully_expanded_with_expected():
                logger.debug(f"Node {node.id[:8]} is already fully expanded")
                return []

            # 检查是否已有预期子节点（并行控制）
            # 修复：使用 >= 而不是 >，因为当 expected_child_count == len(children) 时，
            # 说明之前的扩展已经完成（或者失败后重置），可以重新扩展
            if node.expected_child_count > len(node.children):
                logger.debug(
                    f"Node {node.id[:8]} already has pending expansions "
                    f"({len(node.children)}/{node.expected_child_count})"
                )
                return []

            # 提前设置预期数量，锁定节点（防止并发扩展）
            # 新的预期数量应该是当前子节点数 + 要生成的数量
            node.set_expected_children(len(node.children) + num_strategies)

        # 准备上下文（在锁外执行）
        memory_context = self._fetch_memory(node)

        # 生成新策略（在锁外执行，耗时操作）
        new_strategies = self._generate_strategies(
            task_description=self.task_description,
            memory_context=memory_context,
            num_strategies=num_strategies,
        )

        if not new_strategies:
            # 失败时重置预期数量
            with self.tree_lock:
                node.set_expected_children(len(node.children))
            logger.warning(
                f"Failed to generate strategies for node {node.id[:8]}. "
                f"Node state: expansion={node.expansion}/{node.max_expansions}, "
                f"children={len(node.children)}"
            )
            return []

        # 创建新节点
        child_nodes = []
        with self.tree_lock:
            for strategy_plan in new_strategies:
                strategy = Strategy(
                    id=uuid.uuid4().hex,
                    plan=strategy_plan.get("plan_content", ""),
                    metadata={
                        "parent_id": node.id,
                        "reasoning": strategy_plan.get("reasoning", ""),
                        "agent": "envisioner",
                        "expansion_time": time.time(),
                    },
                )
                child_node = MCTSNode(
                    strategy, parent=node, max_expansions=node.max_expansions
                )
                node.add_child(child_node)
                child_nodes.append(child_node)

            # 更新扩展次数
            node.increment_expansion()

            logger.info(
                f"Expanded node {node.id[:8]}, " f"created {len(child_nodes)} children"
            )

        return child_nodes

    def simulation(self, nodes: List[MCTSNode]) -> Dict[str, ExecutionResult]:
        """
        第三步：Simulation / Evaluation（模拟与执行）—— Executor 细节微调

        流程：
        1. 并行执行：将生成的策略分发给并行运行的多个 Executor
        2. Executor 执行完成后返回：
           - Logger 信息
           - scalar reward
           - summary
        3. 状态标记：评测得到当前节点的分值

        注意：execute 的执行需要和主线程抽离，单独开线程进行执行，
        但需要保证线程总数不超过 max_executor_count

        Args:
            nodes: 要执行的节点列表

        Returns:
            节点 ID 到执行结果的映射
        """
        self.stats["simulations"] += len(nodes)

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_executor_count) as executor:
            future_to_node = {
                executor.submit(self._execute_single_node, node): node for node in nodes
            }

            for future in future_to_node:
                node = future_to_node[future]
                try:
                    result = future.result(timeout=1800)
                    results[node.id] = result
                    logger.info(
                        f"Simulation completed for node {node.id[:8]}, "
                        f"reward: {result.reward:.3f}"
                    )
                except Exception as e:
                    logger.error(f"Simulation failed for node {node.id[:8]}: {e}")
                    results[node.id] = ExecutionResult(
                        success=False,
                        reward=-1.0,
                        summary=f"Execution failed: {str(e)}",
                        logger_info=[f"Error: {str(e)}"],
                    )

        return results

    def backpropagation(
        self, nodes: List[MCTSNode], results: Dict[str, ExecutionResult]
    ):
        """
        第四步：Backpropagation（回溯更新）—— 全局同步

        流程：
        1. 单点更新：当一个 Executor 完成任务后，立即更新该节点的 total_reward
        2. 路径更新：沿着 parent 指针一路向上，更新所有祖先节点的 visits 和 total_reward
        3. 更新全局最佳节点

        Args:
            nodes: 已执行的节点列表
            results: 执行结果字典
        """
        self.stats["backpropagations"] += len(nodes)

        for node in nodes:
            if node.id not in results:
                continue

            result = results[node.id]
            reward = result.reward
            metric = result.metric

            # 更新全局最佳节点
            if not result.is_buggy and metric is not None:
                with self.best_lock:
                    if self.best_metric is None:
                        self.best_metric = metric
                        self.best_node = node
                        logger.info(
                            f"New best metric: {metric:.4f} (node {node.id[:8]})"
                        )
                    else:
                        # 需要判断 lower_is_better
                        lower_is_better = (
                            result.lower_is_better
                            if result.lower_is_better is not None
                            else False
                        )

                        if lower_is_better:
                            # 越小越好
                            if metric < self.best_metric:
                                improvement = self.best_metric - metric
                                self.best_metric = metric
                                self.best_node = node
                                logger.info(
                                    f"New best metric: {metric:.4f} (improvement: {improvement:.4f}, "
                                    f"node {node.id[:8]})"
                                )
                        else:
                            # 越大越好
                            if metric > self.best_metric:
                                improvement = metric - self.best_metric
                                self.best_metric = metric
                                self.best_node = node
                                logger.info(
                                    f"New best metric: {metric:.4f} (improvement: {improvement:.4f}, "
                                    f"node {node.id[:8]})"
                                )

            # 回溯更新路径
            current = node
            with self.tree_lock:
                while current is not None:
                    current.update(reward)
                    current = current.parent

            # 记录到 Memory
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
                metadata={
                    "executor_result": {
                        "success": result.success,
                        "execution_time": result.execution_time,
                    }
                },
            )

            logger.debug(f"Backpropagated reward {reward:.3f} from node {node.id[:8]}")

    # ==================== MCTS Search Loop ====================

    def mcts_step(self, budget: int = 10):
        """
        执行一轮 MCTS 搜索

        Args:
            budget: 本次搜索的迭代次数
        """
        logger.info(f"Starting MCTS step with budget {budget}")

        for i in range(budget):
            try:
                # 1. Selection
                selected_node = self.selection()
                logger.info(f"Selecting Node for {selected_node}")
                if selected_node is None:
                    logger.warning("No node selected for expansion")
                    continue

                # 2. Expansion & Task Dispatch
                new_nodes = self.expansion_and_dispatch(selected_node, num_strategies=3)
                if not new_nodes:
                    logger.debug(f"No new nodes created from {selected_node.id[:8]}")
                    continue

                # 3. Simulation / Evaluation
                results = self.simulation(new_nodes)

                # 4. Backpropagation
                self.backpropagation(new_nodes, results)

                logger.debug(f"MCTS iteration {i + 1}/{budget} completed")

            except Exception as e:
                logger.error(f"MCTS iteration {i + 1} failed: {e}")

        logger.info(f"MCTS step completed. Statistics: {self.stats}")

    # ==================== Helper Methods ====================

    def _fetch_memory(self, node: MCTSNode) -> str:
        """
        获取相关记忆（fetch_memory）- 为扩展节点提供上下文

        Args:
            node: 当前节点

        Returns:
            格式化的记忆上下文
        """
        # 使用 Memory 类的智能上下文检索
        memory_context = self.memory.fetch_context_for_expansion(
            current_node=node.id
        )

        return memory_context

    def _extract_strategies_with_llm(
        self, original_response: str, num_strategies: int
    ) -> List[Dict[str, str]]:
        """
        使用 deepseek-chat 作为提取者，将原始 LLM 响应重新格式化为标准结构

        Args:
            original_response: 原始 LLM 响应文本
            num_strategies: 需要提取的策略数量

        Returns:
            策略列表，每个策略包含 plan_content 和 reasoning
        """
        try:
            extraction_prompt = f"""You are a text extraction specialist. Your task is to extract and reformat machine learning strategies from the given text.

**Original LLM Response:**
```
{original_response}
```

**Your Task:**
Extract {num_strategies} distinct strategies from the text above and format them using the following EXACT structure:

<strategy>
<plan_content>
[The detailed action plan - what model to use, what preprocessing steps, what feature engineering, etc.]
</plan_content>
<reasoning>
[The reasoning for why this approach might work - why this model/technique is suitable, advantages, potential risks]
</reasoning>
</strategy>

**Important Instructions:**
1. Extract EXACTLY {num_strategies} strategies
2. Use ONLY the information from the original text - do not invent new strategies
3. Format MUST match the XML-like tags exactly: <strategy><plan_content>...</plan_content><reasoning>...</reasoning></strategy>
4. Each strategy should have both <plan_content> and <reasoning> sections
5. Keep the content concise but complete

Provide your response below:"""

            logger.info(f"Calling deepseek-chat to extract strategies from original response")

            # 使用 deepseek-chat 提取
            response = self.client.chat.completions.create(
                model=self.feedback_model_name,  # deepseek-chat
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise text extraction specialist. You extract and reformat information following exact structural requirements."
                    },
                    {"role": "user", "content": extraction_prompt},
                ],
                temperature=0.0,  # 使用确定性输出
                max_tokens=3000,
            )

            extracted_content = response.choices[0].message.content

            # 打印提取结果到控制台（蓝色）
            console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
            console.print(f"[bold blue]EXTRACTION MODEL OUTPUT ({self.feedback_model_name})[/bold blue]")
            console.print(f"[bold blue]{'=' * 60}[/bold blue]")
            console.print(f"[blue]{extracted_content[:1500]}[/blue]")
            if len(extracted_content) > 1500:
                console.print(f"[dim]... [truncated {len(extracted_content) - 1500} chars] ...[/dim]")
            console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")

            # 再次解析提取后的内容
            from .utils import parse_strategy_response
            strategies = parse_strategy_response(extracted_content)

            if strategies:
                logger.info(f"Extraction successful: extracted {len(strategies)} strategies")
            else:
                logger.warning("Extraction completed but no strategies were parsed")

            return strategies

        except Exception as e:
            logger.error(f"Failed to extract strategies with deepseek-chat: {e}", exc_info=True)
            return []

    def _generate_strategies(
        self, task_description: str, memory_context: str, num_strategies: int = 3
    ) -> List[Dict[str, str]]:
        """
        使用 LLM 生成新策略

        Args:
            task_description: 任务描述
            memory_context: 记忆上下文
            num_strategies: 要生成的策略数量

        Returns:
            策略列表，每个策略包含 plan_content 和 reasoning
        """
        try:
            user_prompt = format_plan_prompt(
                task_description=task_description, memory_context=memory_context
            )

            logger.info(
                f"Calling LLM to generate strategies. "
                f"model={self.model_name}, num_strategies={num_strategies}"
            )

            # 调用 LLM 生成策略
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # 打印 LLM 输出到控制台（绿色）
            print_llm_output(f"LLM Strategy Generation ({self.model_name})", content, max_length=2000)

            if not content:
                logger.error("Empty LLM response")
                return []

            logger.debug(f"LLM response received: {len(content)} characters")

            # 第一次尝试：直接解析响应
            strategies = parse_strategy_response(content)

            # 如果解析失败或策略数量不足，使用 deepseek-chat 作为提取者
            if not strategies or len(strategies) < num_strategies:
                logger.warning(
                    f"Initial parsing yielded {len(strategies)} strategies (expected {num_strategies}). "
                    f"Using deepseek-chat as extractor to reformat the response."
                )

                # 使用提取模型重新格式化
                extraction_result = self._extract_strategies_with_llm(content, num_strategies)

                if extraction_result:
                    strategies = extraction_result
                    logger.info(f"Successfully extracted {len(strategies)} strategies using deepseek-chat")
                else:
                    # 即使提取失败，也返回原始解析结果（如果有）
                    if strategies:
                        logger.warning(f"Extraction failed, using {len(strategies)} initially parsed strategies")
                    else:
                        logger.error("Both initial parsing and extraction failed")
                        return []

            logger.info(f"Generated {len(strategies)} strategies from LLM")
            return strategies[:num_strategies]

        except Exception as e:
            logger.error(f"Failed to generate strategies: {e}", exc_info=True)
            return []

    def _execute_single_node(self, node: MCTSNode) -> ExecutionResult:
        """
        执行单个节点（用于线程池）

        Args:
            node: 要执行的节点

        Returns:
            执行结果（基于 LLM 提取 metric 的 reward）
        """
        # 获取当前全局最佳节点信息（线程安全）
        with self.best_lock:
            best_metric = self.best_metric
            best_node = self.best_node

        executor = Executor(
            self.interpreter,
            task_description=self.task_description,
            model_name=self.feedback_model_name,
            best_metric=best_metric,
            best_node=best_node,
            executor_id=id(threading.current_thread()),
        )
        return executor.execute(node)

    # ==================== Public API ====================

    def initialize_root(self, initial_strategy: Optional[Strategy] = None):
        """
        初始化根节点

        Args:
            initial_strategy: 初始策略（可选）
        """
        if initial_strategy is None:
            # 构建详细的初始提示词，包含完整的任务描述
            initial_prompt = f"""You are participating in MLE-bench, an offline version of Kaggle competitions.

You will be given a machine learning task. You must solve the task by training a model and running the model on the test set to produce a submission file.

**TASK DESCRIPTION:**
{self.task_description}

**Your Task:**
Based on the task description above, please provide an initial high-level strategy to solve this competition. Your strategy should include:

1. **Problem Understanding**: Briefly summarize what the task is about and what needs to be predicted
2. **Data Analysis**: What features are available and how might they be relevant
3. **Model Selection**: What types of models would be appropriate for this task
4. **Evaluation**: What metric is being used and how to optimize for it
5. **Approach**: A high-level outline of your approach (e.g., preprocessing, feature engineering, model training, validation)

Please be specific but concise (2-3 paragraphs). Focus on the key insights that will guide the search for an optimal solution."""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": initial_prompt},
                ],
                temperature=0.7,
                max_tokens=8888,
            )
            print(response.model_dump())

            content = response.choices[0].message.content

            # 打印 LLM 输出到控制台（绿色）
            print_llm_output("LLM Root Node Initialization", content, max_length=1500)

            if content:
                initial_strategy = Strategy(
                    id=uuid.uuid4().hex,
                    plan=content,
                    metadata={"agent": "envisioner_initial"},
                )

        with self.tree_lock:
            self.root_node = MCTSNode(
                initial_strategy, max_expansions=self.max_node_expansions
            )

        logger.info(f"Root node initialized: {self.root_node.id[:8]}")

    def get_best_node(self) -> Optional[MCTSNode]:
        """
        获取最佳节点（根据访问次数和奖励）

        Returns:
            最佳节点
        """
        with self.tree_lock:
            if self.root_node is None:
                return None

            if not self.root_node.children:
                return self.root_node

            # 选择访问次数最多的节点
            best_child = max(self.root_node.children, key=lambda child: child.visits)

            # 如果访问次数相同，选择奖励更高的
            max_visits = best_child.visits
            candidates = [
                child for child in self.root_node.children if child.visits == max_visits
            ]

            if len(candidates) > 1:
                best_child = max(candidates, key=lambda child: child.value)

            return best_child

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.tree_lock:
            return {
                **self.stats,
                "root_node_id": self.root_node.id if self.root_node else None,
                "tree_size": (
                    self._count_tree_nodes(self.root_node) if self.root_node else 0
                ),
                "memory_stats": self.memory.get_statistics(),
            }

    def _count_tree_nodes(self, node: MCTSNode) -> int:
        """递归计算树节点数量"""
        if node is None:
            return 0
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count
