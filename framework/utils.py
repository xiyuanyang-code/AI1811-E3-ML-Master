"""
Utility Functions for Framework 2
"""

import re
import http
import os
import json
from typing import List, Callable, Optional
from rich.console import Console

# Console for colored output
console = Console()


def print_parse_result(strategies: List[dict]):
    """Print parsed strategies to console for monitoring"""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]PARSED {len(strategies)} STRATEGIES[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
    for i, strategy in enumerate(strategies, 1):
        console.print(f"\n[bold yellow]Strategy {i}:[/bold yellow]")
        console.print(f"[cyan]Plan Content ({len(strategy.get('plan_content', ''))} chars):[/cyan]")
        console.print(f"[white]{strategy.get('plan_content', '')[:500]}[/white]")
        if len(strategy.get('plan_content', '')) > 500:
            console.print(f"[dim]... [truncated {len(strategy.get('plan_content', '')) - 500} chars] ...[/dim]")
        console.print(f"\n[cyan]Reasoning ({len(strategy.get('reasoning', ''))} chars):[/cyan]")
        console.print(f"[white]{strategy.get('reasoning', '')[:300]}[/white]")
        if len(strategy.get('reasoning', '')) > 300:
            console.print(f"[dim]... [truncated {len(strategy.get('reasoning', '')) - 300} chars] ...[/dim]")
        console.print(f"[bold cyan]{'-' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def create_pattern_extractor(pattern_name: str) -> Callable[[str], List[str]]:
    """
    创建一个提取指定模式标签内容的函数工厂

    Args:
        pattern_name: 要提取的标签名称（如 "code" 会匹配 <code></code> 之间的内容）

    Returns:
        一个函数，该函数接收字符串输入，返回匹配到的内容列表

    Example:
        extract_code = create_pattern_extractor("code")
        codes = extract_code(response_text)
    """
    escaped_pattern = re.escape(pattern_name)

    regex_pattern = re.compile(
        rf"<{escaped_pattern}>(.*?)</{escaped_pattern}>",
        re.DOTALL | re.IGNORECASE,
    )

    def extractor(input_str: str) -> List[str]:
        """
        从输入字符串中提取指定模式标签内的内容

        Args:
            input_str: 待提取的字符串

        Returns:
            匹配到的内容列表，如果没有匹配项返回空列表
        """
        matches = regex_pattern.findall(input_str)
        cleaned_matches = [match.strip() for match in matches]
        return cleaned_matches

    return extractor


def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    从 LLM 响应中提取代码

    Args:
        response_text: LLM 响应文本

    Returns:
        提取的代码字符串，如果没有找到则返回 None
    """
    # 方法1：使用 <code></code> 标签
    extract_code = create_pattern_extractor("code")
    codes = extract_code(response_text)
    if codes:
        return codes[0]

    # 方法2：使用 ```python ``` 代码块
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0]

    # 方法3：使用 ``` ``` 代码块
    pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0]

    return None


def extract_summary_from_response(response_text: str) -> Optional[str]:
    """
    从 LLM 响应中提取摘要

    Args:
        response_text: LLM 响应文本

    Returns:
        提取的摘要字符串，如果没有找到则返回 None
    """
    # 使用 <summary></summary> 标签
    extract_summary = create_pattern_extractor("summary")
    summaries = extract_summary(response_text)
    if summaries:
        return summaries[0]

    # 如果没有标签，尝试提取第一段
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('```'):
            return line

    return None


def parse_strategy_response(response_text: str) -> List[dict]:
    """
    解析 LLM 响应中的多个策略

    优先解析 <strategy><plan_content>...</plan_content><reasoning>...</reasoning></strategy> 格式

    Args:
        response_text: LLM 响应文本

    Returns:
        策略列表，每个策略包含 plan_content 和 reasoning
    """
    strategies = []

    # 方法1: 解析 <strategy><plan_content>...</plan_content><reasoning>...</reasoning></strategy> 格式（主要方法）
    strategy_pattern = re.compile(
        r'<strategy>\s*<plan_content>(.*?)</plan_content>\s*<reasoning>(.*?)</reasoning>\s*</strategy>',
        re.DOTALL | re.IGNORECASE
    )
    matches = strategy_pattern.findall(response_text)

    for match in matches:
        plan = match[0].strip()
        reasoning = match[1].strip()

        if plan:
            strategies.append({
                'plan_content': plan,
                'reasoning': reasoning or 'No reasoning provided'
            })

    if strategies:
        console.print(f"[green]✓ Successfully parsed {len(strategies)} strategies using <strategy> tags[/green]")
        print_parse_result(strategies)
        return strategies

    # 方法2: 尝试宽松解析 - 分别查找 <plan_content> 和 <reasoning> 标签
    plan_contents = re.findall(r'<plan_content>(.*?)</plan_content>', response_text, re.DOTALL | re.IGNORECASE)
    reasonings = re.findall(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)

    # 配对 plan_content 和 reasoning
    for i in range(max(len(plan_contents), len(reasonings))):
        plan = plan_contents[i].strip() if i < len(plan_contents) else ""
        reasoning = reasonings[i].strip() if i < len(reasonings) else ""

        if plan:
            strategies.append({
                'plan_content': plan,
                'reasoning': reasoning or 'No reasoning provided'
            })

    if strategies:
        console.print(f"[yellow]⚠ Parsed {len(strategies)} strategies using separate tags[/yellow]")
        print_parse_result(strategies)
        return strategies

    # 方法3: 回退 - 尝试 JSON 格式
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_data = json.loads(json_match.group())
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict):
                        plan = item.get('plan') or item.get('plan_content') or item.get('strategy', '')
                        reasoning = item.get('reasoning') or item.get('explanation', '')
                        if plan:
                            strategies.append({
                                'plan_content': plan,
                                'reasoning': reasoning
                            })
                if strategies:
                    console.print(f"[yellow]⚠ Parsed {len(strategies)} strategies from JSON format[/yellow]")
                    print_parse_result(strategies)
                    return strategies
    except json.JSONDecodeError:
        pass

    # 方法4: 最后回退 - 按段落分割
    console.print(f"[red]✗ Failed to parse using tags, falling back to paragraph parsing[/red]")
    paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip() and len(p.strip()) > 50]

    for para in paragraphs[:3]:  # 最多取前3个段落
        strategies.append({
            'plan_content': para,
            'reasoning': 'Extracted from paragraph (fallback)'
        })

    if strategies:
        print_parse_result(strategies)
    else:
        console.print(f"[red]✗ Failed to parse any strategies from response[/red]")

    return strategies


def calculate_uct(
    value: float,
    parent_visits: int,
    visits: int,
    exploration_constant: float = 1.414
) -> float:
    """
    计算 UCT 值

    UCT = value + C * sqrt(ln(parent_visits) / visits)

    Args:
        value: 节点的平均奖励值
        parent_visits: 父节点的访问次数
        visits: 当前节点的访问次数
        exploration_constant: 探索常数

    Returns:
        UCT 值
    """
    if visits == 0:
        return float('inf')

    exploitation = value
    exploration = exploration_constant * (parent_visits / visits) ** 0.5

    return exploitation + exploration


def normalize_reward(reward: float, min_reward: float = -1.0, max_reward: float = 1.0) -> float:
    """
    归一化奖励到 [0, 1] 范围

    Args:
        reward: 原始奖励值
        min_reward: 最小可能奖励值
        max_reward: 最大可能奖励值

    Returns:
        归一化后的奖励值
    """
    if max_reward == min_reward:
        return 0.5
    return (reward - min_reward) / (max_reward - min_reward)


def format_plan_prompt(
    task_description: str,
    memory_context: str,
    current_state: Optional[str] = None
) -> str:
    """
    格式化计划生成的提示词

    Args:
        task_description: 任务描述
        memory_context: 记忆上下文
        current_state: 当前状态（可选）

    Returns:
        格式化后的提示词
    """
    prompt = f"""Task: {task_description}

You are using Monte Carlo Tree Search (MCTS) as your exploration structure. The nodes you have previously explored and their information include:
Context from Memory:
{memory_context}
"""

    if current_state:
        prompt += f"\nCurrent State:\n{current_state}\n"

    prompt += """
Please provide 2-3 different strategies to solve this task.

**IMPORTANT OUTPUT FORMAT:**
You MUST format each strategy using the following XML-like tags:

<strategy>
<plan_content>
Your detailed action plan goes here. Include:
- Model architecture/approach
- Feature engineering steps
- Preprocessing techniques
- Validation strategy
- Any specific hyperparameters or techniques
</plan_content>
<reasoning>
Your reasoning for why this approach might work well. Include:
- Why this model/technique is suitable for the problem type
- What advantages it has over other approaches
- Any potential risks or limitations
</reasoning>
</strategy>

Provide 2-3 strategies, each wrapped in its own <strategy> tags.
"""

    return prompt



def web_search(query: str) -> str:
    """
    [General Web Search via Google] Perform a broad, general web search (Google Search) for any topic in any language.
    
    This is the **primary search tool** and should be used first to identify relevant web pages.
    It returns a structured JSON object containing snippets (summaries), titles, and crucially, the **URLs (web links)** of matching results.
    
    **AI Usage Guideline:**
    1.  Use this function to find the relevant URL(s) for a given query.
    2.  Once you have a specific URL of interest, you **must** pass that URL to the `web_parse` function to retrieve the full content of that page for detailed analysis.
    
    Args:
        query: The search query, which can be in any language (English, Chinese, etc.).
        
    Returns:
        A JSON string containing the search results, including snippets, titles, and the essential web links (URLs).
    """
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    conn.request("POST", "/search", payload, headers)
    data = conn.getresponse().read().decode("utf-8")
    return data


def web_parse(url: str) -> str:
    """
    [Specific Web Page Content Extractor] Fetch and extract the full, clean text content from a specific web page given its URL.
    
    This tool is designed for deep content retrieval. It takes a complete URL and returns the entire, main body content of that page, stripped of irrelevant elements like headers, footers, and advertisements.
    
    **AI Usage Guideline (Recommended Workflow):**
    1.  **DO NOT** use this function for general searching.
    2.  First, call `Google Search` with your keywords to get a list of potential URLs.
    3.  Then, call `web_parse` using a specific URL retrieved from the `Google Search` output to get the complete text for summary or detailed fact-checking.
    
    Args:
        url: The complete, absolute URL of the page to scrape (e.g., 'https://www.example.com/article-title').
        
    Returns:
        A JSON string containing the full, readable content of the specified URL.
    """
    conn = http.client.HTTPSConnection("scrape.serper.dev")
    payload = json.dumps({"url": url})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    conn.request("POST", "/", payload, headers)
    data = conn.getresponse().read().decode("utf-8")
    return data