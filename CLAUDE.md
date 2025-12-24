# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

E³-ML-Master is a long-horizon, self-evolved ML agent framework that uses Monte Carlo Tree Search (MCTS) and LLM-driven code generation to solve machine learning competition tasks. The framework employs a dual-core design: **Envisioner** (global exploration decision maker) and **Executor** (parallel execution engine).

### Core Components

- **Envisioner** (`framework/agent.py`): Maintains the global MCTS search tree, coordinates multiple parallel Executors, manages global Memory system, tracks global best node
- **Executor** (`framework/agent.py`): Executes strategy code for specific nodes, uses LLM to extract metrics from execution output, supports multi-turn refinement for code optimization
- **MCTSNode** (`framework/node.py`): Represents a strategy in the search tree with fields for visits, expansion, rewards, and UCT calculation
- **Memory** (`framework/memory.py`): Context-aware memory system storing all exploration history with indexed retrieval
- **Interpreter** (`interpreter/interpreter_parallel.py`): Parallel code execution engine that runs Python code in isolated environments
- **Backend** (`backend/`): LLM API interface layer supporting OpenAI-compatible and Qwen backends

## Environment Setup

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Python 3.13+ required
```

### API Keys

Configure in `.env` file:

```env
OPENAI_API_KEY="Your key"         # DeepSeek or OpenAI-compatible API key
BASE_URL="https://api.deepseek.com"  # API base URL
SERPER_API_KEY="Your key"         # For web search/parse tools
```

### Default Model Configuration

- **Envisioner strategy generation**: `deepseek-reasoner` (deepseek-r1)
- **Executor metric extraction**: `deepseek-chat` (deepseek-v3)

These can be overridden via `scripts/run.sh` or command-line arguments.

## Running the Agent

```bash
# Basic run (default task: nomad2018-predict-transparent-conductors)
python main.py

# Using run script for custom configuration
scripts/run.sh
```

The run script supports overriding models via:
- `agent.code.model` - Code execution model
- `agent.feedback.model` - Feedback/metric extraction model
- `agent.code.base_url` / `agent.feedback.base_url` - API endpoints
- `agent.code.api_key` / `agent.feedback.api_key` - API keys

## MCTS Algorithm Flow

The agent executes a four-phase MCTS loop (see `framework/agent.py:mcts_step`):

1. **Selection** (`selection`): Traverse tree from root using UCT to find node needing expansion
2. **Expansion** (`expansion_and_dispatch`): Generate 2-3 new strategies via LLM, create child nodes
3. **Simulation** (`simulation`): Parallel execute nodes via ThreadPoolExecutor, extract metrics via LLM
4. **Backpropagation** (`backpropagation`): Update node statistics, track global best, record to Memory

### Key Formulas

**UCT (Upper Confidence Bound for Trees):**
```
UCT(node) = value + C × sqrt(parent_visits / visits)
```
- Default `C = 1.414` (exploration_constant)
- `value = total_rewards / visits`

**Reward Calculation:**
```
-1.0  # Bug or no valid metric
 1.0  # Successful execution
 2.0  # Successful + improved best metric
```

## Architecture Notes

### Node Lifecycle

1. Created during Expansion with `Strategy` (plan + optional code)
2. Executed during Simulation via `Executor.execute()`
3. Updated during Backpropagation with reward
4. Stored in Memory with execution results

### Memory Retrieval

The `fetch_context_for_expansion` method provides context-aware retrieval:
- Ancestors path (root to current node)
- Parent/sibling/child node history
- Global best strategies (Top 10)
- Recent explorations
- Failed and buggy explorations

### Concurrency Control

- `tree_lock`: Protects MCTS tree modifications
- `best_lock`: Protects global best node updates
- `expected_child_count`: Pre-allocates expansion slots to prevent concurrent duplicate expansions

### Multi-Turn Refinement

Executor supports iterative code refinement:
- Tools: `run_python_code`, `web_search`, `web_parse`
- Generates initial code, then iterates up to `max_refinement_turns`
- LLM can call tools to experiment with hyperparameters/minor changes
- Final code generated after all refinement iterations

## Project Structure

```
framework/
├── agent.py          # Envisioner + Executor implementations
├── node.py           # MCTSNode and Strategy dataclasses
├── memory.py         # Memory system with indexed retrieval
└── utils.py          # Strategy parsing, web tools

interpreter/
└── interpreter_parallel.py  # Parallel code execution engine

backend/
├── call.py           # Unified LLM query interface
├── backend_openai.py # OpenAI-compatible backend
└── backend_qwen.py   # Qwen-specific backend

utils/
├── config_mcts.py    # Configuration loading, workspace prep
├── llm_caller.py     # LLM retry logic
└── metric.py         # Metric calculation utilities

scripts/
├── run.sh            # Main run script with model override support
├── kill.sh           # Kill running agent processes
└── grading.sh        # Grade submission against ground truth
```

## Important Implementation Details

- **Task configuration**: Loaded from `dataset/full_instructions/{EXP_ID}/full_instructions.txt`
- **Workspace**: Agent workspace prepared at `{workspace_dir}/` with data copied from `{data_dir}`
- **Submission path**: `./submission/submission.csv` (auto-created directory)
- **Logging**: Logs saved to `logs/{exp_name}/` with verbose and regular levels
- **Memory persistence**: Pickled to `logs/{exp_name}/memory.pkl`
- **Grading**: Custom grading function per task (see `main.py:grade()` for nomad2018 example)

### Agent Configuration

Key configuration parameters (in `config/` or via CLI):
- `agent.steps` - Total MCTS steps
- `agent.search.parallel_search_num` - Max parallel Executors
- `agent.decay.exploration_constant` - UCT exploration constant C
- `exec.timeout` - Code execution timeout per node

## Common Tasks

### Run with custom model

```bash
python main.py agent.code.model=gpt-4 agent.code.base_url=https://api.openai.com/v1
```

### Add new ML competition task

1. Create dataset directory: `data/{task_name}/prepared/public/`
2. Add task description: `dataset/full_instructions/{task_name}/full_instructions.txt`
3. Implement grading function in `main.py` if needed
4. Update `scripts/run.sh` with new `EXP_ID`
