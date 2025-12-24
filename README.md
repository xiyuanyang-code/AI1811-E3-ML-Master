# $E^3$-ML-Master: Advanced Envisioning-Executing for Goal-Driven Self-Evolved ML-Master

<div align="center">
  <img src="./assets/cover_new.png" width="80%" />
</div>

## Introduction


## Quick StartUp

### Environment SetUp

```bash
uv sync
source .venv/bin/activate
# Python 3.13.5
```

### APIKEY SetUp

By default, we use `deepseek-v3.2` (`deepseek-chat`) for the backbone of `Executor` Agent for writing codes and instruction followings, and we use `deepseek-r1` (`deepseek-reasoner`) for the backbone of core `Envisioner` Agent for self-reflection and expanding new strategies. 

- `OPENAI_API_KEY` with deepseek backbone: [Official API](https://platform.deepseek.com/usage)
- `SERPER_API_KEY` for web search and web parse tool calling. [SERPER_DEV](https://serper.dev/api-keys)

Write the following api keys into `.env` file.

```env
OPENAI_API_KEY="Your key"
BASE_URL="https://api.deepseek.com"
SERPER_API_KEY="Your key"
```

## Usage

```bash
python main.py
```

## Detailed Workflow

<div align="center">
  <img src="./assets/workflow_2.png" width="50%" />
</div>

> [!IMPORTANT]
> All the source code of the new architecture is in `framework` folders.