---
title: Super Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-step reasoning agent for GAIA Level 3 benchmark questions
tags:
  - langchain
  - langgraph
  - openai
  - gaia-benchmark
  - multi-step-reasoning
  - agent
  - tools
---

# 🧠 GAIA Level 3 LangGraph Agent

A production-ready multi-step reasoning agent built with **LangGraph**, **LangChain**, and **OpenAI GPT-4o** for tackling [GAIA benchmark](https://huggingface.co/gaia-benchmark) Level 3 questions.

## Architecture

```
Question → Planner → Agent (ReAct) ⇄ Tools → Critic → Synthesiser → Answer
```

### Nodes
| Node | Role |
|------|------|
| **Planner** | Decomposes the question, identifies required tools, creates a numbered execution plan |
| **Agent** | GPT-4o with tool binding via `bind_tools()`. Decides which tool to call next (ReAct loop) |
| **Tools** | LangGraph `ToolNode` executing: web_search, wikipedia, calculator, python_repl, reasoning_scratchpad |
| **Critic** | Evaluates evidence quality, identifies gaps, optionally triggers more tool calls |
| **Synthesiser** | Combines all evidence into a single precise GAIA-style answer |

### Tools
| Tool | Purpose |
|------|---------|
| 🔍 `web_search` | DuckDuckGo search for current/factual information |
| 📖 `wikipedia_search` | Encyclopedic knowledge retrieval |
| 🧮 `calculator` | Safe arithmetic, algebra, unit conversions |
| ⚡ `python_repl` | Sandboxed Python for complex computation |
| 🧠 `reasoning_scratchpad` | Structured chain-of-thought template for logic problems |

## Usage

1. Enter your **OpenAI API key** (never stored)
2. Paste a GAIA Level 3 question
3. Click **Run Agent**
4. Watch the agent reason step-by-step across the tabs

## Local Development

```bash
git clone <this-repo>
cd gaia-agent
pip install -r requirements.txt
python app.py
```

## HuggingFace Spaces Deployment

1. Fork this Space or create a new one with `gradio` SDK
2. Add your `OPENAI_API_KEY` in **Settings → Secrets**
3. The app reads it automatically from the environment

## Example Questions (GAIA Level 3 Style)

- *What is the total population of all countries whose names contain the word 'land', and which has the highest GDP per capita?*
- *Which element on the periodic table has an atomic number equal to the number of bones in an adult human hand?*
- *How many Academy Award Best Picture winners were released in the same year as the first Moon landing?*

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent graph with conditional edges
- [LangChain](https://github.com/langchain-ai/langchain) — tool abstraction and LLM integration
- [OpenAI GPT-4o](https://openai.com) — primary reasoning model
- [Gradio](https://gradio.app) — interactive web UI with streaming
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — web search tool
- [Wikipedia](https://pypi.org/project/wikipedia/) — encyclopedic knowledge
