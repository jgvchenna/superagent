"""
graph.py — LangGraph agent implementation for GAIA Level 3 questions.

Architecture:
  ┌─────────┐     ┌──────────────┐     ┌─────────────────┐
  │ Planner │────▶│ Agent (ReAct) │────▶│ Tool Executor   │
  └─────────┘     └──────────────┘     └─────────────────┘
                        ▲                       │
                        └───────────────────────┘
                        (loop until done or max_iter)
                               │ done
                               ▼
                        ┌──────────────┐
                        │   Critic     │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Synthesiser  │
                        └──────────────┘
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.tools import ALL_TOOLS, TOOL_METADATA

# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Core conversation
    messages: Annotated[list[BaseMessage], operator.add]
    question: str

    # Planning
    plan: str
    sub_questions: list[str]

    # Execution tracking
    iteration: int
    max_iterations: int
    tool_calls_log: list[dict]      # {tool, input, output, iteration}

    # Critique & synthesis
    critique: str
    final_answer: str
    confidence: str                  # HIGH / MEDIUM / LOW

    # UI streaming
    status: str                      # planning | reasoning | tool_use | critique | done
    current_node: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_llm(api_key: str, model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        streaming=False,
    )


def _extract_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Extract a structured log of all tool calls from message history."""
    log = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                log.append({
                    "tool": tc["name"],
                    "input": tc["args"],
                    "output": None,
                })
        elif isinstance(msg, ToolMessage):
            # Attach output to the most recent matching entry
            for entry in reversed(log):
                if entry["output"] is None:
                    entry["output"] = msg.content[:800]
                    break
    return log


# ─── Node 1: Planner ─────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are an expert strategic planner for GAIA Level 3 benchmark questions.
GAIA Level 3 questions are extremely difficult, requiring multi-step reasoning, diverse tool use,
and synthesis across multiple knowledge domains.

Your task:
1. Analyse the question carefully
2. Identify what domain knowledge is needed
3. Break it into concrete sub-questions that can each be answered with a single tool call
4. Produce a clear execution plan

Available tools:
- web_search: current events, recent data, URLs, news
- wikipedia_search: encyclopedic knowledge, history, science, geography
- calculator: arithmetic, algebra, unit conversions, statistics
- python_repl: complex computation, data analysis, string manipulation, algorithms
- reasoning_scratchpad: logic puzzles, multi-constraint deduction

Respond with a JSON block (and nothing else):
{
  "analysis": "brief analysis of what makes this question hard",
  "domain": "primary knowledge domain",
  "sub_questions": ["list", "of", "sub-questions"],
  "tool_strategy": "which tools to use and why",
  "plan": "numbered step-by-step execution plan as a single string"
}"""


def planner_node(state: AgentState, api_key: str, model: str) -> dict:
    llm = _build_llm(api_key, model)
    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}"),
    ])

    raw = response.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\n?|^```\n?|\n?```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        plan = data.get("plan", raw)
        sub_questions = data.get("sub_questions", [])
        analysis = data.get("analysis", "")
        tool_strategy = data.get("tool_strategy", "")
        plan_text = (
            f"ANALYSIS: {analysis}\n\n"
            f"DOMAIN: {data.get('domain', 'general')}\n\n"
            f"TOOL STRATEGY: {tool_strategy}\n\n"
            f"EXECUTION PLAN:\n{plan}"
        )
    except json.JSONDecodeError:
        plan_text = raw
        sub_questions = []

    # Inject the plan as context for the main agent
    plan_message = SystemMessage(
        content=f"""You are a GAIA Level 3 question-answering agent.
        
ORIGINAL QUESTION: {state['question']}

STRATEGIC PLAN:
{plan_text}

Execute this plan using the available tools. Be thorough and precise.
After gathering all necessary information, you will synthesise a final answer.
Use tools multiple times if needed. Verify key facts from multiple sources when possible."""
    )

    return {
        "plan": plan_text,
        "sub_questions": sub_questions,
        "messages": [plan_message, HumanMessage(content=state["question"])],
        "iteration": 0,
        "tool_calls_log": [],
        "status": "reasoning",
        "current_node": "planner",
    }


# ─── Node 2: React Agent ──────────────────────────────────────────────────────

def agent_node(state: AgentState, api_key: str, model: str) -> dict:
    llm = _build_llm(api_key, model).bind_tools(ALL_TOOLS)
    response = llm.invoke(state["messages"])

    new_iteration = state["iteration"] + 1
    status = "tool_use" if response.tool_calls else "reasoning"

    # Update tool log
    tool_log = list(state.get("tool_calls_log", []))
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_log.append({
                "tool": tc["name"],
                "input": tc["args"],
                "output": None,
                "iteration": new_iteration,
            })

    return {
        "messages": [response],
        "iteration": new_iteration,
        "tool_calls_log": tool_log,
        "status": status,
        "current_node": "agent",
    }


# ─── Node 3: Tool Executor ────────────────────────────────────────────────────

def tools_node_fn(state: AgentState) -> dict:
    """Wraps LangGraph's ToolNode to update our status and log tool outputs."""
    tool_node = ToolNode(ALL_TOOLS)
    result = tool_node.invoke(state)

    # Attach tool outputs back to log entries with None output
    messages = result.get("messages", [])
    tool_log = list(state.get("tool_calls_log", []))
    tool_outputs = [m for m in messages if isinstance(m, ToolMessage)]

    unresolved = [e for e in tool_log if e.get("output") is None]
    for entry, tm in zip(unresolved, tool_outputs):
        entry["output"] = tm.content[:1000]

    return {
        **result,
        "tool_calls_log": tool_log,
        "status": "reasoning",
        "current_node": "tools",
    }


# ─── Node 4: Critic ───────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a rigorous answer critic for GAIA Level 3 questions.
Your job is to assess the quality of reasoning and information gathered.

Evaluate:
1. Are all sub-questions answered?
2. Is the evidence sufficient and reliable?
3. Are there logical gaps or unsupported leaps?
4. Is the answer specific and precise (GAIA requires exact answers)?

Respond in JSON:
{
  "assessment": "SUFFICIENT | NEEDS_MORE_INFO",
  "gaps": ["list of gaps if any"],
  "critique": "detailed critique",
  "confidence": "HIGH | MEDIUM | LOW",
  "suggested_queries": ["additional queries if NEEDS_MORE_INFO"]
}"""


def critic_node(state: AgentState, api_key: str, model: str) -> dict:
    llm = _build_llm(api_key, model)

    # Summarise tool results for the critic
    tool_summary = "\n".join([
        f"• [{e['tool']}] Input: {str(e['input'])[:200]} → Output: {str(e.get('output',''))[:300]}"
        for e in state.get("tool_calls_log", [])
    ])

    # Pull out the last AI text response
    last_reasoning = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_reasoning = msg.content
            break

    prompt = f"""Question: {state['question']}

Plan:
{state.get('plan', 'N/A')}

Tools used and results:
{tool_summary or 'No tools used yet.'}

Agent's reasoning so far:
{last_reasoning[:1000]}

Please critique whether we have sufficient information to answer the question precisely."""

    response = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    raw = re.sub(r"^```json\n?|^```\n?|\n?```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        assessment = data.get("assessment", "SUFFICIENT")
        critique = data.get("critique", raw)
        confidence = data.get("confidence", "MEDIUM")
        gaps = data.get("gaps", [])
        suggested = data.get("suggested_queries", [])
    except json.JSONDecodeError:
        assessment = "SUFFICIENT"
        critique = raw
        confidence = "MEDIUM"
        gaps = []
        suggested = []

    # If gaps exist and we haven't hit max iterations, inject follow-up message
    new_messages = []
    if assessment == "NEEDS_MORE_INFO" and state["iteration"] < state["max_iterations"] - 1:
        follow_up = (
            f"The critic identified gaps: {'; '.join(gaps)}. "
            f"Please gather more information on: {'; '.join(suggested[:2])}."
        )
        new_messages = [HumanMessage(content=follow_up)]

    return {
        "critique": f"Assessment: {assessment}\nConfidence: {confidence}\n\n{critique}",
        "confidence": confidence,
        "messages": new_messages,
        "status": "critique",
        "current_node": "critic",
        "_critic_assessment": assessment,
    }


# ─── Node 5: Synthesiser ──────────────────────────────────────────────────────

SYNTHESISER_SYSTEM = """You are the final answer synthesiser for a GAIA Level 3 agent.
GAIA requires extremely precise, concise, and correct answers — often a single value, name, date, or short phrase.

Instructions:
1. Review all gathered evidence and reasoning
2. Synthesise the SINGLE best answer
3. GAIA answers are typically: a number, a name, a date, a short phrase, or a list
4. Be exact — do not hedge or add unnecessary qualifiers
5. If the question asks for a count, give the exact number
6. If the question asks for a name, give the exact name
7. Format your response as:

FINAL ANSWER: [your precise answer here]

REASONING SUMMARY:
[2-3 sentences explaining how you arrived at the answer]

CONFIDENCE: HIGH / MEDIUM / LOW
SOURCES: [key sources used]"""


def synthesiser_node(state: AgentState, api_key: str, model: str) -> dict:
    llm = _build_llm(api_key, model)

    tool_summary = "\n".join([
        f"[{e['tool']}] {str(e['input'])[:200]}\n  → {str(e.get('output',''))[:500]}"
        for e in state.get("tool_calls_log", [])
    ])

    # Collect all AI messages as reasoning trace
    reasoning_trace = "\n\n".join([
        f"[Turn {i}] {msg.content[:600]}"
        for i, msg in enumerate(state["messages"])
        if isinstance(msg, AIMessage) and msg.content
    ])

    prompt = f"""Question: {state['question']}

Strategic Plan:
{state.get('plan', 'N/A')}

Tool Results:
{tool_summary or 'No tools used.'}

Agent Reasoning Trace:
{reasoning_trace[:3000]}

Critic Assessment:
{state.get('critique', 'Not available')}

Now synthesise the final precise answer."""

    response = llm.invoke([
        SystemMessage(content=SYNTHESISER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    answer_text = response.content.strip()

    # Extract the FINAL ANSWER line
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", answer_text)
    final_answer = match.group(1).strip() if match else answer_text

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=answer_text)],
        "status": "done",
        "current_node": "synthesiser",
    }


# ─── Routing Logic ────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide whether to call tools, critique, or finish."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if state["iteration"] >= state["max_iterations"]:
        return "critic"

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tools"
        else:
            return "critic"

    return "critic"


def after_critic(state: AgentState) -> str:
    """After critique, decide whether to do more reasoning or synthesise."""
    assessment = state.get("_critic_assessment", "SUFFICIENT")
    if assessment == "NEEDS_MORE_INFO" and state["iteration"] < state["max_iterations"] - 1:
        return "agent"
    return "synthesiser"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph(api_key: str, model: str = "gpt-4o", max_iterations: int = 8):
    """Compile and return the LangGraph StateGraph."""

    # Bind runtime args via closures
    def _planner(state):   return planner_node(state, api_key, model)
    def _agent(state):     return agent_node(state, api_key, model)
    def _tools(state):     return tools_node_fn(state)
    def _critic(state):    return critic_node(state, api_key, model)
    def _synthesiser(state): return synthesiser_node(state, api_key, model)

    graph = StateGraph(AgentState)

    graph.add_node("planner",     _planner)
    graph.add_node("agent",       _agent)
    graph.add_node("tools",       _tools)
    graph.add_node("critic",      _critic)
    graph.add_node("synthesiser", _synthesiser)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "critic": "critic"},
    )

    graph.add_edge("tools", "agent")

    graph.add_conditional_edges(
        "critic",
        after_critic,
        {"agent": "agent", "synthesiser": "synthesiser"},
    )

    graph.add_edge("synthesiser", END)

    return graph.compile()


# ─── Run Helper (for streaming updates) ──────────────────────────────────────

def run_agent(
    question: str,
    api_key: str,
    model: str = "gpt-4o",
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Run the full agent and return the final state."""
    graph = build_graph(api_key, model, max_iterations)

    initial_state: AgentState = {
        "messages": [],
        "question": question,
        "plan": "",
        "sub_questions": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "tool_calls_log": [],
        "critique": "",
        "final_answer": "",
        "confidence": "MEDIUM",
        "status": "planning",
        "current_node": "start",
    }

    final_state = graph.invoke(initial_state)
    return final_state
