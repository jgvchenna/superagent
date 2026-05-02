"""
tools.py — Custom LangChain tools for the GAIA Level 3 agent.
Each tool is a @tool-decorated function exposed to the LangGraph agent.
"""

from __future__ import annotations

import ast
import io
import math
import operator
import re
import sys
import textwrap
import traceback
from contextlib import redirect_stdout
from typing import Optional

import wikipedia
from duckduckgo_search import DDGS
from langchain_core.tools import tool


# ─── Web Search ───────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 6) -> str:
    """Search the web using DuckDuckGo. Use for current events, facts,
    recent data, or anything that requires up-to-date information.
    Returns a formatted list of results with titles, URLs, and snippets.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"**{r['title']}**\nURL: {r['href']}\n{r['body']}\n"
                )
        if not results:
            return "No results found for the query."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Web search error: {e}"


# ─── Wikipedia ────────────────────────────────────────────────────────────────

@tool
def wikipedia_search(query: str, sentences: int = 10) -> str:
    """Search Wikipedia for encyclopedic information about a topic.
    Returns a summary and key facts. Best for historical events, people,
    scientific concepts, geography, and established knowledge.
    """
    try:
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        output_parts = []
        for title in search_results[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                output_parts.append(
                    f"## {page.title}\nURL: {page.url}\n\n{summary}"
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first disambiguation option
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
                    output_parts.append(
                        f"## {page.title}\nURL: {page.url}\n\n{summary}"
                    )
                except Exception:
                    continue
            except Exception:
                continue

        if not output_parts:
            return f"Could not retrieve Wikipedia content for '{query}'."
        return "\n\n---\n\n".join(output_parts)
    except Exception as e:
        return f"Wikipedia search error: {e}"


# ─── Calculator ───────────────────────────────────────────────────────────────

# Safe math namespace
_SAFE_MATH = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_SAFE_MATH.update({
    "abs": abs, "round": round, "int": int, "float": float,
    "sum": sum, "min": min, "max": max, "len": len,
    "pow": pow, "divmod": divmod,
})

def _safe_eval(expr: str):
    """Evaluate a mathematical expression safely."""
    # Strip markdown fences if present
    expr = re.sub(r"```.*?```", "", expr, flags=re.DOTALL).strip()
    # Allow only safe tokens
    allowed = re.compile(r"^[\d\s\+\-\*\/\(\)\.\,\%\*\*\^epsqrtlogincostanfloorabsroundminmax_]+$", re.I)
    # Normalise ^ to **
    expr = expr.replace("^", "**")
    try:
        node = ast.parse(expr, mode="eval")
        return eval(compile(node, "<string>", "eval"), {"__builtins__": {}}, _SAFE_MATH)
    except Exception as e:
        return f"Eval error: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions and perform calculations.
    Supports arithmetic, algebra, trigonometry, logarithms, and unit conversions.
    Examples: '2**10', 'math.sqrt(144)', 'sum([1,2,3,4,5])', '(45 * 1.5) / 3.6'
    For multi-step problems, write each step on its own line.
    """
    lines = [l.strip() for l in expression.strip().splitlines() if l.strip()]
    results = []
    context = dict(_SAFE_MATH)

    for line in lines:
        # Skip comment lines
        if line.startswith("#"):
            results.append(line)
            continue
        # Handle assignment for context reuse
        if "=" in line and not line.startswith("="):
            try:
                exec(line, {"__builtins__": {}}, context)  # noqa: S102
                var, val_expr = line.split("=", 1)
                val = context.get(var.strip(), "?")
                results.append(f"{line.strip()}  →  {val}")
                continue
            except Exception:
                pass
        line_norm = line.replace("^", "**")
        try:
            val = eval(line_norm, {"__builtins__": {}}, context)  # noqa: S307
            results.append(f"{line}  =  {val}")
        except Exception as e:
            results.append(f"{line}  →  ERROR: {e}")

    return "\n".join(results) if results else "No valid expressions found."


# ─── Python Code Executor ─────────────────────────────────────────────────────

_FORBIDDEN = re.compile(
    r"\b(import\s+os|import\s+sys|import\s+subprocess|__import__|open\s*\(|"
    r"exec\s*\(|eval\s*\(|compile\s*\(|__builtins__|globals\s*\(|locals\s*\()\b"
)

@tool
def python_repl(code: str) -> str:
    """Execute Python code for data analysis, complex calculations, string manipulation,
    or algorithmic problem solving. Has access to: math, re, json, itertools, collections,
    functools, datetime, pandas, numpy. Returns stdout output and the value of the
    last expression. Do NOT use for file I/O, network calls, or system operations.
    """
    # Strip markdown fences
    code = re.sub(r"^```python\n?|^```\n?|\n?```$", "", code.strip(), flags=re.MULTILINE)

    if _FORBIDDEN.search(code):
        return "Error: Code contains forbidden operations (os, sys, subprocess, file I/O)."

    # Build a safe execution namespace
    safe_globals: dict = {"__builtins__": {
        "print": print, "range": range, "len": len, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "sum": sum, "min": min,
        "max": max, "abs": abs, "round": round, "bool": bool, "type": type,
        "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        "repr": repr, "format": format, "chr": chr, "ord": ord,
        "True": True, "False": False, "None": None,
    }}

    # Inject safe scientific libraries
    try:
        import math as _math, re as _re, json as _json
        import itertools as _it, collections as _col, functools as _fn
        from datetime import datetime as _dt, date as _date, timedelta as _td
        safe_globals.update({
            "math": _math, "re": _re, "json": _json,
            "itertools": _it, "collections": _col, "functools": _fn,
            "datetime": _dt, "date": _date, "timedelta": _td,
        })
    except ImportError:
        pass

    try:
        import numpy as np
        import pandas as pd
        safe_globals.update({"np": np, "pd": pd})
    except ImportError:
        pass

    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            # Try to capture the last expression value
            lines = code.strip().splitlines()
            if lines:
                try:
                    last_expr = ast.parse(lines[-1], mode="eval")
                    block = "\n".join(lines[:-1])
                    if block.strip():
                        exec(compile(block, "<code>", "exec"), safe_globals)  # noqa: S102
                    last_val = eval(compile(last_expr, "<code>", "eval"), safe_globals)  # noqa: S307
                except SyntaxError:
                    exec(compile(code, "<code>", "exec"), safe_globals)  # noqa: S102
                    last_val = None
            else:
                last_val = None

        output = stdout_capture.getvalue()
        result_parts = []
        if output.strip():
            result_parts.append(f"Output:\n{output.strip()}")
        if last_val is not None:
            result_parts.append(f"Result: {last_val!r}")
        return "\n".join(result_parts) if result_parts else "Code executed successfully (no output)."
    except Exception:
        output = stdout_capture.getvalue()
        tb = traceback.format_exc(limit=5)
        parts = []
        if output.strip():
            parts.append(f"Output before error:\n{output.strip()}")
        parts.append(f"Error:\n{tb}")
        return "\n".join(parts)


# ─── Reasoning / Chain-of-Thought Scratchpad ─────────────────────────────────

@tool
def reasoning_scratchpad(problem: str) -> str:
    """Use this tool to break down complex multi-step reasoning problems.
    Provide a problem statement and this tool will structure a step-by-step
    deductive reasoning chain. Best used for logic puzzles, causal chains,
    multi-constraint problems, and any question requiring systematic deduction.
    The tool returns a structured reasoning template with key components.
    """
    template = textwrap.dedent(f"""
    ═══ STRUCTURED REASONING CHAIN ═══

    PROBLEM STATEMENT:
    {problem}

    STEP 1 — IDENTIFY KNOWNS & UNKNOWNS
    • Known facts from the question:  [extract explicit facts]
    • What we need to find:           [the target answer]
    • Implicit constraints:           [unstated rules]

    STEP 2 — DECOMPOSE INTO SUB-PROBLEMS
    • Sub-problem A: ___
    • Sub-problem B: ___
    • Dependencies: A must be solved before B if ___

    STEP 3 — APPLY DOMAIN KNOWLEDGE
    • Relevant formulas / principles: ___
    • Edge cases to watch for: ___

    STEP 4 — SYNTHESISE ANSWER
    • Combine results from sub-problems: ___
    • Sanity check (units, magnitude, logic): ___

    STEP 5 — FINAL ANSWER
    • Answer: ___
    • Confidence: HIGH / MEDIUM / LOW
    • If LOW: what additional information would help?

    ═══════════════════════════════════
    """).strip()
    return template


# ─── Tool Registry ────────────────────────────────────────────────────────────

ALL_TOOLS = [
    web_search,
    wikipedia_search,
    calculator,
    python_repl,
    reasoning_scratchpad,
]

TOOL_METADATA = {
    "web_search":          {"icon": "🔍", "color": "#3B8BD4", "label": "Web Search"},
    "wikipedia_search":    {"icon": "📖", "color": "#D85A30", "label": "Wikipedia"},
    "calculator":          {"icon": "🧮", "color": "#1D9E75", "label": "Calculator"},
    "python_repl":         {"icon": "⚡", "color": "#7F77DD", "label": "Python REPL"},
    "reasoning_scratchpad":{"icon": "🧠", "color": "#BA7517", "label": "Reasoning"},
}
