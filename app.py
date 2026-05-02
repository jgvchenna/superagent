"""
app.py — Gradio front-end for the GAIA Level 3 LangGraph agent.
Designed for HuggingFace Spaces deployment.

Run locally:  python app.py
HF Spaces:    set OPENAI_API_KEY in Space secrets
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
import traceback
from typing import Generator

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from superagent.graph import build_graph, AgentState
from superagent.tools import TOOL_METADATA

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

NODE_DESCRIPTIONS = {
    "planner":     ("🗺️", "Planner",     "Analysing the question and building a strategic execution plan"),
    "agent":       ("🤖", "Agent",       "Reasoning and deciding which tools to call next"),
    "tools":       ("🔧", "Tool Use",    "Executing tool call and processing results"),
    "critic":      ("🔎", "Critic",      "Evaluating gathered evidence and identifying gaps"),
    "synthesiser": ("✨", "Synthesiser", "Combining all evidence into a precise final answer"),
}

EXAMPLE_QUESTIONS = [
    "What is the total population of all countries whose names contain the word 'land', and which of those countries has the highest GDP per capita?",
    "If a train travels at 120 km/h and needs to cover a distance equal to the diameter of the Moon (3,474 km), how many hours would the journey take, and what is that in days and hours?",
    "Which element on the periodic table has an atomic number equal to the number of bones in the adult human hand, and what is its standard atomic weight?",
    "How many Academy Award Best Picture winners were released in the same year as the first human Moon landing, and what were their names?",
    "What is the sum of the number of letters in the names of all the planets in our solar system that have at least one moon?",
]

# ─── Formatting Helpers ───────────────────────────────────────────────────────

def _format_tool_badge(tool_name: str) -> str:
    meta = TOOL_METADATA.get(tool_name, {"icon": "🔧", "label": tool_name})
    return f"{meta['icon']} {meta['label']}"


def _truncate(text: str, n: int = 400) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f"… [+{len(text)-n} chars]"


def _render_plan(plan: str) -> str:
    if not plan:
        return ""
    lines = plan.strip().splitlines()
    rendered = ["### 🗺️ Strategic Plan\n"]
    for line in lines:
        line = line.strip()
        if not line:
            rendered.append("")
        elif line.startswith("ANALYSIS:"):
            rendered.append(f"**{line}**")
        elif line.startswith("DOMAIN:"):
            rendered.append(f"**{line}**")
        elif line.startswith("TOOL STRATEGY:"):
            rendered.append(f"**{line}**")
        elif line.startswith("EXECUTION PLAN:"):
            rendered.append(f"\n**{line}**")
        elif re.match(r"^\d+[\.\)]\s", line):
            rendered.append(f"  {line}")
        else:
            rendered.append(line)
    return "\n".join(rendered)


def _render_tool_log(tool_calls_log: list[dict]) -> str:
    if not tool_calls_log:
        return "*No tool calls yet.*"
    rows = ["| # | Tool | Input | Output |", "|---|------|-------|--------|"]
    for i, entry in enumerate(tool_calls_log, 1):
        tool = entry.get("tool", "?")
        badge = _format_tool_badge(tool)
        inp = _truncate(str(entry.get("input", "")), 120).replace("|", "\\|").replace("\n", " ")
        out = _truncate(str(entry.get("output") or "…"), 200).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {i} | {badge} | {inp} | {out} |")
    return "\n".join(rows)


def _render_reasoning(messages) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            text = _truncate(msg.content, 600)
            parts.append(f"**🤖 Agent:**\n{text}")
        elif isinstance(msg, ToolMessage):
            text = _truncate(msg.content, 400)
            parts.append(f"**🔧 Tool result:**\n```\n{text}\n```")
    return "\n\n---\n\n".join(parts[-6:]) if parts else "*Waiting for agent…*"


def _graph_dot(current_node: str) -> str:
    """Generate a minimal Graphviz DOT string showing the current graph state."""
    nodes = [
        ("planner",     "🗺️ Planner"),
        ("agent",       "🤖 Agent"),
        ("tools",       "🔧 Tools"),
        ("critic",      "🔎 Critic"),
        ("synthesiser", "✨ Synthesiser"),
    ]
    edges = [
        ("planner", "agent"),
        ("agent", "tools"),
        ("agent", "critic"),
        ("tools", "agent"),
        ("critic", "agent"),
        ("critic", "synthesiser"),
        ("synthesiser", "END"),
    ]
    lines = ['digraph G {', '  rankdir=LR;', '  node [shape=box, style=rounded, fontname="Arial"];']
    for node_id, label in nodes:
        if node_id == current_node:
            lines.append(f'  {node_id} [label="{label}", style="filled,rounded", fillcolor="#4F8EF7", fontcolor=white, penwidth=2];')
        else:
            lines.append(f'  {node_id} [label="{label}"];')
    lines.append('  END [shape=oval, label="END"];')
    for src, dst in edges:
        lines.append(f'  {src} -> {dst};')
    lines.append("}")
    return "\n".join(lines)


# ─── Main Agent Runner (streaming via generator) ──────────────────────────────

def run_agent_streaming(
    question: str,
    api_key: str,
    model: str,
    max_iterations: int,
) -> Generator[tuple, None, None]:
    """
    Yields tuples of Gradio component updates as the agent progresses.
    Yielded order matches the `outputs` list in the Gradio event handler.
    """

    def _yield(status="", node="", plan="", reasoning="", tool_log="",
               critique="", answer="", confidence="", progress=0):
        return (
            status,       # status_md
            node,         # node_md
            plan,         # plan_md
            reasoning,    # reasoning_md
            tool_log,     # tool_log_md
            critique,     # critique_md
            answer,       # answer_box
            confidence,   # confidence_md
            progress,     # progress_bar (0-1 float)
        )

    if not question.strip():
        yield _yield(status="⚠️ Please enter a question.", answer="", progress=0)
        return

    if not api_key.strip():
        yield _yield(status="⚠️ Please provide your OpenAI API key.", progress=0)
        return

    try:
        graph = build_graph(api_key=api_key, model=model, max_iterations=int(max_iterations))
    except Exception as e:
        yield _yield(status=f"❌ Failed to build graph: {e}", progress=0)
        return

    initial_state: AgentState = {
        "messages":       [],
        "question":       question,
        "plan":           "",
        "sub_questions":  [],
        "iteration":      0,
        "max_iterations": int(max_iterations),
        "tool_calls_log": [],
        "critique":       "",
        "final_answer":   "",
        "confidence":     "MEDIUM",
        "status":         "planning",
        "current_node":   "start",
    }

    # Emit initial "planning" state
    yield _yield(
        status=f"⏳ **Planning** — analysing question…",
        node="🗺️ **Planner** — building execution strategy",
        progress=0.05,
    )

    accumulated_state = dict(initial_state)
    step = 0
    total_steps = int(max_iterations) * 2 + 4  # rough estimate

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                step += 1
                progress = min(0.92, step / total_steps)

                # Merge into accumulated state
                for k, v in node_output.items():
                    if k == "messages" and isinstance(v, list):
                        accumulated_state["messages"] = (
                            accumulated_state.get("messages", []) + v
                        )
                    else:
                        accumulated_state[k] = v

                # Render each panel
                icon, label, desc = NODE_DESCRIPTIONS.get(
                    node_name, ("⚙️", node_name, "")
                )
                iteration = accumulated_state.get("iteration", 0)
                max_iter  = accumulated_state.get("max_iterations", max_iterations)
                tool_log  = accumulated_state.get("tool_calls_log", [])
                n_tools   = len(tool_log)

                status_text = (
                    f"**{icon} {label}** — {desc}  \n"
                    f"Iteration {iteration}/{max_iter} · {n_tools} tool call(s)"
                )

                # Node indicator with progress dots
                dot = lambda n: "🟢" if n == node_name else "⚪"
                node_text = (
                    f"{dot('planner')} Planner → "
                    f"{dot('agent')} Agent → "
                    f"{dot('tools')} Tools → "
                    f"{dot('critic')} Critic → "
                    f"{dot('synthesiser')} Synthesiser"
                )

                plan_text      = _render_plan(accumulated_state.get("plan", ""))
                reasoning_text = _render_reasoning(accumulated_state.get("messages", []))
                tool_log_text  = _render_tool_log(tool_log)
                critique_text  = accumulated_state.get("critique", "")
                answer_text    = accumulated_state.get("final_answer", "")
                confidence     = accumulated_state.get("confidence", "")

                conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
                confidence_text = f"{conf_emoji} **Confidence:** {confidence}" if confidence else ""

                yield _yield(
                    status=status_text,
                    node=node_text,
                    plan=plan_text,
                    reasoning=reasoning_text,
                    tool_log=tool_log_text,
                    critique=critique_text,
                    answer=answer_text,
                    confidence=confidence_text,
                    progress=progress,
                )

                # Small visual delay so the UI feels animated
                time.sleep(0.1)

    except Exception:
        tb = traceback.format_exc()
        yield _yield(
            status=f"❌ **Error during execution**",
            reasoning=f"```\n{tb}\n```",
            progress=0,
        )
        return

    # Final state
    final_answer   = accumulated_state.get("final_answer", "No answer generated.")
    confidence     = accumulated_state.get("confidence", "")
    conf_emoji     = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
    confidence_txt = f"{conf_emoji} **Confidence:** {confidence}"
    n_tools        = len(accumulated_state.get("tool_calls_log", []))

    yield _yield(
        status=f"✅ **Done** — {n_tools} tool call(s) · Confidence: {confidence}",
        node="⚪ Planner → ⚪ Agent → ⚪ Tools → ⚪ Critic → 🟢 Synthesiser",
        plan=_render_plan(accumulated_state.get("plan", "")),
        reasoning=_render_reasoning(accumulated_state.get("messages", [])),
        tool_log=_render_tool_log(accumulated_state.get("tool_calls_log", [])),
        critique=accumulated_state.get("critique", ""),
        answer=final_answer,
        confidence=confidence_txt,
        progress=1.0,
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
/* Global typography */
.gradio-container { font-family: 'Inter', sans-serif !important; }

/* Header banner */
#header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
    border: 1px solid #e94560;
}
#header-banner h1 { color: #ffffff; font-size: 1.8rem; margin: 0 0 6px 0; }
#header-banner p  { color: #a8b2d8; margin: 0; font-size: 0.95rem; }

/* Panel cards */
.panel-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    margin-top: 8px;
}

/* Status bar */
#status-bar {
    background: #eef2ff;
    border-left: 4px solid #4F8EF7;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.9rem;
}

/* Answer box */
#answer-box textarea {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
    background: #f0fdf4 !important;
    border: 2px solid #22c55e !important;
    border-radius: 8px !important;
}

/* Run button */
#run-btn {
    background: linear-gradient(135deg, #4F8EF7, #7C3AED) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 12px 28px !important;
}
#run-btn:hover { opacity: 0.9 !important; }

/* Node pipeline */
#node-md { font-size: 0.85rem; letter-spacing: 0.02em; }

/* Tab styling */
.tab-nav button { font-size: 0.88rem !important; }

/* Progress bar */
#progress-bar .wrap { border-radius: 8px !important; }

/* Tool log table */
.tool-log-table { font-size: 0.82rem; }

/* Confidence badge */
#confidence-md { font-size: 0.9rem; margin-top: 6px; }

/* Example questions */
.example-btn { font-size: 0.82rem !important; }
"""

def create_ui():
    with gr.Blocks(css=CSS, title="GAIA Level 3 Agent", theme=gr.themes.Soft()) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="header-banner">
          <h1>🧠 GAIA Level 3 Agent</h1>
          <p>
            Multi-step reasoning agent powered by <strong>LangGraph</strong> +
            <strong>LangChain</strong> + <strong>OpenAI GPT-4o</strong>.
            Tackles GAIA benchmark Level 3 questions requiring tool use, multi-hop reasoning,
            and cross-domain synthesis.
          </p>
        </div>
        """)

        # ── Config row ─────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    info="Your key stays in your browser session and is never stored.",
                )
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=MODELS,
                    value="gpt-4o",
                    label="🤖 Model",
                    info="GPT-4o recommended for Level 3",
                )
            with gr.Column(scale=1):
                max_iter_slider = gr.Slider(
                    minimum=3, maximum=12, value=8, step=1,
                    label="🔄 Max Iterations",
                    info="More = thorough but slower",
                )

        # ── Question input ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=5):
                question_input = gr.Textbox(
                    label="❓ GAIA Level 3 Question",
                    placeholder="Enter a complex multi-step reasoning question…",
                    lines=3,
                )
            with gr.Column(scale=1, min_width=140):
                run_button = gr.Button("▶ Run Agent", elem_id="run-btn", variant="primary")
                clear_button = gr.Button("🗑 Clear", variant="secondary")

        # ── Example questions ───────────────────────────────────────────────
        with gr.Accordion("📚 Example GAIA Level 3 Questions", open=False):
            gr.Markdown("*Click any question to load it:*")
            for q in EXAMPLE_QUESTIONS:
                btn = gr.Button(q[:110] + ("…" if len(q) > 110 else ""),
                                elem_classes=["example-btn"])
                btn.click(lambda x=q: x, outputs=question_input)

        gr.Markdown("---")

        # ── Live status ─────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                status_md  = gr.Markdown("*Agent idle. Enter a question and click Run.*",
                                         elem_id="status-bar")
            with gr.Column(scale=3):
                node_md = gr.Markdown("", elem_id="node-md")

        progress_bar = gr.Slider(
            minimum=0, maximum=1, value=0, step=0.01,
            label="Progress", interactive=False, elem_id="progress-bar",
        )

        # ── Main tabs ───────────────────────────────────────────────────────
        with gr.Tabs():

            # Tab 1: Final Answer
            with gr.Tab("✅ Answer"):
                answer_box = gr.Textbox(
                    label="Final Answer",
                    lines=4,
                    interactive=False,
                    elem_id="answer-box",
                    placeholder="Answer will appear here after the agent finishes…",
                )
                confidence_md = gr.Markdown("", elem_id="confidence-md")

            # Tab 2: Strategic Plan
            with gr.Tab("🗺️ Plan"):
                plan_md = gr.Markdown("*Plan will appear once the Planner node runs…*")

            # Tab 3: Reasoning trace
            with gr.Tab("🤖 Reasoning Trace"):
                reasoning_md = gr.Markdown("*Reasoning trace will stream here…*")

            # Tab 4: Tool calls log
            with gr.Tab("🔧 Tool Calls"):
                gr.Markdown(
                    "All tool invocations and results are logged here in real-time."
                )
                tool_log_md = gr.Markdown("*No tool calls yet.*", elem_classes=["tool-log-table"])

            # Tab 5: Critique
            with gr.Tab("🔎 Critic"):
                gr.Markdown(
                    "The Critic node evaluates evidence quality and identifies gaps "
                    "before final synthesis."
                )
                critique_md = gr.Markdown("*Critique will appear after the Critic node runs…*")

            # Tab 6: Architecture
            with gr.Tab("📐 Architecture"):
                gr.Markdown(textwrap.dedent("""
                ## LangGraph Agent Architecture

                ```
                ┌────────────┐
                │  Question  │
                └─────┬──────┘
                      │
                      ▼
                ┌────────────┐    Decompose question into sub-tasks,
                │  Planner   │    select tools, build execution plan
                └─────┬──────┘
                      │
                      ▼
                ┌─────────────┐   ReAct loop: reason → select tool
                │    Agent    │◀──── ──────────────────────────────┐
                └──────┬──────┘                                    │
                       │ tool_calls?                               │
                  Yes  │           No                              │
                  ┌────▼────┐   ┌──────────┐                      │
                  │  Tools  │   │  Critic  │                       │
                  └────┬────┘   └────┬─────┘                      │
                       │             │ needs_more_info? ───────────┘
                       └──────┬──────┘
                              │ sufficient
                              ▼
                       ┌─────────────┐   Combine all evidence,
                       │ Synthesiser │   produce precise GAIA answer
                       └─────┬───────┘
                              │
                              ▼
                         Final Answer
                ```

                ## Nodes

                | Node | Role |
                |------|------|
                | **Planner** | Decomposes the question, identifies required tools, creates numbered execution plan |
                | **Agent** | GPT-4o with tools bound via `bind_tools()`. Decides which tool to call next |
                | **Tools** | LangGraph `ToolNode` executing: web_search, wikipedia, calculator, python_repl, reasoning_scratchpad |
                | **Critic** | Evaluates gathered evidence, identifies gaps, requests follow-up if needed |
                | **Synthesiser** | Synthesises all evidence into a single precise GAIA-style answer |

                ## Tools

                | Tool | Purpose |
                |------|---------|
                | 🔍 **web_search** | DuckDuckGo search for current/factual information |
                | 📖 **wikipedia_search** | Encyclopedic knowledge retrieval |
                | 🧮 **calculator** | Safe arithmetic and algebraic evaluation |
                | ⚡ **python_repl** | Sandboxed Python for complex computation |
                | 🧠 **reasoning_scratchpad** | Structured chain-of-thought template |
                """))

        # ── Outputs list (must match _yield tuple order) ────────────────────
        outputs = [
            status_md,
            node_md,
            plan_md,
            reasoning_md,
            tool_log_md,
            critique_md,
            answer_box,
            confidence_md,
            progress_bar,
        ]

        # ── Event handlers ──────────────────────────────────────────────────
        run_button.click(
            fn=run_agent_streaming,
            inputs=[question_input, api_key_input, model_dropdown, max_iter_slider],
            outputs=outputs,
        )

        question_input.submit(
            fn=run_agent_streaming,
            inputs=[question_input, api_key_input, model_dropdown, max_iter_slider],
            outputs=outputs,
        )

        def clear_all():
            return ("", "", "", "", "", "", "", "", 0, "")

        clear_button.click(
            fn=lambda: ("", "*Agent idle.*", "", "*Plan will appear once the Planner node runs…*",
                        "*No tool calls yet.*", "*Critique will appear after the Critic node runs…*",
                        "", "", 0),
            outputs=outputs,
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Read API key from environment (for HF Spaces secrets)
    _env_key = os.environ.get("OPENAI_API_KEY", "")

    demo = create_ui()

    # Pre-fill API key from environment if set
    if _env_key:
        demo.load(
            fn=lambda: _env_key,
            outputs=demo.blocks.get("api_key_input"),
        )

    demo.launch(
        server_name="0.0.0.0",    # required for HF Spaces
        server_port=7860,          # default HF Spaces port
        show_error=True,
        share=False,
    )
