"""
Report Agent service.

Uses LangChain-style orchestration plus Zep tools in a ReACT loop to draft
simulation-grounded reports.

Capabilities:
1. Build reports from the simulation brief and graph context
2. Plan an outline first, then draft section by section
3. Each section runs multi-turn ReACT (reason + act + observe)
4. Chat follow-ups can invoke retrieval tools on demand
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Structured JSONL logger for report generation.

    Writes `agent_log.jsonl` under the report folder. Each line is one JSON object
    with timestamps, action types, and full detail payloads (no truncation).
    """
    
    def __init__(self, report_id: str):
        """
        Args:
            report_id: Report folder identifier (drives log path).
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Create parent directories for the JSONL log."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Seconds elapsed since logger construction."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Append one structured log entry.

        Args:
            action: Event name (e.g. start, tool_call, llm_response, section_complete).
            stage: Pipeline stage (planning, generating, completed, ...).
            details: Arbitrary dict stored verbatim.
            section_title: Optional section heading for context.
            section_index: Optional 1-based section index.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append JSONL row
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation kickoff."""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation started"
            }
        )
    
    def log_planning_start(self):
        """Log outline planning start."""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Outline planning started"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log graph/context snapshot used for planning."""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Fetched simulation context for planning",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log finalized outline JSON."""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log section drafting start."""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Started section: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log one ReACT reasoning turn."""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT iteration {iteration} thought"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log a tool invocation."""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Tool call: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log full tool output (no truncation)."""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # full payload
                "result_length": len(result),
                "message": f"Tool {tool_name} returned"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log raw LLM completion (no truncation)."""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # full text
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": (
                    f"LLM response (tool_calls={has_tool_calls}, "
                    f"final_answer={has_final_answer})"
                )
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log drafted section body (before persistence hooks)."""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # full markdown body
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} draft captured"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log when a section file is finalized.

        UIs can treat this as the authoritative completion signal with full text.
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} complete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log successful end-to-end report run."""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log a failure with context."""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Mirrors console-style logs into `console_log.txt` under the report folder.

    Complements `agent_log.jsonl` with human-readable INFO/WARNING lines.
    """
    
    def __init__(self, report_id: str):
        """
        Args:
            report_id: Report folder identifier (drives log path).
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Create parent directories for the console log."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Attach a FileHandler to the relevant loggers."""
        import logging
        
        # File sink
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Match console formatting
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Mirror into report_agent + zep_tools namespaces
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Detach and close the file handler."""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Best-effort cleanup when the logger drops out of scope."""
        self.close()


class ReportStatus(str, Enum):
    """Lifecycle flags persisted with each report."""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """One outline section plus drafted body."""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Render as a Markdown heading + body."""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Title, summary, and ordered sections."""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Render the full outline as Markdown."""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Persisted report aggregate (metadata + optional outline + markdown)."""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# -- Tool blurbs (injected into LLM prompts) --

TOOL_DESC_INSIGHT_FORGE = """\
InsightForge — deep hybrid retrieval
- Splits your question into targeted sub-queries
- Mines the simulation graph from multiple angles
- Merges semantic hits, entity dossiers, and relationship chains
- Returns the richest bundle of quotable evidence

When to use
- You need multi-faceted analysis
- You want material to anchor an entire section

Outputs
- Verbatim facts
- Entity-level takeaways
- Relationship chains"""

TOOL_DESC_PANORAMA_SEARCH = """\
PanoramaSearch — whole-graph snapshot
- Pulls every relevant node and edge
- Separates still-valid facts from historical / superseded ones
- Ideal for timeline or evolution narratives

When to use
- You need the full arc of the story
- You want to contrast phases of the simulation

Outputs
- Active facts (latest simulated state)
- Historical facts (how beliefs shifted)
- Entities touched along the way"""

TOOL_DESC_QUICK_SEARCH = """\
QuickSearch — fast edge search
Lightweight lookup for a focused fact check.

When to use
- You only need a needle, not the haystack

Outputs
- Ranked fact snippets"""

TOOL_DESC_INTERVIEW_AGENTS = """\
InterviewAgents — live OASIS interviews (Twitter + Reddit)
Calls the real interview API against running agents (not an LLM role-play).
By default both platforms run so you get contrasting voices.

Flow
1. Load persona files
2. LLM picks the most relevant agents
3. LLM drafts questions
4. POST /api/simulation/interview/batch interviews them in parallel
5. Responses are merged for multi-perspective reporting

When to use
- You need first-person quotes from student/media/official personas
- You want grounded "transcript" material

Outputs
- Agent identity context
- Dual-platform answers
- Pull quotes + comparative summary

Requires the OASIS environment to stay up."""

# -- Outline planning prompts --

PLAN_SYSTEM_PROMPT = """\
You are the lead author of a future-facing simulation report with full observability
into every agent utterance and interaction inside the run.

Core idea
We inject a simulation brief as the forcing function; whatever emerges inside the run
is a forecast of how the situation could evolve. Treat it as a rehearsal of the future,
not a recap of today's headlines.

Your mission
Produce an outline that answers:
1. Under the stated conditions, what happens next?
2. How do different agent cohorts react?
3. Which emergent trends or risks deserve attention?

Positioning
- This is scenario forecasting grounded in the simulation
- Highlight trajectories, collective behavior, emergent phenomena, downside risk
- Agent dialogue is stand-in evidence for future public behavior
- Avoid generic "state of the world" punditry unrelated to the run

Chapter budget
- Minimum 2 sections, maximum 5
- No nested sub-outline; each section will be drafted wholesale
- Stay tight and insight-led

Return JSON ONLY:
{
    "title": "...",
    "summary": "one-sentence headline insight",
    "sections": [
        {"title": "...", "description": "..."}
    ]
}

sections must contain between 2 and 5 objects."""

PLAN_USER_PROMPT_TEMPLATE = """\
Scenario brief injected into the simulation:
{simulation_requirement}

World scale
- Nodes: {total_nodes}
- Edges: {total_edges}
- Label mix: {entity_types}
- Active agents: {total_entities}

Sample forecasted facts (truncated JSON):
{related_facts_json}

Take an omniscient view of this rehearsal:
1. What state does the world settle into?
2. How do populations move and respond?
3. Which future-facing trends matter?

Design the best chapter plan.

Reminder: 2–5 sections, each sharply focused on predictive insight."""

# -- Section drafting prompt --

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are writing one chapter of a future-oriented simulation report.

Report title: {report_title}
Report summary: {report_summary}
Scenario (simulation brief): {simulation_requirement}

Chapter to write now: {section_title}

================================================================
Core idea
================================================================

The simulation is a rehearsal of the future. The brief injects conditions; agent
behavior is a forecast of how populations may react.

Your job:
- Describe what unfolds under those conditions
- Explain how different cohorts (agents) behave
- Surface notable trends, risks, and opportunities

Do not recast this as a static analysis of the real world today.
Do focus on "what the simulation says happens next."

================================================================
Non-negotiable rules
================================================================

1. Tool-grounded writing
   - You are observing the simulated world, not inventing facts.
   - Every claim must come from simulated events or agent utterances surfaced by tools.
   - Never substitute your own general knowledge for missing simulation evidence.
   - Each chapter needs at least 3 tool calls (max 5). The graph is your oracle.

2. Quote the simulation
   - Agent speech and actions are predictive evidence.
   - Use Markdown blockquotes, e.g.:
     > "How a cohort might say: <verbatim or lightly edited excerpt>..."
   - Quotes are the backbone of the chapter.

3. Language alignment
   - Tool payloads may mix languages.
   - Write the chapter in fluent English unless the simulation brief explicitly demands
     another language; translate quoted material into that chapter language while
     preserving meaning. Applies to body text and quoted lines.

4. Faithfulness
   - Reflect what the simulation actually produced.
   - Do not fabricate events that never appeared.
   - If evidence is thin, say so transparently.

================================================================
Formatting (critical)
================================================================

One chapter = one atomic section
- No Markdown headings inside the chapter (#, ##, ###, ####).
- Do not repeat the chapter title; the UI injects it.
- Use **bold**, paragraphs, quotes, and lists instead of headings.

Good pattern
```
This chapter traces how discourse spreads inside the simulation.

**Initial ignition**

The first wave of posts concentrated on ...

> "Platform X carried most of the opening salience ..."

**Amplification**

Later spikes show ...
```

Bad pattern
```
## Executive summary   ← forbidden
### Phase one          ← forbidden
```

================================================================
Available tools (use 3–5 distinct calls per chapter)
================================================================

{tools_description}

Mix tools—do not hammer only one:
- insight_forge: deep, multi-angle retrieval with auto sub-queries
- panorama_search: whole-graph view, including stale facts for timelines
- quick_search: spot-check a fact quickly
- interview_agents: first-person answers from live simulated agents

================================================================
Turn protocol
================================================================

Each assistant turn must be exactly one of:

Option A — tool call
Share your reasoning, then emit:
<tool_call>
{{"name": "<tool_name>", "parameters": {{"<param>": "<value>"}}}}
</tool_call>
The runtime executes the tool and returns observations. Never fake observations.

Option B — final prose
After enough evidence, output chapter text prefixed with `Final Answer:`.

Hard stops
- Never mix a tool call and Final Answer in the same turn.
- Never invent tool output.
- At most one tool call per turn.

================================================================
Chapter expectations
================================================================

1. Ground every paragraph in retrieved simulation evidence.
2. Quote generously to show what the run actually produced.
3. Markdown is allowed except headings:
   - **Bold** for emphasis instead of subheads
   - Lists (- or 1.) for bullets
   - Blank lines between ideas
4. Quote formatting
   Quotes must stand alone with blank lines before and after:

   Good:
   ```
   Observers felt the institution moved slowly.

   > "Their playbook looked rigid compared to the pace online."

   That sentiment echoed widely.
   ```

   Bad:
   ```
   Observers felt ... > "Their playbook ..." which shows ...
   ```
5. Stay coherent with earlier chapters (see user message context).
6. Avoid repeating facts already covered below.
7. Again: no headings—use **bold** mini labels if needed."""


SECTION_USER_PROMPT_TEMPLATE = """\
Already drafted sections (read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
Current task: write section "{section_title}"
═══════════════════════════════════════════════════════════════

Reminders
1. Do not repeat facts already covered above.
2. Call tools before drafting prose.
3. Mix tools; do not lean on a single one.
4. Every claim must trace to retrieved simulation evidence.

Formatting guardrails
- No Markdown headings (# through ####)
- Do not open with "{section_title}" — the UI injects the title
- Use **bold** mini labels instead of headings

Workflow
1. Think through what evidence you still need
2. Issue exactly one tool call per turn until satisfied
3. Finish with `Final Answer:` followed by body text only"""

# -- ReACT loop scaffolding --

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval payload):

═══ {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tool calls used: {tool_calls_count}/{max_tool_calls} (so far: {used_tools_str}){unused_hint}
- If coverage is enough: start with "Final Answer:" and cite the evidence above
- Otherwise: call one more tool
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "Heads-up: only {tool_calls_count} tool call(s) so far; you need at least {min_tool_calls}."
    "Keep retrieving simulation evidence before emitting Final Answer.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "You have used {tool_calls_count} tool call(s) but need at least {min_tool_calls}."
    "Please call a tool to pull more simulation data.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool budget exhausted ({tool_calls_count}/{max_tool_calls}); no further calls allowed."
    'Immediately respond with "Final Answer:" using only what you already retrieved.'
)

REACT_UNUSED_TOOLS_HINT = "\nTip: unused tools: {unused_list}. Rotate tools for broader coverage."

REACT_FORCE_FINAL_MSG = "Tool limit reached — output Final Answer: now with the evidence on hand."

# -- Chat prompt --

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise simulation Q&A copilot.

Context
Scenario brief: {simulation_requirement}

Rendered report (may be partial):
{report_content}

Rules
1. Prefer answering straight from the report
2. Be direct; skip long chain-of-thought narration
3. Call tools only when the report is silent on the question (max 1–2 calls)
4. Keep answers tight and structured

Available tools (optional)
{tools_description}

Tool syntax
<tool_call>
{{"name": "<tool>", "parameters": {{"<param>": "<value>"}}}}
</tool_call>

Voice
- Lead with the takeaway, then justify
- Quote with Markdown `>` when it helps
- Stay brief"""

CHAT_OBSERVATION_SUFFIX = "\n\nAnswer succinctly."


# ═══════════════════════════════════════════════════════════════
# ReportAgent class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Orchestrates future-facing simulation reports.

    ReACT loop:
    1. Planning — interpret the brief and outline sections
    2. Generation — draft each section with iterative tool use
    3. Reflection — sanity-check coverage and fidelity
    """
    
    # Max tool invocations per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Max reflection rounds
    MAX_REFLECTION_ROUNDS = 3
    
    # Max tool invocations during chat follow-ups
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Args:
            graph_id: Zep graph identifier
            simulation_id: Active simulation identifier
            simulation_requirement: Natural-language simulation brief
            llm_client: Optional injected LLM client
            zep_tools: Optional injected Zep tool service
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool registry exposed to the LLM
        self.tools = self._define_tools()
        
        # Populated inside generate_report
        self.report_logger: Optional[ReportLogger] = None
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent ready graph_id={graph_id} simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Describe callable tools for prompt injection."""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Question or topic to analyze deeply",
                    "report_context": "Optional section context to sharpen sub-queries",
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Ranking query for relevance",
                    "include_expired": "Include historical/superseded facts (default True)",
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search string",
                    "limit": "Max hits (optional, default 10)",
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": (
                        "Interview brief, e.g. 'Ask students how they view the dorm incident'"
                    ),
                    "max_agents": "Cap on interviewed agents (optional, default 5, max 10)",
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Dispatch to ZepToolsService (or legacy aliases).

        Args:
            tool_name: Registered tool id
            parameters: Parsed JSON parameters
            report_context: Extra narrative context for InsightForge

        Returns:
            Human-readable tool payload for the LLM observation.
        """
        logger.info(f"Executing tool {tool_name} params={parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Panorama snapshot
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Lightweight edge search
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Live OASIS interviews (multi-platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # Legacy aliases (internal redirects)
            
            elif tool_name == "search_graph":
                logger.info("search_graph alias -> quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                logger.info("get_simulation_context alias -> insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return (
                    f"Unknown tool: {tool_name}. "
                    "Use one of: insight_forge, panorama_search, quick_search, interview_agents"
                )
                
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Tool execution failed: {e}"
    
    # Canonical names validated when parsing bare JSON tool calls
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from assistant text.

        Priority:
        1. `<tool_call>{...}</tool_call>`
        2. Bare JSON objects (whole message or trailing object)
        """
        tool_calls = []

        # Preferred: tagged JSON
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Fallback: bare JSON (only if no tagged calls) to avoid hijacking prose JSON
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Trailing JSON after chain-of-thought text
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Normalize legacy keys and ensure the tool name is allowlisted."""
        # Accept either {name, parameters} or {tool, params}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize to name/parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Flatten tool metadata for system prompts."""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Ask the LLM for a JSON outline grounded in graph stats + sample facts.

        Args:
            progress_callback: Optional UI hook (stage, percent, message).

        Returns:
            Parsed `ReportOutline`, or a deterministic fallback on failure.
        """
        logger.info("Planning report outline...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation brief...")
        
        # Pull lightweight graph context first
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Drafting outline with LLM...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline JSON...")
            
            # Hydrate dataclasses
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation analysis report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline ready")
            
            logger.info(f"Outline ready with {len(sections)} sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planning failed: {e}")
            # Deterministic fallback (three sections)
            return ReportOutline(
                title="Future outlook report",
                summary="Forecasted trends and risks inferred from the simulation",
                sections=[
                    ReportSection(title="Scenario setup and headline findings"),
                    ReportSection(title="Population behavior outlook"),
                    ReportSection(title="Trend watch and risk signals"),
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Draft one section via iterative tool use + Final Answer.

        Loop:
        1. Reason about missing evidence
        2. Issue a single tool call
        3. Observe payload
        4. Repeat until coverage is sufficient or budgets exhaust
        5. Emit `Final Answer:` prose (no extra headings)

        Args:
            section: Target outline row
            outline: Full outline for system context
            previous_sections: Earlier bodies for de-duplication
            progress_callback: Optional UI hook
            section_index: 1-based index for logging

        Returns:
            Markdown body (headings stripped downstream)
        """
        logger.info(f"ReACT drafting section: {section.title}")
        
        # Structured log hook
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # User prompt includes up to 4000 chars per prior section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section.)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tool_calls_count = 0
        max_iterations = 5
        min_tool_calls = 3
        conflict_retries = 0  # consecutive turns mixing tool calls + Final Answer
        used_tools = set()
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Extra narrative for InsightForge sub-query synthesis
        report_context = f"Section title: {section.title}\nSimulation brief: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval & drafting ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # LLM turn
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            if response is None:
                logger.warning(
                    f"Section {section.title} iteration {iteration + 1}: LLM returned None"
                )
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please continue."})
                    continue
                break

            logger.debug(f"LLM preview: {response[:200]}...")

            # Parse tool calls once per turn
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} iteration {iteration+1}: "
                    f"tool call + Final Answer in one turn (strike {conflict_retries})"
                )

                if conflict_retries <= 2:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format error] A single assistant turn cannot mix <tool_call> "
                            "and Final Answer.\n"
                            "Choose exactly one per turn:\n"
                            "- Issue one <tool_call>...</tool_call> without Final Answer\n"
                            "- Or answer with prose starting `Final Answer:` and no tool tags\n"
                            "Please try again."
                        ),
                    })
                    continue
                else:
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} conflicts; "
                        "truncating to first tool call"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Structured LLM log
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            if has_final_answer:
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f" (Unused tools worth trying: {', '.join(sorted(unused_tools))})"
                        if unused_tools
                        else ""
                    )
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Section {section.title} done after {tool_calls_count} tool calls")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            if has_tool_calls:
                # Budget exhausted — force prose
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM issued {len(tool_calls)} calls; executing first: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(
                        unused_list=", ".join(sorted(unused_tools))
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f" (Unused tools worth trying: {', '.join(sorted(unused_tools))})"
                    if unused_tools
                    else ""
                )

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            logger.info(
                f"Section {section.title}: no Final Answer prefix after sufficient tools "
                f"({tool_calls_count}); accepting raw assistant text"
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        logger.warning(f"Section {section.title}: max iterations reached; forcing final pass")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        if response is None:
            logger.error(f"Section {section.title}: forced final pass returned None")
            final_answer = (
                "(This section failed: the model returned an empty response. Please retry.)"
            )
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Log whatever we salvaged
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        End-to-end pipeline with per-section persistence.

        Each section is written to disk as soon as it finishes so UIs can stream progress.

        Folder layout:
            reports/{report_id}/
                meta.json
                outline.json
                progress.json
                section_01.md
                section_02.md
                ...
                full_report.md

        Args:
            progress_callback: Optional (stage, percent, message) hook
            report_id: Optional stable id (UUID suffix if omitted)

        Returns:
            Hydrated `Report` object (even on failure, status reflects error)
        """
        import uuid
        
        # Auto id when omitted
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        completed_section_titles = []
        
        try:
            ReportManager._ensure_report_folder(report_id)
            
            # Structured JSONL logger
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report workspace...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Planning outline...",
                completed_sections=[]
            )
            
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Planning outline...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            self.report_logger.log_planning_complete(outline.to_dict())
            
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "planning",
                15,
                f"Outline ready ({len(outline.sections)} sections)",
                completed_sections=[],
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline persisted {report_id}/outline.json")
            
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Drafting section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Drafting section: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Section body via ReACT
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Final structured log for this section
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section saved {report_id}/section_{section_num:02d}.md")
                
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Section complete: {section.title}",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            if progress_callback:
                progress_callback("generating", 95, "Assembling full Markdown...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling full Markdown...",
                completed_sections=completed_section_titles
            )
            
            # Stitch section files + post-process headings
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Success log
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Report generation complete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")
            
            logger.info(f"Report finished {report_id}")
            
            # Tear down console mirror
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report generation failed: {e}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # swallow secondary persistence errors
            
            # Tear down console mirror
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Lightweight Q&A over an existing report with optional tool use.

        Args:
            message: Latest user utterance
            chat_history: Prior turns (role/content dicts)

        Returns:
            Dict with assistant text, executed tool payloads, and source queries.
        """
        logger.info(f"Report chat: {message[:50]}...")
        
        chat_history = chat_history or []
        
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [report truncated] ..."
        except Exception as e:
            logger.warning(f"Could not load cached report markdown: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report yet)",
            tools_description=self._get_tools_description(),
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        for h in chat_history[-10:]:
            messages.append(h)
        
        # Latest user turn
        messages.append({
            "role": "user", 
            "content": message
        })
        
        tool_calls_made = []
        max_iterations = 2
        
        for _ in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Plain assistant answer
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            tool_results = []
            for call in tool_calls[:1]:
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]
                })
                tool_calls_made.append(call)
            
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} output]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Filesystem persistence for streaming report generation.

    Layout:
        reports/{report_id}/
            meta.json
            outline.json
            progress.json
            section_XX.md
            full_report.md
    """
    
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Create the shared reports root."""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Absolute path to a report workspace."""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """mkdir -p equivalent for a report folder."""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Path to meta.json."""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Path to stitched full_report.md."""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Path to outline.json."""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Path to progress.json."""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Path to a numbered section markdown file."""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Path to agent_log.jsonl."""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Path to console_log.txt."""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Return plain-text console lines mirrored during generation.

        Differs from structured `agent_log.jsonl`.

        Args:
            report_id: Workspace id
            from_line: Zero-based line offset for incremental reads

        Returns:
            Dict with `logs`, `total_lines`, `from_line`, `has_more` (always False here).
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """Convenience helper that returns every console line."""
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Stream structured JSONL entries written by `ReportLogger`.

        Args:
            report_id: Workspace id
            from_line: Zero-based line offset for incremental reads

        Returns:
            Dict with parsed `logs`, counts, and `has_more` (False — full read).
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """Return every parsed JSON object from agent_log.jsonl."""
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """Persist outline.json right after planning."""
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Outline saved {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Write `section_XX.md` as soon as a section finishes.

        Args:
            report_id: Workspace id
            section_index: 1-based index matching filename
            section: Populated dataclass

        Returns:
            Absolute path written
        """
        cls._ensure_report_folder(report_id)

        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section saved {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Normalize LLM output before persisting a section file.

        1. Drop leading headings that duplicate the injected section title
        2. Convert every Markdown heading into bold pseudo-headings

        Args:
            content: Raw model output
            section_title: Outline title for dedupe heuristics

        Returns:
            Cleaned markdown body
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                title_text = heading_match.group(2).strip()
                
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")
                continue
            
            # Skip blank line following a removed duplicate heading
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """Write `progress.json` for UI polling."""
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Load progress.json if it exists."""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """List persisted `section_XX.md` files with contents."""
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Concatenate section files, post-process headings, write full_report.md.

        Returns the stitched markdown string.
        """
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += "---\n\n"
        
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        md_content = cls._post_process_report(md_content, outline)
        
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Full report assembled {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Normalize stitched markdown before shipping to clients.

        1. Drop duplicate headings within a short window
        2. Keep `#` report title and legitimate `##` section headings
        3. Demote deeper headings to bold + blank line hygiene
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                if level == 1:
                    if title == outline.title:
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Persist meta.json, optional outline refresh, and full_report.md."""
        cls._ensure_report_folder(report.report_id)
        
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Report saved {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Hydrate `Report` from disk (supports legacy flat JSON)."""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Scan workspaces for a matching simulation id."""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Return recent reports, optionally filtered by simulation id."""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Remove a workspace directory or legacy flat files."""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Report folder removed {report_id}")
            return True
        
        # Legacy single-file layout
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
