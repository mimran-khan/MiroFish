"""
LLM-driven simulation config generator.

Builds detailed simulation parameters from requirements, documents, and graph
entities. Fully automated defaults with no manual tuning required.

Staged generation (avoids one huge JSON response):
1. Time configuration
2. Event configuration
3. Agent configs in batches
4. Platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ..config import Config
from ..utils.logger import get_logger
from ..utils.vertex_ai import create_openai_client
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# Beijing (China Standard Time) diurnal pattern reference
CHINA_TIMEZONE_CONFIG = {
    "dead_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "peak_hours": [19, 20, 21, 22],
    "night_hours": [23],
    "activity_multipliers": {
        "dead": 0.05,
        "morning": 0.4,
        "work": 0.7,
        "peak": 1.5,
        "night": 0.5,
    },
}


@dataclass
class AgentActivityConfig:
    """Per-agent activity parameters."""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    activity_level: float = 0.5
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    response_delay_min: int = 5
    response_delay_max: int = 60
    sentiment_bias: float = 0.0
    stance: str = "neutral"
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Clock and cadence settings (default: Beijing-style diurnal curve)."""
    total_simulation_hours: int = 72
    minutes_per_round: int = 60
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    work_hours: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Narrative hooks and seed content."""
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    hot_topics: List[str] = field(default_factory=list)
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Per-platform feed tuning."""
    platform: str
    recency_weight: float = 0.4
    popularity_weight: float = 0.3
    relevance_weight: float = 0.3
    viral_threshold: int = 10
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Full parameter bundle for one simulation run."""
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    event_config: EventConfig = field(default_factory=EventConfig)
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    llm_model: str = ""
    llm_base_url: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """JSON string for persistence."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    LLM-backed generator for SimulationParameters.

    Stages:
    1. Time + event configs (small JSON)
    2. Agent configs in batches (~15 entities)
    3. Platform configs (rule-based from flags)
    """

    MAX_CONTEXT_LENGTH = 50000
    AGENTS_PER_BATCH = 15
    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.base_url = base_url or Config.LLM_BASE_URL
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Generate SimulationParameters end-to-end.

        Args:
            simulation_id / project_id / graph_id: workspace ids
            simulation_requirement: natural-language scenario brief
            document_text: optional source document
            entities: filtered EntityNode list
            enable_twitter / enable_reddit: platform toggles
            progress_callback: optional (step, total_steps, message)

        Returns:
            SimulationParameters
        """
        logger.info(
            f"Generating simulation config: simulation_id={simulation_id}, entities={len(entities)}"
        )

        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        report_progress(1, "Generating time configuration...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"time: {time_config_result.get('reasoning', 'ok')}")

        report_progress(2, "Generating event config and hot topics...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"events: {event_config_result.get('reasoning', 'ok')}")

        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                f"Generating agent configs ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"agents: generated {len(all_agent_configs)}")

        logger.info("Assigning poster agents for initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"initial_posts: assigned posters for {assigned_count} posts")

        report_progress(total_steps, "Generating platform configuration...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Simulation config ready: {len(params.agent_configs)} agent rows")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Assemble LLM context with hard caps."""

        entity_summary = self._summarize_entities(entities)

        context_parts = [
            f"## Simulation requirement\n{simulation_requirement}",
            f"\n## Entities ({len(entities)})\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n... (document truncated)"
            context_parts.append(f"\n## Source document\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Grouped textual summary of entities."""
        lines = []

        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")

            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retries and JSON repair."""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                if finish_reason == "length":
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(e)[:80]}")

                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM call failed")

    def _fix_truncated_json(self, content: str) -> str:
        """Heuristically close truncated JSON."""
        content = content.strip()

        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        if content and content[-1] not in '",}]':
            content += '"'
        
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON salvage for config payloads."""
        import re

        content = self._fix_truncated_json(content)

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """LLM step: temporal cadence JSON."""
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Using the simulation brief below, produce a JSON time configuration.

{context_truncated}

## Task
Return ONLY JSON (no markdown fences).

### Diurnal guidance (Beijing / China Standard Time style — adapt to the scenario):
- Population skews toward CST-like rhythms unless the brief says otherwise.
- 00:00-05:00: very quiet (activity multiplier ~0.05)
- 06:00-08:00: ramping up (~0.4)
- 09:00-18:00: moderate (~0.7)
- 19:00-22:00: peak (~1.5)
- 23:00: taper (~0.5)
- **Important:** tune hour lists to the event and cohorts (students peak 21-23; newsrooms nearly 24/7; government mostly 9-17; breaking news may shorten off-peak windows).

### Example shape
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "short rationale"
}}

### Fields
- total_simulation_hours (int): 24-168 (short crises vs long arcs)
- minutes_per_round (int): 30-120 (60 recommended)
- agents_per_hour_min / max (int): each in [1, {max_agents_allowed}]
- peak_hours / off_peak_hours / morning_hours / work_hours: int hour lists (0-23)
- reasoning (string): why these choices fit the brief"""

        system_prompt = (
            "You are a social simulation planner. Output pure JSON. "
            "Honor Beijing-style diurnal curves unless the brief specifies another timezone."
        )

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM failed: {e}; using defaults")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Fallback CST-shaped schedule."""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Default CST-style cadence, 60 simulated minutes per round",
        }

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Hydrate TimeSimulationConfig with sanity clamps vs entity count."""
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))

        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) > entity count ({num_entities}); clamping"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) > entity count ({num_entities}); clamping"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max; adjusted min to {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """LLM step: narrative + seed posts."""

        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""From the simulation brief, emit an event configuration JSON.

Requirement: {simulation_requirement}

{context_truncated}

## Available entity types (with examples)
{type_info}

## Deliverables
- hot topic keywords
- narrative_direction string
- initial_posts: each object MUST include poster_type taken **verbatim** from the available types so we can route posts to agents.

Examples:
- Official statements -> University / GovernmentAgency types
- News -> MediaOutlet
- Student takes -> Student

Return JSON only (no markdown):
{{
    "hot_topics": ["topic_a", "topic_b"],
    "narrative_direction": "how discourse should evolve",
    "initial_posts": [
        {{"content": "post text", "poster_type": "EntityTypeFromList"}},
        ...
    ],
    "reasoning": "short note"
}}"""

        system_prompt = (
            "You are a narrative designer for opinion simulations. "
            "Return pure JSON; poster_type must exactly match a provided entity type."
        )

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM failed: {e}; using empty defaults")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "default (LLM failure)",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Materialize EventConfig dataclass."""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Map each initial post's poster_type to a concrete poster_agent_id.
        """
        if not event_config.initial_posts:
            return event_config

        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            matched_agent_id = None

            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            if matched_agent_id is None:
                logger.warning(
                    f"No agent type match for poster_type='{poster_type}'; using highest influence agent"
                )
                if agent_configs:
                    sorted_agents = sorted(
                        agent_configs, key=lambda a: a.influence_weight, reverse=True
                    )
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"Initial post routing: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """LLM batch step for agent activity rows."""

        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""For every entity below, output social-activity parameters as JSON.

Requirement: {simulation_requirement}

## Entities
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Guidance (Beijing-style dayparts unless brief overrides)
- Dead night 0-5: almost inactive; peak engagement 19-22.
- Official orgs (University/GovernmentAgency): low activity 0.1-0.3, mostly 9-17, slow responses 60-240m, high influence 2.5-3.0.
- MediaOutlet: medium 0.4-0.6, long hours ~8-23, fast response 5-30m, influence 2.0-2.5.
- Individuals (Student/Person/Alumni): high 0.6-0.9, evenings 18-23, fast 1-15m, lower influence 0.8-1.2.
- Public figures / experts: medium activity, influence 1.5-2.0.

Return JSON only:
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <float>,
            "comments_per_hour": <float>,
            "active_hours": [<ints 0-23>],
            "response_delay_min": <int minutes>,
            "response_delay_max": <int minutes>,
            "sentiment_bias": <-1.0..1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <float>
        }}
    ]
}}"""

        system_prompt = (
            "You model social-media behavior for simulations. "
            "Return pure JSON aligned with Beijing diurnal patterns unless stated otherwise."
        )

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent batch LLM failed: {e}; using rule-based fill")
            llm_configs = {}

        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Heuristic agent row when LLM output is missing."""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

