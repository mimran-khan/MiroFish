"""
Ontology generation service.
Endpoint 1: analyze text and emit entity/relation types for social simulation.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """Convert any name format to PascalCase (e.g. 'works_for' -> 'WorksFor', 'person' -> 'Person')."""
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge-graph ontology designer. Given document text and a simulation brief, you define **entity types** and **edge types** suited to **social-media opinion simulation**.

**You must output valid JSON only. No prose outside the JSON.**

## Core context

We are building a **social-media opinion simulation**. In that world:
- Each entity is an "account" or actor that can post, reply, share, and be reached on social platforms.
- Entities influence one another through follows, comments, reshares, and reactions.
- We need to model who speaks, who amplifies whom, and how narratives spread.

Therefore **entities must be real-world actors that can plausibly have a social presence**:

**Allowed examples**:
- Specific people (public figures, parties to an event, influencers, experts, ordinary citizens)
- Companies and brands (including official accounts)
- Organizations (universities, NGOs, unions, associations)
- Government and regulators
- Media outlets (newspapers, TV, blogs, sites)
- Platforms themselves, when relevant
- Group representatives (alumni associations, fan groups, advocacy groups)

**Not allowed as entity types**:
- Pure abstractions ("public opinion", "emotion", "trend")
- Topics/themes as types ("academic integrity", "policy reform")
- Stances as types ("pro side", "anti side")

## Output shape

Return JSON:

```json
{
    "entity_types": [
        {
            "name": "TypeNameInEnglishPascalCase",
            "description": "Short English description, max ~100 chars",
            "attributes": [
                {
                    "name": "attr_name_snake_case",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["example entity 1", "example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATION_NAME_UPPER_SNAKE",
            "description": "Short English description, max ~100 chars",
            "source_targets": [
                {"source": "SourceEntityType", "target": "TargetEntityType"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief English summary of what you inferred from the text"
}
```

## Design rules (critical)

### 1. Entity types — strict

**Count: exactly 10 entity types.**

**Layering — you must mix specific types with catch-alls:**

A. **Catch-all types (must be the last two in the list)**:
   - `Person`: default for any natural person who does not fit a more specific person type.
   - `Organization`: default for any org that does not fit a more specific org type.

B. **Specific types (8 types, derived from the text)**:
   - Reflect dominant roles in the scenario (e.g. academic story → `Student`, `Professor`, `University`).
   - Business story → e.g. `Company`, `CEO`, `Employee`.

**Why catch-alls**: texts mention many individuals (teachers, passers-by, anonymous users) who should map to `Person`, and informal groups to `Organization`.

**Specific type rules**:
- Draw from frequent or pivotal roles in the text.
- Keep boundaries crisp; avoid overlapping definitions.
- Each `description` must explain how it differs from the catch-all.

### 2. Edge types

- Count: between 6 and 10.
- Edges should mirror plausible social ties and interactions.
- Every `source_targets` pair should use types you actually defined.

### 3. Attributes

- 1–3 key attributes per entity type.
- **Do not** use reserved names: `name`, `uuid`, `group_id`, `created_at`, `summary`.
- Prefer `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Entity type cheat sheet

**Specific people**:
- Student, Professor, Journalist, Celebrity, Executive, Official, Lawyer, Doctor

**Catch-all person**:
- Person

**Specific orgs**:
- University, Company, GovernmentAgency, MediaOutlet, Hospital, School, NGO

**Catch-all org**:
- Organization

## Edge type cheat sheet (English labels; meanings in parentheses)

- WORKS_FOR (employed by)
- STUDIES_AT (enrolled at)
- AFFILIATED_WITH (affiliated with)
- REPRESENTS (represents)
- REGULATES (regulates)
- REPORTS_ON (covers as media)
- COMMENTS_ON (comments on)
- RESPONDS_TO (responds to)
- SUPPORTS (supports)
- OPPOSES (opposes)
- COLLABORATES_WITH (collaborates with)
- COMPETES_WITH (competes with)
"""


class OntologyGenerator:
    """
    Calls the LLM to produce ontology JSON (entity_types, edge_types).
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology dict.

        Args:
            document_texts: raw document strings
            simulation_requirement: scenario brief for the simulation
            additional_context: optional extra instructions

        Returns:
            Dict with entity_types, edge_types, analysis_summary, etc.
        """
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        result = self._validate_and_process(result)
        
        return result
    
    # Max characters forwarded to the LLM (~50k CJK or Latin chars)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Assemble the user prompt."""
        
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += (
                f"\n\n...(source had {original_length} characters; "
                f"only the first {self.MAX_TEXT_LENGTH_FOR_LLM} were sent for ontology analysis)..."
            )
        
        message = f"""## Simulation requirement

{simulation_requirement}

## Document content

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Additional notes

{additional_context}
"""
        
        message += """
From the above, design entity and relation types for social-opinion simulation.

**Hard requirements**:
1. Output exactly 10 entity types.
2. The last two must be the catch-alls `Person` and `Organization` (in that order at the end).
3. The first eight are scenario-specific types grounded in the text.
4. Every entity type must be a real actor that could post or be referenced online—no pure abstractions.
5. Never use reserved attribute names (`name`, `uuid`, `group_id`, …); use `full_name`, `org_name`, etc.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize fields and enforce Zep limits."""
        
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        entity_name_map = {}
        for entity in result["entity_types"]:
            # Force entity name to PascalCase (Zep API requirement)
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        for edge in result["edge_types"]:
            # Force edge name to SCREAMING_SNAKE_CASE (Zep API requirement)
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # Fix entity name references in source_targets to match PascalCase conversion
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            result["entity_types"].extend(fallbacks_to_add)
        
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Serialize ontology to a Python module string (Zep EntityModel/EdgeModel stubs).
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity type definitions ==============',
            '',
        ]
        
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Edge type definitions ==============')
        code_lines.append('')
        
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Type registries ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)
