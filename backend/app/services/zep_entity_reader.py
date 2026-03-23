"""
Zep entity reader and filter.
Loads graph nodes and keeps only those tagged with ontology entity labels.
"""

import json
import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_entity_reader')

_CLASSIFY_BATCH_SIZE = 30

# Generic helper type var
T = TypeVar('T')


@dataclass
class EntityNode:
    """In-memory entity with optional neighborhood context."""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }
    
    def get_entity_type(self) -> Optional[str]:
        """First non-default ontology label, if any."""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Result of filter_defined_entities."""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    Reads Zep graphs, filters to ontology-tagged entities, optionally enriches edges.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")
        
        self.client = Zep(api_key=self.api_key)
    
    def _call_with_retry(
        self, 
        func: Callable[[], T], 
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> T:
        """Run a Zep call with exponential backoff."""
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} attempt {attempt + 1} failed: {str(e)[:100]}, "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} failed after {max_retries} attempts: {str(e)}")
        
        raise last_exception
    
    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Return every node in the graph (paged internally)."""
        logger.info(f"Fetching all nodes for graph {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                "name": node.name or "",
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
            })

        logger.info(f"Loaded {len(nodes_data)} nodes")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Return every edge in the graph."""
        logger.info(f"Fetching all edges for graph {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            })

        logger.info(f"Loaded {len(edges_data)} edges")
        return edges_data
    
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Edges touching a node (with retries)."""
        try:
            edges = self._call_with_retry(
                func=lambda: self.client.graph.node.get_entity_edges(node_uuid=node_uuid),
                operation_name=f"node_edges(node={node_uuid[:8]}...)"
            )
            
            edges_data = []
            for edge in edges:
                edges_data.append({
                    "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes or {},
                })
            
            return edges_data
        except Exception as e:
            logger.warning(f"Failed to load edges for node {node_uuid}: {str(e)}")
            return []
    
    def filter_defined_entities(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Keep nodes whose labels include a custom ontology type (not only Entity/Node).

        If ``defined_entity_types`` is set, only those labels qualify.
        """
        logger.info(f"Filtering entities in graph {graph_id}...")
        
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        # All edges for neighborhood expansion
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        
        # uuid -> node dict
        node_map = {n["uuid"]: n for n in all_nodes}
        
        # Entities passing the label filter
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            
            # Need at least one label beyond Entity/Node
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            
            if not custom_labels:
                # Only Zep defaults — skip
                continue
            
            # Optional allowlist
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]
            
            entity_types_found.add(entity_type)
            
            # Materialize EntityNode
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            if enrich_with_edges:
                entity.related_edges, entity.related_nodes = self._enrich_entity(
                    node["uuid"], all_edges, node_map
                )
            
            filtered_entities.append(entity)
        
        # LLM fallback: if Zep failed to classify any nodes, use LLM to assign types
        if len(filtered_entities) == 0 and total_count > 0 and defined_entity_types:
            logger.warning(
                f"0 entities passed Zep label filter out of {total_count} nodes. "
                f"Triggering LLM fallback classification..."
            )
            untyped_nodes = [
                n for n in all_nodes
                if all(l in ("Entity", "Node") for l in n.get("labels", []))
            ]
            if untyped_nodes:
                classified = self._classify_untyped_nodes(untyped_nodes, defined_entity_types)
                for node in classified:
                    entity_type = node.get_entity_type()
                    if entity_type:
                        entity_types_found.add(entity_type)
                        if enrich_with_edges:
                            node.related_edges, node.related_nodes = self._enrich_entity(
                                node.uuid, all_edges, node_map
                            )
                        filtered_entities.append(node)
                logger.info(
                    f"LLM fallback recovered {len(filtered_entities)} entities, "
                    f"types: {entity_types_found}"
                )
        
        logger.info(f"Filter done: total_nodes={total_count}, kept={len(filtered_entities)}, "
                   f"types={entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    @staticmethod
    def _enrich_entity(
        entity_uuid: str,
        all_edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, Any]],
    ) -> tuple:
        """Return (related_edges, related_nodes) for a given entity."""
        related_edges = []
        related_node_uuids: Set[str] = set()
        
        for edge in all_edges:
            if edge["source_node_uuid"] == entity_uuid:
                related_edges.append({
                    "direction": "outgoing",
                    "edge_name": edge["name"],
                    "fact": edge["fact"],
                    "target_node_uuid": edge["target_node_uuid"],
                })
                related_node_uuids.add(edge["target_node_uuid"])
            elif edge["target_node_uuid"] == entity_uuid:
                related_edges.append({
                    "direction": "incoming",
                    "edge_name": edge["name"],
                    "fact": edge["fact"],
                    "source_node_uuid": edge["source_node_uuid"],
                })
                related_node_uuids.add(edge["source_node_uuid"])
        
        related_nodes = []
        for rid in related_node_uuids:
            if rid in node_map:
                rn = node_map[rid]
                related_nodes.append({
                    "uuid": rn["uuid"],
                    "name": rn["name"],
                    "labels": rn["labels"],
                    "summary": rn.get("summary", ""),
                })
        
        return related_edges, related_nodes
    
    def _classify_untyped_nodes(
        self,
        nodes: List[Dict[str, Any]],
        entity_types: List[str],
    ) -> List[EntityNode]:
        """
        Use the LLM to classify nodes that Zep failed to assign ontology labels to.
        
        Sends batches of node names + summaries to the LLM and asks it to pick
        the best matching entity type from the ontology, or "SKIP" if none fit.
        """
        from ..utils.llm_client import LLMClient
        
        llm = LLMClient()
        types_str = ", ".join(entity_types)
        
        classified: List[EntityNode] = []
        
        for batch_start in range(0, len(nodes), _CLASSIFY_BATCH_SIZE):
            batch = nodes[batch_start:batch_start + _CLASSIFY_BATCH_SIZE]
            
            nodes_desc = []
            for i, node in enumerate(batch):
                summary = (node.get("summary") or "")[:200]
                nodes_desc.append(f'{i}. name="{node["name"]}", summary="{summary}"')
            nodes_text = "\n".join(nodes_desc)
            
            prompt = (
                f"You are classifying knowledge graph nodes into entity types.\n"
                f"Available entity types: [{types_str}]\n\n"
                f"For each node below, choose the BEST matching entity type from the list above. "
                f"If no type fits, use \"SKIP\".\n\n"
                f"Nodes:\n{nodes_text}\n\n"
                f"Respond with a JSON object: {{\"classifications\": [{{\"index\": 0, \"type\": \"Stock\"}}, ...]}}"
            )
            
            try:
                result = llm.chat_json(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )
                classifications = result.get("classifications", [])
            except Exception as e:
                logger.warning(f"LLM classification batch failed: {e}")
                continue
            
            for entry in classifications:
                idx = entry.get("index")
                assigned_type = entry.get("type", "SKIP")
                if assigned_type == "SKIP" or idx is None or idx >= len(batch):
                    continue
                
                node = batch[idx]
                new_labels = list(node.get("labels", [])) + [assigned_type]
                classified.append(EntityNode(
                    uuid=node["uuid"],
                    name=node["name"],
                    labels=new_labels,
                    summary=node.get("summary", ""),
                    attributes=node.get("attributes", {}),
                ))
        
        logger.info(f"LLM classified {len(classified)}/{len(nodes)} untyped nodes")
        return classified
    
    def get_entity_with_context(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """Single entity plus neighborhood (retries on fetch)."""
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=entity_uuid),
                operation_name=f"node_detail(uuid={entity_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            edges = self.get_node_edges(entity_uuid)
            
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            # Build related edge structs
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            # Pull neighbor summaries
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
            
        except Exception as e:
            logger.error(f"Failed to load entity {entity_uuid}: {str(e)}")
            return None
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """Convenience wrapper around filter_defined_entities for one label."""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities


