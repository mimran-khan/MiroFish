"""
LLM client wrapper.
OpenAI-compatible calls; supports Vertex AI and API-key auth.
"""

import json
import re
from typing import Optional, Dict, Any, List

from ..config import Config
from .vertex_ai import create_openai_client

# Some models wrap hidden reasoning in XML-like tags; build pattern without embedding raw tags in source.
_THINK_BLOCK = re.compile("".join((r"<", r"think", r">[\s\S]*?", r"<", r"/think", r">")))


class LLMClient:
    """Thin wrapper around the configured chat completion client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.model = model or Config.LLM_MODEL_NAME
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Run a chat completion.

        Args:
            messages: OpenAI-style messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            response_format: Optional response_format (e.g. JSON mode)

        Returns:
            Assistant message content
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        content = _THINK_BLOCK.sub("", content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Chat completion that returns parsed JSON.

        Args:
            messages: OpenAI-style messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {cleaned_response}")
