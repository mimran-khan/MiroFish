"""
Google Vertex AI integration.

Creates OpenAI-compatible clients authenticated via service account credentials.
Tokens are automatically refreshed before each request via httpx event hooks,
so the returned client is safe for long-running processes.
"""

import json
import os
from typing import Optional

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from openai import OpenAI

from ..config import Config

VERTEX_AI_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_credentials = None


def _load_credentials():
    """Load and cache Google service account credentials."""
    global _credentials
    if _credentials is not None:
        return _credentials

    creds_path = Config.GOOGLE_APPLICATION_CREDENTIALS
    if creds_path:
        resolved = os.path.join(
            os.path.dirname(__file__), '../../..', creds_path
        ) if not os.path.isabs(creds_path) else creds_path

        if os.path.exists(resolved):
            _credentials = service_account.Credentials.from_service_account_file(
                resolved, scopes=VERTEX_AI_SCOPES
            )
            return _credentials

    return None


def get_vertex_base_url(
    project_id: str,
    location: str = "us-central1",
) -> str:
    """Build the Vertex AI OpenAI-compatible base URL."""
    return (
        f"https://{location}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_id}/locations/{location}/endpoints/openapi/"
    )


def get_access_token() -> Optional[str]:
    """Return a valid access token, refreshing if needed."""
    credentials = _load_credentials()
    if credentials is None:
        return None
    if not credentials.valid:
        credentials.refresh(GoogleAuthRequest())
    return credentials.token


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    Create an OpenAI client.

    If Vertex AI is configured (VERTEX_AI_PROJECT is set), returns a client
    authenticated with service account credentials and auto-refreshing tokens.
    Otherwise falls back to standard API-key auth.
    """
    if Config.VERTEX_AI_PROJECT:
        credentials = _load_credentials()
        if credentials is None:
            raise ValueError(
                "VERTEX_AI_PROJECT is set but no credentials found. "
                "Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file path."
            )

        vertex_base_url = get_vertex_base_url(
            Config.VERTEX_AI_PROJECT,
            Config.VERTEX_AI_LOCATION,
        )

        def _inject_auth(request: httpx.Request):
            if not credentials.valid:
                credentials.refresh(GoogleAuthRequest())
            request.headers["Authorization"] = f"Bearer {credentials.token}"

        http_client = httpx.Client(
            event_hooks={"request": [_inject_auth]},
        )

        return OpenAI(
            api_key="VERTEX_AI_PLACEHOLDER",
            base_url=vertex_base_url,
            http_client=http_client,
        )

    key = api_key or Config.LLM_API_KEY
    url = base_url or Config.LLM_BASE_URL
    if not key:
        raise ValueError("LLM_API_KEY is not configured")

    return OpenAI(api_key=key, base_url=url)
