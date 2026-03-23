"""
Configuration management.
Loads settings from the project root `.env` file.
"""

import os
from dotenv import load_dotenv

# Load `.env` from project root.
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If root `.env` is missing, still load from the process environment (e.g. production).
    load_dotenv(override=True)


class Config:
    """Flask configuration."""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON: disable ASCII escaping so non-ASCII text is not shown as \uXXXX.
    JSON_AS_ASCII = False
    
    # LLM (OpenAI-compatible)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gemini-2.0-flash')
    
    # Google Vertex AI
    VERTEX_AI_PROJECT = os.environ.get('VERTEX_AI_PROJECT')
    VERTEX_AI_LOCATION = os.environ.get('VERTEX_AI_LOCATION', 'us-central1')
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Zep
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # Uploads
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown', 'csv', 'xlsx'}
    
    # Text chunking defaults
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    
    # OASIS simulation
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS platform actions
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report agent
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """Validate required settings."""
        errors = []
        if not cls.VERTEX_AI_PROJECT and not cls.LLM_API_KEY:
            errors.append("Neither LLM_API_KEY nor VERTEX_AI_PROJECT is configured")
        if cls.VERTEX_AI_PROJECT and not cls.GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("VERTEX_AI_PROJECT is set but GOOGLE_APPLICATION_CREDENTIALS is missing")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured")
        return errors
