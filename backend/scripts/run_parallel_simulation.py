"""
OASIS dual-platform parallel simulation driver.

Runs Twitter and Reddit simulations together from the same config.

Features:
- Twitter + Reddit in parallel
- After the run, keeps environments alive for IPC (unless --no-wait)
- Interview commands over IPC
- Single-agent and batch interviews
- Remote close-environment command

Usage:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

Log layout:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter action log
    ├── reddit/
    │   └── actions.jsonl    # Reddit action log
    ├── simulation.log       # Main process log
    └── run_state.json       # Run state (for API)
"""

# ============================================================
# Windows: force UTF-8 before imports (OASIS opens files without encoding)
# ============================================================
import sys
import os

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Monkey-patch open() so text mode defaults to UTF-8
    import builtins
    _original_open = builtins.open
    
    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, 
                   newline=None, closefd=True, opener=None):
        """
        Wrap open(): default text mode encoding to UTF-8 for libraries that omit encoding.
        """
        # Text mode only when encoding not set
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, 
                              newline, closefd, opener)
    
    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import multiprocessing
import random
import signal
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# Globals for signal handling
_shutdown_event = None
_cleanup_done = False

# Script lives under backend/scripts/
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Load project root .env (LLM_API_KEY, etc.)
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Loaded env from: {_env_file}")
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Loaded env from: {_backend_env}")


class MaxTokensWarningFilter(logging.Filter):
    """Drop camel-ai max_tokens warnings (we omit max_tokens on purpose)."""

    def filter(self, record):
        # Drop max_tokens warning lines
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    Silence verbose OASIS loggers; we use action_logger instead.
    """
    # CRITICAL on OASIS loggers
    oasis_loggers = [
        "social.agent",
        "social.twitter", 
        "social.rec",
        "oasis.env",
        "table",
    ]
    
    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers.clear()
        logger.propagate = False


def init_logging_for_simulation(simulation_dir: str):
    """
    Disable OASIS noise logging and remove legacy log/ under simulation_dir.

    Args:
        simulation_dir: Simulation directory path
    """
    disable_oasis_logging()

    # Remove old OASIS log dir if present
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager, PlatformActionLogger

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"Error: missing dependency {e}")
    print("Install with: pip install oasis-ai camel-ai")
    sys.exit(1)


# Twitter actions (INTERVIEW is ManualAction-only)
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Reddit actions (INTERVIEW is ManualAction-only)
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
]


# IPC paths
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """IPC command type strings."""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """
    IPC for both platforms: routes interview commands to Twitter/Reddit envs.
    """
    
    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None
    ):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str):
        """Write env status JSON."""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Next pending command JSON, oldest first."""
        if not os.path.exists(self.commands_dir):
            return None

        # Sort command files by mtime
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Write response and delete the command file."""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def _get_env_and_graph(self, platform: str):
        """
        Args:
            platform: "twitter" or "reddit"

        Returns:
            (env, agent_graph, platform_name) or (None, None, None)
        """
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None
    
    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict[str, Any]:
        """
        Run interview on one platform.

        Returns:
            Result dict or {"error": ...}
        """
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        
        if not env or not agent_graph:
            return {"platform": platform, "error": f"{platform} environment unavailable"}
        
        try:
            agent = agent_graph.get_agent(agent_id)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            actions = {agent: interview_action}
            await env.step(actions)
            
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
            
        except Exception as e:
            return {"platform": platform, "error": str(e)}
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str, platform: str = None) -> bool:
        """
        Single-agent interview on one or both platforms.

        Args:
            command_id: Command id
            agent_id: Agent id
            prompt: Interview prompt
            platform: Optional "twitter", "reddit", or None for both

        Returns:
            True on success
        """
        # Single platform
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(f"  Interview failed: agent_id={agent_id}, platform={platform}, error={result['error']}")
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(f"  Interview done: agent_id={agent_id}, platform={platform}")
                return True
        
        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="No simulation environment available")
            return False
        
        results = {
            "agent_id": agent_id,
            "prompt": prompt,
            "platforms": {}
        }
        success_count = 0
        
        tasks = []
        platforms_to_interview = []
        
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")
        
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")
        
        platform_results = await asyncio.gather(*tasks)
        
        for platform_name, platform_result in zip(platforms_to_interview, platform_results):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1
        
        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(f"  Interview done: agent_id={agent_id}, platforms_ok={success_count}/{len(platforms_to_interview)}")
            return True
        else:
            errors = [f"{p}: {r.get('error', 'unknown error')}" for p, r in results["platforms"].items()]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(f"  Interview failed: agent_id={agent_id}, all platforms failed")
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict], platform: str = None) -> bool:
        """
        Batch interview; optional per-item or default platform.

        Args:
            command_id: Command id
            interviews: [{"agent_id", "prompt", optional "platform"}, ...]
            platform: Default platform; None means both where envs exist
        """
        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []
        
        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:
                both_platforms_interviews.append(interview)

        # Fan out "both" to each running env
        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)
        
        results = {}
        
        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Warning: could not resolve Twitter agent {agent_id}: {e}")
                
                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)
                    
                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  Twitter batch interview failed: {e}")

        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Warning: could not resolve Reddit agent {agent_id}: {e}")
                
                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)
                    
                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  Reddit batch interview failed: {e}")
        
        if results:
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Batch interview done: {len(results)} agent(s)")
            return True
        else:
            self.send_response(command_id, "failed", error="No successful interviews")
            return False
    
    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """Latest interview row from trace table."""
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
            return result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))
            
            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            
            conn.close()
            
        except Exception as e:
            print(f"  Failed to read interview result: {e}")
        
        return result
    
    async def process_commands(self) -> bool:
        """
        Process one pending command if any.

        Returns:
            False after CLOSE_ENV (stop waiting); True to keep polling.
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\nIPC command: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Close-environment command received")
            self.send_response(command_id, "completed", result={"message": "Environment will close"})
            return False

        else:
            self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
            return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON simulation config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Low-value actions to skip in exported logs
FILTERED_ACTIONS = {'refresh', 'sign_up'}

# DB action name -> canonical label
ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}


def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Build agent_id -> entity_name from simulation_config for nicer action logs.

    Args:
        config: Parsed simulation_config.json

    Returns:
        Mapping agent_id -> entity_name
    """
    agent_names = {}
    agent_configs = config.get("agent_configs", [])
    
    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name
    
    return agent_names


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Read new trace rows since last_rowid and enrich with post/comment context.

    Args:
        db_path: SQLite path
        last_rowid: Last consumed SQLite rowid (avoids created_at format differences)
        agent_names: agent_id -> display name

    Returns:
        (actions_list, new_last_rowid)
    """
    actions = []
    new_last_rowid = last_rowid
    
    if not os.path.exists(db_path):
        return actions, new_last_rowid
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # rowid tracks progress (Twitter vs Reddit use different created_at types)
        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))
        
        for rowid, user_id, action, info_json in cursor.fetchall():
            new_last_rowid = rowid

            if action in FILTERED_ACTIONS:
                continue

            # Parse action payload
            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}
            
            simplified_args = {}
            if 'content' in action_args:
                simplified_args['content'] = action_args['content']
            if 'post_id' in action_args:
                simplified_args['post_id'] = action_args['post_id']
            if 'comment_id' in action_args:
                simplified_args['comment_id'] = action_args['comment_id']
            if 'quoted_id' in action_args:
                simplified_args['quoted_id'] = action_args['quoted_id']
            if 'new_post_id' in action_args:
                simplified_args['new_post_id'] = action_args['new_post_id']
            if 'follow_id' in action_args:
                simplified_args['follow_id'] = action_args['follow_id']
            if 'query' in action_args:
                simplified_args['query'] = action_args['query']
            if 'like_id' in action_args:
                simplified_args['like_id'] = action_args['like_id']
            if 'dislike_id' in action_args:
                simplified_args['dislike_id'] = action_args['dislike_id']
            
            action_type = ACTION_TYPE_MAP.get(action, action.upper())

            _enrich_action_context(cursor, action_type, simplified_args, agent_names)
            
            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified_args,
            })
        
        conn.close()
    except Exception as e:
        print(f"Failed to read actions from database: {e}")
    
    return actions, new_last_rowid


def _enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str]
) -> None:
    """
    Mutate action_args with post/comment/author context where applicable.

    Args:
        cursor: DB cursor
        action_type: Canonical action type
        action_args: Payload dict (updated in place)
        agent_names: agent_id -> display name
    """
    try:
        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
        
        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:
                cursor.execute("""
                    SELECT original_post_id FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(cursor, original_post_id, agent_names)
                    if original_info:
                        action_args['original_content'] = original_info.get('content', '')
                        action_args['original_author_name'] = original_info.get('author_name', '')
        
        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')
            
            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args['original_content'] = original_info.get('content', '')
                    action_args['original_author_name'] = original_info.get('author_name', '')
            
            if new_post_id:
                cursor.execute("""
                    SELECT quote_content FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]
        
        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:
                cursor.execute("""
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """, (follow_id,))
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args['target_user_name'] = target_name
        
        elif action_type == 'MUTE':
            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args['target_user_name'] = target_name
        
        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args['comment_content'] = comment_info.get('content', '')
                    action_args['comment_author_name'] = comment_info.get('author_name', '')
        
        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
    
    except Exception as e:
        print(f"Action context enrichment failed: {e}")


def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Args:
        cursor: DB cursor
        post_id: Post id
        agent_names: agent_id -> display name

    Returns:
        {"content", "author_name"} or None
    """
    try:
        cursor.execute("""
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """, (post_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]

            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str]
) -> Optional[str]:
    """
    Args:
        cursor: DB cursor
        user_id: User row id
        agent_names: agent_id -> display name

    Returns:
        Display name or None
    """
    try:
        cursor.execute("""
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]
            
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except Exception:
        pass
    return None


def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Args:
        cursor: DB cursor
        comment_id: Comment id
        agent_names: agent_id -> display name

    Returns:
        {"content", "author_name"} or None
    """
    try:
        cursor.execute("""
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """, (comment_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def _get_vertex_ai_token() -> str:
    """Get a fresh Vertex AI access token from service account credentials."""
    from google.oauth2 import service_account as sa
    from google.auth.transport.requests import Request as GoogleAuthRequest

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not os.path.isabs(creds_path):
        creds_path = os.path.join(_project_root, creds_path)

    credentials = sa.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(GoogleAuthRequest())
    return credentials.token


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    Create LLM: Vertex AI if VERTEX_AI_PROJECT set; else optional boost LLM_*; else standard API key.

    Args:
        config: Simulation config dict
        use_boost: Prefer LLM_BOOST_* when True and present
    """
    vertex_project = os.environ.get("VERTEX_AI_PROJECT", "")
    llm_model = os.environ.get("LLM_MODEL_NAME", "")

    if vertex_project:
        location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
        vertex_base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{vertex_project}/locations/{location}/endpoints/openapi/"
        )

        token = _get_vertex_ai_token()
        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_API_BASE_URL"] = vertex_base_url

        if not llm_model:
            llm_model = config.get("llm_model", "gemini-2.0-flash")

        print(f"[Vertex AI] model={llm_model}, project={vertex_project}, location={location}")

        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )

    # Non-Vertex path: standard API key auth
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)

    if use_boost and has_boost_config:
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or llm_model
        config_label = "[boost LLM]"
    else:
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        config_label = "[default LLM]"

    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")

    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Missing API key: set LLM_API_KEY or VERTEX_AI_PROJECT in project root .env")

    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url

    print(f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'default'}...")

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """Select active agents for this round from time_config and per-agent settings."""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])
    
    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    
    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0
    
    target_count = int(random.uniform(base_min, base_max) * multiplier)
    
    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        
        if current_hour not in active_hours:
            continue
        
        if random.random() < activity_level:
            candidates.append(agent_id)
    
    selected_ids = random.sample(
        candidates, 
        min(target_count, len(candidates))
    ) if candidates else []
    
    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    
    return active_agents


class PlatformSimulation:
    """Holds env, agent_graph, and action count for one platform run."""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Run Twitter side of the parallel simulation.

    Args:
        config: Simulation config dict
        simulation_dir: Directory with profiles and DB
        action_logger: Optional per-platform JSONL logger
        main_logger: Optional simulation.log manager
        max_rounds: Optional round cap

    Returns:
        PlatformSimulation with env and agent_graph (env left open for IPC)
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")
    
    log_info("Initializing...")
    
    model = create_model(config, use_boost=False)

    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"Error: profile file not found: {profile_path}")
        return result
    
    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    
    agent_names = get_agent_names_from_config(config)
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=30,  # cap concurrent LLM calls
    )
    
    await result.env.reset()
    log_info("Environment up")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # last consumed SQLite rowid
    
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"Posted {len(initial_actions)} initial post(s)")
    
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Rounds capped: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Shutdown signal: stopping after round {round_num + 1}")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)
        
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop done in {elapsed:.1f}s, total actions: {total_actions}")
    
    return result


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Run Reddit side of the parallel simulation.

    Args:
        config: Simulation config dict
        simulation_dir: Directory with profiles and DB
        action_logger: Optional per-platform JSONL logger
        main_logger: Optional simulation.log manager
        max_rounds: Optional round cap

    Returns:
        PlatformSimulation with env and agent_graph (env left open for IPC)
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")
    
    log_info("Initializing...")
    
    model = create_model(config, use_boost=True)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Error: profile file not found: {profile_path}")
        return result
    
    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    
    agent_names = get_agent_names_from_config(config)
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=30,  # cap concurrent LLM calls
    )
    
    await result.env.reset()
    log_info("Environment up")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # last consumed SQLite rowid
    
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    ))
                else:
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"Posted {len(initial_actions)} initial post(s)")
    
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Rounds capped: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Shutdown signal: stopping after round {round_num + 1}")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)
        
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop done in {elapsed:.1f}s, total actions: {total_actions}")
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='OASIS dual-platform parallel simulation')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to simulation_config.json'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='Run Twitter simulation only'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='Run Reddit simulation only'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Optional max rounds (truncate long runs)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Exit immediately after simulation (no IPC wait mode)'
    )

    args = parser.parse_args()

    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    
    init_logging_for_simulation(simulation_dir)

    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    
    log_manager.info("=" * 60)
    log_manager.info("OASIS dual-platform parallel simulation")
    log_manager.info(f"Config: {args.config}")
    log_manager.info(f"Simulation ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info(f"Wait-for-commands: {'on' if wait_for_commands else 'off'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info("Simulation parameters:")
    log_manager.info(f"  - Total simulated hours: {total_hours}")
    log_manager.info(f"  - Minutes per round: {minutes_per_round}")
    log_manager.info(f"  - Configured total rounds: {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - max_rounds cap: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - Effective rounds: {args.max_rounds} (truncated)")
    log_manager.info(f"  - Agent count: {len(config.get('agent_configs', []))}")

    log_manager.info("Log layout:")
    log_manager.info("  - Main: simulation.log")
    log_manager.info("  - Twitter actions: twitter/actions.jsonl")
    log_manager.info("  - Reddit actions: reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    if args.twitter_only:
        twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds)
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds)
    else:
        results = await asyncio.gather(
            run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds),
            run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds),
        )
        twitter_result, reddit_result = results
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"Simulation loops finished in {total_elapsed:.1f}s total")

    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("Waiting for commands (environments stay up)")
        log_manager.info("Commands: interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None
        )
        ipc_handler.update_status("alive")
        
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # shutdown signal
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            print("\nInterrupted")
        except asyncio.CancelledError:
            print("\nTask cancelled")
        except Exception as e:
            print(f"\nCommand handling error: {e}")

        log_manager.info("\nClosing environments...")
        ipc_handler.update_status("stopped")
    
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] Environment closed")

    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] Environment closed")

    log_manager.info("=" * 60)
    log_manager.info("All done.")
    log_manager.info("Log files:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)


def setup_signal_handlers(loop=None):
    """
    SIGTERM/SIGINT: set _shutdown_event first so asyncio can close envs;
    second signal forces exit.
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n{sig_name} received, shutting down...")

        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()

        else:
            print("Forced exit...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except SystemExit:
        pass
    finally:
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Simulation process exited")
