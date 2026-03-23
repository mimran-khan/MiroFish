"""
OASIS simulation runner
Runs OASIS in a subprocess, tails action logs, exposes live status.
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from .zep_graph_memory_updater import ZepGraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse

logger = get_logger('mirofish.simulation_runner')

# Whether atexit/signal hooks are registered
_cleanup_registered = False

# Platform detection
IS_WINDOWS = sys.platform == 'win32'


class RunnerStatus(str, Enum):
    """Runner lifecycle flag."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """One parsed agent action."""
    round_num: int
    timestamp: str
    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Per-round aggregate stats."""
    round_num: int
    start_time: str
    end_time: Optional[str] = None
    simulated_hour: int = 0
    twitter_actions: int = 0
    reddit_actions: int = 0
    active_agents: List[int] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Mutable run state mirrored to run_state.json."""
    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE
    
    # Progress
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0
    
    # Per-platform round/time for dual-platform UI
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0
    
    # Per-platform flags
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0
    
    # Platform completion via simulation_end in actions.jsonl
    twitter_completed: bool = False
    reddit_completed: bool = False
    
    # Round summaries
    rounds: List[RoundSummary] = field(default_factory=list)
    
    # Recent actions for live feed
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50
    
    # Timestamps
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    # Error string
    error: Optional[str] = None
    
    # Child PID
    process_pid: Optional[int] = None
    
    def add_action(self, action: AgentAction):
        """Push onto recent_actions ring buffer."""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[:self.max_recent_actions]
        
        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1
        
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(self.current_round / max(self.total_rounds, 1) * 100, 1),
            # Per-platform counters
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }
    
    def to_detail_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    SimulationRunner
    
    Responsibilities:
    1. Spawn OASIS driver scripts
    2. Parse JSONL actions
    3. Expose live state via getters
    4. Stop/kill child process tree
    """
    
    # Where run_state.json lives
    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )
    
    # backend/scripts
    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )
    
    # In-memory cache
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # stdout log handles
    _stderr_files: Dict[str, Any] = {}  # stderr handles (unused when merged)
    
    # Zep memory updater handle
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled
    
    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Return cached SimulationRunState."""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]
        
        # Hydrate from disk if missing
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state
    
    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Load run_state.json into dataclass."""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Per-platform fields
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )
            
            # Deserialize recent_actions
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", ""),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))
            
            return state
        except Exception as e:
            logger.error(f"Failed to load run state: {str(e)}")
            return None
    
    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Persist run_state.json."""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")
        
        data = state.to_detail_dict()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        cls._run_states[state.simulation_id] = state
    
    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # optional cap on rounds
        enable_graph_memory_update: bool = False,  # push actions to Zep
        graph_id: str = None  # required when memory update enabled
    ) -> SimulationRunState:
        """
        Start OASIS subprocess(es).
        
        Args:
            simulation_id: workspace id
            platform: twitter|reddit|parallel
            max_rounds: optional round cap
            enable_graph_memory_update: stream actions to Zep graph
            graph_id: target graph when memory update on
            
        Returns:
            SimulationRunState
        """
        # Reject double start
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            raise ValueError(f"Simulation already running: {simulation_id}")
        
        # Read simulation_config.json
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            raise ValueError("Simulation config missing; run /prepare first")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Seed SimulationRunState
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)
        
        # Clamp total_rounds
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(f"Rounds clamped: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )
        
        cls._save_run_state(state)
        
        # Optional Zep updater
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("graph_id required when graph memory update is enabled")
            
            try:
                ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(f"Graph memory update on: simulation_id={simulation_id}, graph_id={graph_id}")
            except Exception as e:
                logger.error(f"Failed to create graph memory updater: {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False
        
        # Resolve driver under backend/scripts/
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True
        
        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            raise ValueError(f"Script not found: {script_path}")
        
        # Allocate deque for tail
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue
        
        # subprocess.Popen
        try:
            # argv
            # Log layout:
            #   twitter/actions.jsonl
            #   reddit/actions.jsonl
            #   simulation.log (driver)
            
            cmd = [
                sys.executable,
                script_path,
                "--config", config_path,
            ]
            
            # Optional --max-rounds
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])
            
            # Tee stdout/stderr to file to avoid pipe deadlock
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')
            
            # Force UTF-8 for Windows OASIS file reads
            # Mitigates libraries that open() without encoding=
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # cwd = simulation workspace
            # New session => kill process group on shutdown
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            
            # Track log handle
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None
            
            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)
            
            # Tail logs thread
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id,),
                daemon=True
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread
            
            logger.info(f"Simulation started: {simulation_id}, pid={process.pid}, platform={platform}")
            
        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise
        
        return state
    
    @classmethod
    def _monitor_simulation(cls, simulation_id: str):
        """Poll process + tail JSONL."""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        # Per-platform JSONL
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)
        
        if not process or not state:
            return
        
        twitter_position = 0
        reddit_position = 0
        
        try:
            while process.poll() is None:
                # Tail twitter/actions.jsonl
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )
                
                # Tail reddit/actions.jsonl
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )
                
                # Mutate state
                cls._save_run_state(state)
                time.sleep(2)
            
            # Final tail pass
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(twitter_actions_log, twitter_position, state, "twitter")
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(reddit_actions_log, reddit_position, state, "reddit")
            
            # Child exited
            exit_code = process.returncode
            
            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
                logger.info(f"Simulation finished: {simulation_id}")
            else:
                state.runner_status = RunnerStatus.FAILED
                # Snarf driver log tail
                main_log_path = os.path.join(sim_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, 'r', encoding='utf-8') as f:
                            error_info = f.read()[-2000:]
                except Exception:
                    pass
                state.error = f"exit_code={exit_code}, detail={error_info}"
                logger.error(f"Simulation failed: {simulation_id}, error={state.error}")
            
            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)
            
        except Exception as e:
            logger.error(f"Monitor thread error: {simulation_id}, error={str(e)}")
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
        
        finally:
            # Stop Zep updater
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    ZepGraphMemoryManager.stop_updater(simulation_id)
                    logger.info(f"Graph memory updater stopped: simulation_id={simulation_id}")
                except Exception as e:
                    logger.error(f"Failed to stop graph memory updater: {e}")
                cls._graph_memory_enabled.pop(simulation_id, None)
            
            # Release handles
            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)
            
            # Close log files
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)
    
    @classmethod
    def _read_action_log(
        cls, 
        log_path: str, 
        position: int, 
        state: SimulationRunState,
        platform: str
    ) -> int:
        """
        Tail one platform's actions.jsonl.
        
        Args:
            log_path: path to JSONL
            position: last file offset
            state: SimulationRunState
            platform: twitter|reddit
            
        Returns:
            New offset
        """
        # Optional Zep streaming
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = ZepGraphMemoryManager.get_updater(state.simulation_id)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)
                            
                            # Control-plane JSON rows
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")
                                
                                # simulation_end marks platform completion
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(f"Twitter simulation complete: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(f"Reddit simulation complete: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    
                                    # All enabled platforms done?
                                    # Single-platform run
                                    # Dual-platform needs both
                                    all_completed = cls._check_all_platforms_completed(state)
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        logger.info(f"All enabled platforms finished: {state.simulation_id}")
                                
                                # round_end aggregates
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get("simulated_hours", 0)
                                    
                                    # Per-platform counters
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours
                                    
                                    # Global round = max
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Global sim time = max
                                    state.simulated_hours = max(state.twitter_simulated_hours, state.reddit_simulated_hours)
                                
                                continue
                            
                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)
                            
                            # Bump counters
                            if action.round_num and action.round_num > state.current_round:
                                state.current_round = action.round_num
                            
                            # Forward to Zep updater
                            if graph_updater:
                                graph_updater.add_activity_from_dict(action_data, platform)
                            
                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Failed to read action log: {log_path}, error={e}")
            return position
    
    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        True when every enabled platform emitted simulation_end.
        
        Platform enabled iff its actions.jsonl exists.
        
        Returns:
            bool
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        # Enabled if JSONL path exists
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)
        
        # Enabled but incomplete => False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False
        
        # Need >=1 enabled and done
        return twitter_enabled or reddit_enabled
    
    @classmethod
    def _terminate_process(cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10):
        """
        Kill subprocess tree (Windows + Unix).
        
        Args:
            process: subprocess.Popen handle
            simulation_id: workspace id (for logs)
            timeout: seconds
        """
        if IS_WINDOWS:
            # Windows: taskkill /T
            # /F /T
            logger.info(f"Killing process tree (Windows): simulation={simulation_id}, pid={process.pid}")
            try:
                # try terminate()
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True,
                    timeout=5
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # force kill
                    logger.warning(f"Process unresponsive, forcing kill: {simulation_id}")
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill failed, trying terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: kill process group
            # start_new_session => pgid == child pid
            pgid = os.getpgid(process.pid)
            logger.info(f"Killing process group (Unix): simulation={simulation_id}, pgid={pgid}")
            
            # SIGTERM whole group
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # escalate to SIGKILL
                logger.warning(f"Process group ignored SIGTERM, SIGKILL: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)
    
    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Stop subprocess and updater."""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(f"Simulation not running: {simulation_id}, status={state.runner_status}")
        
        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)
        
        # Kill tree
        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                # Already dead
                pass
            except Exception as e:
                logger.error(f"Failed to kill process group: {simulation_id}, error={e}")
                # Fallback: process.kill
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        
        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)
        
        # Stop Zep graph memory updater
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Graph memory updater stopped: simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Failed to stop graph memory updater: {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)
        
        logger.info(f"Simulation stopped: {simulation_id}")
        return state
    
    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Parse JSONL rows into AgentAction records.
        
        Args:
            file_path: JSONL path
            default_platform: fallback platform field
            platform_filter: optional
            agent_id: optional
            round_num: optional
        """
        if not os.path.exists(file_path):
            return []
        
        actions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip control events
                    if "event_type" in data:
                        continue
                    
                    # Require agent_id
                    if "agent_id" not in data:
                        continue
                    
                    # platform column or default
                    record_platform = data.get("platform") or default_platform or ""
                    
                    # Apply filters
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue
                    
                    actions.append(AgentAction(
                        round_num=data.get("round", 0),
                        timestamp=data.get("timestamp", ""),
                        platform=record_platform,
                        agent_id=data.get("agent_id", 0),
                        agent_name=data.get("agent_name", ""),
                        action_type=data.get("action_type", ""),
                        action_args=data.get("action_args", {}),
                        result=data.get("result"),
                        success=data.get("success", True),
                    ))
                    
                except json.JSONDecodeError:
                    continue
        
        return actions
    
    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Merge per-platform logs (unbounded).
        
        Args:
            simulation_id: workspace id
            platform: optional
            agent_id: optional
            round_num: optional
            
        Returns:
            Newest-first list
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        actions = []
        
        # twitter/actions.jsonl
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(cls._read_actions_from_file(
                twitter_actions_log,
                default_platform="twitter",
                platform_filter=platform,
                agent_id=agent_id, 
                round_num=round_num
            ))
        
        # reddit/actions.jsonl
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(cls._read_actions_from_file(
                reddit_actions_log,
                default_platform="reddit",
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            ))
        
        # Legacy single actions.jsonl
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            )
        
        # Sort desc by timestamp
        actions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return actions
    
    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Paginated slice of merged actions.
        
        Args:
            simulation_id: workspace id
            limit: page size
            offset: skip
            platform: optional platform filter
            agent_id: optional
            round_num: optional
            
        Returns:
            list[AgentAction]
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        # Paginate
        return actions[offset:offset + limit]
    
    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build per-round timeline
        
        Args:
            simulation_id: workspace id
            start_round: start_round
            end_round: end_round
            
        Returns:
            Per-round summary rows
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        # Group by round
        rounds: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            round_num = action.round_num
            
            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue
            
            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            r = rounds[round_num]
            
            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1
            
            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = r["action_types"].get(action.action_type, 0) + 1
            r["last_action_time"] = action.timestamp
        
        # Materialize sorted list
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append({
                "round_num": round_num,
                "twitter_actions": r["twitter_actions"],
                "reddit_actions": r["reddit_actions"],
                "total_actions": r["twitter_actions"] + r["reddit_actions"],
                "active_agents_count": len(r["active_agents"]),
                "active_agents": list(r["active_agents"]),
                "action_types": r["action_types"],
                "first_action_time": r["first_action_time"],
                "last_action_time": r["last_action_time"],
            })
        
        return result
    
    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """
        Per-agent aggregates
        
        Returns:
            Sorted agent stats
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        agent_stats: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            agent_id = action.agent_id
            
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            stats = agent_stats[agent_id]
            stats["total_actions"] += 1
            
            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1
            
            stats["action_types"][action.action_type] = stats["action_types"].get(action.action_type, 0) + 1
            stats["last_action_time"] = action.timestamp
        
        # Sort by total_actions desc
        result = sorted(agent_stats.values(), key=lambda x: x["total_actions"], reverse=True)
        
        return result
    
    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Delete ephemeral run artifacts (force restart)
        
        Deletes:
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db
        - reddit_simulation.db
        - env_status.json
        
        Keeps simulation_config.json and profile files
        
        Args:
            simulation_id: workspace id
            
        Returns:
            Cleanup summary dict
        """
        import shutil
        
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return {"success": True, "message": "No workspace; nothing to clean"}
        
        cleaned_files = []
        errors = []
        
        # Files to delete
        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            "twitter_simulation.db",
            "reddit_simulation.db",
            "env_status.json",
        ]
        
        # Dirs to clean
        dirs_to_clean = ["twitter", "reddit"]
        
        # Delete files
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Delete {filename} failed: {str(e)}")
        
        # Remove platform action logs
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(f"Delete {dir_name}/actions.jsonl failed: {str(e)}")
        
        # Drop cached run state
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]
        
        logger.info(f"Run log cleanup done: {simulation_id}, files={cleaned_files}")
        
        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None
        }
    
    # Idempotent cleanup guard
    _cleanup_done = False
    
    @classmethod
    def cleanup_all_simulations(cls):
        """
        Stop every child simulation process
        
        Called on shutdown to terminate children
        """
        # Prevent duplicate cleanup
        if cls._cleanup_done:
            return
        cls._cleanup_done = True
        
        # Skip noisy logs when idle
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)
        
        if not has_processes and not has_updaters:
            return
        
        logger.info("Stopping all simulation processes...")
        
        # Stop Zep updaters first
        try:
            ZepGraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Failed to stop graph memory updater: {e}")
        cls._graph_memory_enabled.clear()
        
        # Snapshot dict before iteration
        processes = list(cls._processes.items())
        
        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # still alive
                    logger.info(f"Stopping simulation process: {simulation_id}, pid={process.pid}")
                    
                    try:
                        # Use cross-platform kill
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Already dead; try terminate
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()
                    
                    # Persist run_state.json
                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Server shutdown terminated simulation"
                        cls._save_run_state(state)
                    
                    # Mirror SimulationManager state.json to stopped
                    try:
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Updating state.json: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                            state_data['status'] = 'stopped'
                            state_data['updated_at'] = datetime.now().isoformat()
                            with open(state_file, 'w', encoding='utf-8') as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"state.json -> stopped: {simulation_id}")
                        else:
                            logger.warning(f"state.json missing: {state_file}")
                    except Exception as state_err:
                        logger.warning(f"state.json update failed: {simulation_id}, error={state_err}")
                        
            except Exception as e:
                logger.error(f"Process cleanup failed: {simulation_id}, error={e}")
        
        # Close log handles
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()
        
        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()
        
        # Clear in-memory registry
        cls._processes.clear()
        cls._action_queues.clear()
        
        logger.info("Simulation cleanup finished")
    
    @classmethod
    def register_cleanup(cls):
        """
        Register shutdown hooks
        
        Call from app factory so shutdown cleans simulations
        """
        global _cleanup_registered
        
        if _cleanup_registered:
            return
        
# Flask debug: register cleanup in reloader child only
        # WERKZEUG_RUN_MAIN=true marks the reloader child
        # Non-debug runs still need hooks
        is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('WERKZEUG_RUN_MAIN') is not None
        
        # Debug: child only; production: always register
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # avoid double-register in spawn
            return
        
        # Chain prior signal handlers
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP: Unix only
        original_sighup = None
        has_sighup = hasattr(signal, 'SIGHUP')
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)
        
        def cleanup_handler(signum=None, frame=None):
            """Run cleanup then chain previous handler."""
            # Log only if work to do
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Received signal {signum}, starting cleanup...")
            cls.cleanup_all_simulations()
            
            # Chain Flask exit
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: terminal closed
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # default: exit 0
                    sys.exit(0)
            else:
                # Non-callable handler -> KeyboardInterrupt
                raise KeyboardInterrupt
        
        # atexit fallback
        atexit.register(cls.cleanup_all_simulations)
        
        # signals on main thread only
        try:
            # SIGTERM
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP (Unix)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # Non-main thread: atexit only
            logger.warning("Signals unavailable; atexit only")
        
        _cleanup_registered = True
    
    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        List running simulation ids
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running
    
    # ============== Interview ==============
    
    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        IPC env alive for interviews

        Args:
            simulation_id: workspace id

        Returns:
            True when IPC env accepts commands
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        env_status.json detail

        Args:
            simulation_id: workspace id

        Returns:
            Dict with status, twitter_available, reddit_available, timestamp
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")
        
        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None
        }
        
        if not os.path.exists(status_file):
            return default_status
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp")
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Interview one agent

        Args:
            simulation_id: workspace id
            agent_id: Agent ID
            prompt: prompt
            platform: optional
                - "twitter": Twitter only
                - "reddit": Reddit only
                - None: dual-platform merged responses
            timeout: timeout seconds

        Returns:
            result dict

        Raises:
            ValueError: bad id or env down
            TimeoutError: IPC timeout
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation not found: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Env not running; cannot interview: {simulation_id}")

        logger.info(f"Sending interview IPC: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}")

        response = ipc_client.send_interview(
            agent_id=agent_id,
            prompt=prompt,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Batch interview

        Args:
            simulation_id: workspace id
            interviews: list of {agent_id, prompt, optional platform}
            platform: default (per-item overrides)
                - "twitter": default Twitter
                - "reddit": default Reddit
                - None: interview both platforms per agent
            timeout: timeout seconds

        Returns:
            batch result

        Raises:
            ValueError: bad id or env down
            TimeoutError: IPC timeout
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation not found: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"Env not running; cannot interview: {simulation_id}")

        logger.info(f"Sending batch interview IPC: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}")

        response = ipc_client.send_batch_interview(
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: str = None,
        timeout: float = 180.0
    ) -> Dict[str, Any]:
        """
        Interview every agent

        Same prompt for all agents

        Args:
            simulation_id: workspace id
            prompt: shared question
            platform: optional
                - "twitter": Twitter only
                - "reddit": Reddit only
                - None: interview both platforms per agent
            timeout: timeout seconds

        Returns:
            global interview payload
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation not found: {simulation_id}")

        # Load agents from config
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Simulation config missing: {simulation_id}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(f"No agents in simulation config: {simulation_id}")

        # Build interview payloads
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({
                    "agent_id": agent_id,
                    "prompt": prompt
                })

        logger.info(f"Sending global interview IPC: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}")

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )
    
    @classmethod
    def close_simulation_env(
        cls,
        simulation_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
Ask OASIS to close its IPC loop (not SIGKILL).
        
Sends close_env IPC so the driver exits wait mode.
        
        Args:
            simulation_id: workspace id
            timeout: timeout seconds
            
        Returns:
            Result dict
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        ipc_client = SimulationIPCClient(sim_dir)
        
        if not ipc_client.check_env_alive():
            return {
                "success": True,
"message": "Environment already closed"
            }
        
        logger.info(f"Sending close-env IPC: simulation_id={simulation_id}")
        
        try:
            response = ipc_client.send_close_env(timeout=timeout)
            
            return {
                "success": response.status.value == "completed",
"message": "Close-env command sent",
                "result": response.result,
                "timestamp": response.timestamp
            }
        except TimeoutError:
            # Timeout may mean env is shutting down
            return {
                "success": True,
"message": "Close-env sent (timeout while shutting down)"
            }
    
    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Interview rows from one platform DB."""
        import sqlite3
        
        if not os.path.exists(db_path):
            return []
        
        results = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if agent_id is not None:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (agent_id, limit))
            else:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}
                
                results.append({
                    "agent_id": user_id,
                    "response": info.get("response", info),
                    "prompt": info.get("prompt", ""),
                    "timestamp": created_at,
                    "platform": platform_name
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to read interview history ({platform_name}): {e}")
        
        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: str = None,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Read interview table(s) from SQLite.
        
        Args:
            simulation_id: workspace id
            platform: reddit|twitter|None
                - "reddit": Reddit only
                - "twitter": Twitter only
                - None: merge both
            agent_id: optional filter
            limit: per-platform cap
            
        Returns:
            History rows
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        results = []
        
        # Platforms to query
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # None => both
            platforms = ["twitter", "reddit"]
        
        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path,
                platform_name=p,
                agent_id=agent_id,
                limit=limit
            )
            results.extend(platform_results)
        
        # Sort newest first
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Cap combined rows
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]
        
        return results

