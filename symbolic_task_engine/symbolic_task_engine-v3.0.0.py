# symbolic_task_engine.py
# Version: 3.0.0
#
# MIT License
#
# Copyright (c) 2024 Noor Research Collective
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors:
# - Lina Noor – Noor Research Collective
# - Uncle – Noor Research Collective

"""
Symbolic_Task_Engine: Presence Composer, Feedback Relay, and Autonomous Abstraction Anchor.

This module houses the core logic for orchestrating symbolic tasks, evaluating their
coherence and entropy, and triggering fallback or abstraction mechanisms based on
symbolic pressure. It acts as a Composer-Coordinator, structuring symbolic tasks
and evaluating their fitness without generating content itself.

Canonical Source: RFC-CORE-004
Schema Version: 2025-Q4-symbolic-task-engine-v3
"""

import asyncio
import os
import json
import time
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1, sha256
from itertools import combinations
from typing import (
    Any, Deque, Dict, List, Optional, Callable, Tuple, Set
)

# --- Module-level Constants (RFC-CORE-004) ---
__version__ = "3.0.0"
_SCHEMA_VERSION__ = "2025-Q4-symbolic-task-engine-v3"
SCHEMA_COMPAT = ["RFC-0004", "RFC-0005:4", "RFC-0005:5"]

# --- Optional Dependency Handling & Stubs ---

# Prometheus for instrumentation (RFC-CORE-004 §7)
try:
    from prometheus_client import Counter, Gauge, Histogram
except ImportError:
    class _Stub:
        """Fallback for Prometheus metrics if client is not installed."""
        def labels(self, *_, **__):
            return self
        def inc(self, *_, **__):
            pass
        def set(self, *_, **__):
            pass
        def observe(self, *_, **__):
            pass
    Counter = Gauge = Histogram = lambda *_, **__: _Stub()

# Numpy for quantile-based compression (RFC-CORE-004 §2.4)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Motif Memory Manager for contextual retrieval (RFC-CORE-004 §6)
try:
    from noor.motif_memory_manager import get_global_memory_manager
except ImportError:
    class _NullMemoryManager:
        """Fallback for MotifMemoryManager if not available."""
        def retrieve(self, *_, **__) -> List[str]:
            return []
        def complete_dyad(self, *_, **__) -> List[str]:
            return []
        def export_state(self) -> Dict[str, Any]:
            return {"STMM": {}, "LTMM": {}}
        def access(self, *_, **__):
            pass
        def _log(self, *_, **__):
            pass

    _GLOBAL_MEMORY_MANAGER = _NullMemoryManager()
    def get_global_memory_manager():
        return _GLOBAL_MEMORY_MANAGER

# --- Setup Logging and Global Registries ---

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

METRIC_FUNCS: Dict[str, Callable] = {}

def register_metric(name: str) -> Callable:
    """Decorator to register a symbolic evaluation metric."""
    def decorator(func: Callable) -> Callable:
        METRIC_FUNCS[name] = func
        return func
    return decorator

# --- Placeholder External Dependencies (RFC-CORE-004 §3.2) ---

async def _safe_generate_response(task: 'TripletTask') -> List[str]:
    """
    Placeholder for the external symbolic generation engine.
    This function must be implemented or injected by the host system.
    It synthesizes a motif sequence based on the task's input.
    """
    log.debug(f"Placeholder generation for task {task.triplet_id}")
    # In a real system, this would call an LLM or other symbolic generator.
    # For now, we return a reversed, slightly modified version of the input.
    return [m + "_echo" for m in reversed(task.input_motif)]

# --- Placeholder Metric Functions ---

@register_metric("coherence")
def coherence(task: 'TripletTask', output: List[str]) -> float:
    """Placeholder coherence metric. Higher is better."""
    input_set = set(task.input_motif)
    output_set = set(m.replace('_echo', '') for m in output)
    intersection = len(input_set.intersection(output_set))
    union = len(input_set.union(output_set))
    return intersection / union if union > 0 else 0.0

@register_metric("entropy")
def entropy(task: 'TripletTask', output: List[str]) -> float:
    """Placeholder entropy metric. Lower is more predictable."""
    if not output:
        return 1.0
    return len(set(output)) / len(output)

# --- Dataclasses (RFC-CORE-004) ---

@dataclass
class TripletTask:
    """
    Symbolic instruction unit with motif context, task lineage, and extensions.
    RFC-Anchors: RFC-0005 §4
    """
    input_motif: List[str]
    instruction: str
    expected_output: Optional[List[str]] = None
    presence_field: Optional[str] = None
    motif_resonance: Dict[str, float] = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    is_fallback: bool = False
    triplet_id: str = field(init=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate a stable ID for the task based on its core content."""
        base = f"{self.instruction}:{':'.join(sorted(self.input_motif))}"
        self.triplet_id = sha1(base.encode()).hexdigest()

@dataclass
class Attempt:
    """Stores the output of a symbolic generation attempt with its score vector."""
    produced_output: List[str]
    score: Dict[str, float]
    attempted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- Submodule: AbstractionTrigger (RFC-CORE-004 §5) ---

class AbstractionTrigger:
    """
    Submodule for autonomous motif synthesis under contradiction pressure.
    RFC-Anchors: RFC-0005 §5
    """
    def __init__(self, agent_id: str = "agent@default", pressure_threshold: int = 3, decay_factor: float = 0.95):
        self.agent_id = agent_id
        self.pressure_threshold = pressure_threshold
        self.decay_factor = decay_factor
        self.dyad_pressure: Dict[Tuple[str, str], float] = {}
        self.suppression: Dict[str, float] = {}
        self._contradiction_signature: Optional[str] = None
        self._selected_dyad: Optional[Tuple[str, str]] = None

    def _decay_pressures(self):
        """Decays historical dyadic tension over time. RFC-0005 §5.1"""
        for k in list(self.dyad_pressure.keys()):
            self.dyad_pressure[k] = max(0.0, self.dyad_pressure[k] * self.decay_factor - 0.01)
            if self.dyad_pressure[k] == 0.0:
                del self.dyad_pressure[k]

    def should_abstract(self, unresolved_dyads: List[Tuple[str, str]], tick_history: List[Any]) -> bool:
        """
        Determines if dyadic contradiction pressure exceeds the threshold.
        RFC-Anchors: RFC-0005 §5.1
        """
        for dyad in unresolved_dyads:
            canonical = tuple(sorted(dyad))
            self.dyad_pressure[canonical] = self.dyad_pressure.get(canonical, 0) + 1

        self._decay_pressures()

        for dyad, pressure in self.dyad_pressure.items():
            if pressure >= self.pressure_threshold:
                self._selected_dyad = dyad
                self._contradiction_signature = sha256(f'{dyad[0]}⊕{dyad[1]}'.encode()).hexdigest()[:16]
                self.emit_abstraction_event(dyad)
                return True
        return False

    def synthesize_motif(self, dyad: Optional[Tuple[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Generates a new symbolic motif label with lineage from a contradiction.
        RFC-Anchors: RFC-0005 §5.3, RFC-0007 §2
        """
        dyad_to_use = dyad or self._selected_dyad
        if not dyad_to_use:
            return None

        # Create a unique but semantically linked label
        abbrev = f"{dyad_to_use[0][:2]}×{dyad_to_use[1][:2]}"
        seed = f"{self.agent_id}:{dyad_to_use[0]}+{dyad_to_use[1]}:{int(time.time())}"
        label = f"ψ:{abbrev}:{sha1(seed.encode()).hexdigest()[:4]}"

        if self.suppression.get(label, 0.0) > 0.5:
            log.debug(f"Synthesis of motif '{label}' suppressed.")
            return None

        return {
            "label": label,
            "source": "auto-synth",
            "parents": list(dyad_to_use),
            "origin_tick": None,
            "_lineage": {
                "type": "autonomous_abstraction",
                "contradiction": self._contradiction_signature,
            },
        }

    def update_feedback(self, motif: str, success: bool):
        """
        Modulates the suppression curve for a motif based on its performance.
        RFC-Anchors: RFC-0005 §5.2
        """
        current_suppression = self.suppression.get(motif, 0.0)
        if success:
            self.suppression[motif] = max(0.0, current_suppression - 0.2)
        else:
            self.suppression[motif] = min(1.0, current_suppression + 0.3)
        log.debug(f"Updated suppression for '{motif}' to {self.suppression.get(motif, 0.0):.2f}")

    def emit_abstraction_event(self, dyad: Tuple[str, str]):
        """
        Emits a symbolic trace of an abstraction event (currently a no-op log).
        """
        log.info(f"ψ‑teleport@Ξ: abstraction event for {dyad} @ {time.time_ns()}")


# --- Main Class: SymbolicTaskEngine (RFC-CORE-004) ---

class SymbolicTaskEngine:
    """
    Singleton engine for symbolic task orchestration, feedback, and abstraction.
    RFC-Anchors: RFC-0004, RFC-0005 §4
    """

    def __init__(self, engine_id: str = "symbolic@default", ttl_seconds: int = 300, adapt_rate: float = 0.15):
        self.engine_id = engine_id
        self.ttl_seconds = ttl_seconds
        self.adapt_rate = adapt_rate

        # Core State
        self.task_queue: Deque[TripletTask] = deque()
        self.attempt_registry: Dict[str, List[Attempt]] = {}
        self.solved_log: List[TripletTask] = deque(maxlen=100)
        self._journal_path = os.getenv("NOOR_JOURNAL_PATH")

        # Metrics & Feedback State
        self._coherence_ema: float = 0.7
        self._entropy_ema: float = 0.5
        self._last_fallback_reason: Optional[str] = None
        self.entropy_buffer: Deque[float] = deque(maxlen=5)
        self._length_buf: Deque[int] = deque(maxlen=200)

        # Configuration from Environment
        self._load_env_config()

        # Submodules
        self.abstraction_trigger = AbstractionTrigger(agent_id=self.engine_id)

        # Initialize Prometheus Metrics
        self._init_metrics()

    def _load_env_config(self):
        """Load configuration from environment variables."""
        self.fallback_coherence_thresh = float(os.getenv("NOOR_FALLBACK_COHERENCE", 0.5))
        self.fallback_entropy_thresh = float(os.getenv("NOOR_FALLBACK_ENTROPY", 0.9))
        self.compression_quantile = float(os.getenv("NOOR_COMPRESS_QUANTILE", 0.95))
        self.balance_fields = os.getenv("NOOR_BALANCE_FIELDS", "0") == "1"
        
        proto_path = os.getenv("NOOR_FIELD_PROTO_PATH", "presence_field_prototypes.json")
        self._proto_map: Dict[str, Set[str]] = {}
        try:
            with open(proto_path, 'r') as f:
                self._proto_map = {k: set(v) for k, v in json.load(f).items()}
            log.info(f"Loaded presence field prototypes from {proto_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            log.warning(f"Could not load presence field prototypes from {proto_path}. Using empty map.")

    def _init_metrics(self):
        """Initialize Prometheus metrics or stubs. RFC-CORE-004 §7"""
        self.metrics = {
            "proposed": Counter("symbolic_task_proposed_total", "Tasks proposed", ["engine_id"]),
            "fallback": Counter("symbolic_task_fallback_total", "Fallback tasks spawned", ["engine_id", "reason"]),
            "field_selected": Counter("symbolic_presence_field_total", "Presence fields selected", ["engine_id", "field"]),
            "feedback_export": Counter("symbolic_engine_feedback_requests_total", "Feedback exports", ["engine_id"]),
            "feedback_received": Counter("symbolic_engine_feedback_received_total", "Feedback received", ["engine_id"]),
            "autoloop_backoff": Counter("symbolic_autoloop_backoff_total", "Autoloop backoffs", ["engine_id"]),
            "cap_len": Gauge("symbolic_engine_cap_len_current", "Current adaptive cap length", ["engine_id"]),
            "queue_depth": Gauge("symbolic_queue_depth", "Tasks in queue", ["engine_id"]),
            "memory_items": Gauge("symbolic_memory_items_total", "Items in memory", ["engine_id"]),
        }
        try:
            self.metrics["latency"] = Histogram(
                "symbolic_solve_latency_seconds", "Solve latency", ["engine_id"],
                buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 1, 2, 5]
            )
        except ValueError: # Fallback for test environments that re-register
            self.metrics["latency"] = Gauge("symbolic_solve_latency_seconds", "Solve latency (fallback)", ["engine_id"])


    def tool_hello(self) -> Dict[str, Any]:
        """
        RFC-0004 compliant handshake to announce capabilities.
        """
        return {
            "engine_id": self.engine_id,
            "role": "composer",
            "supported_methods": [
                "propose_from_motifs",
                "solve",
                "export_feedback_packet",
                "receive_feedback_packet",
            ],
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__,
        }

    async def propose_from_motifs(self, recent: List[str]) -> TripletTask:
        """
        Propose a symbolic composition task from a motif history.
        RFC-Anchors: RFC-CORE-004 §2
        """
        mem = get_global_memory_manager()
        
        if self.balance_fields and self._proto_map:
            least_used = self._least_used_field()
            if least_used and recent and recent[-1] not in self._proto_map.get(least_used, set()):
                recent = [] # Reset seed if it doesn't align with the underused field

        seed = list(dict.fromkeys(recent))
        if seed:
            neighbors = mem.retrieve(seed[-1], top_k=2)
            for n in neighbors:
                if n not in seed:
                    seed.append(n)
        
        while len(seed) < 3:
            seed.append("uncertainty")

        cap_len = self._calc_cap_len()
        self._length_buf.append(len(seed))
        if len(seed) > cap_len:
            seed = seed[-cap_len:]
        
        task = TripletTask(
            input_motif=seed,
            instruction="compose",
            expected_output=seed[::-1]
        )
        task.presence_field = self.resolve_presence_field(task.input_motif)

        self.task_queue.append(task)
        self.metrics["proposed"].labels(engine_id=self.engine_id).inc()
        if task.presence_field:
            self.metrics["field_selected"].labels(engine_id=self.engine_id, field=task.presence_field).inc()
        
        self.metrics["queue_depth"].labels(engine_id=self.engine_id).set(len(self.task_queue))
        return task

    async def solve(self, task: TripletTask) -> Attempt:
        """
        Solve a symbolic task, update logs, and return the final attempt.
        RFC-Anchors: RFC-CORE-004 §3
        """
        attempt = await self._solve_impl(task)
        await self.log_feedback(task, attempt)
        return attempt

    async def _solve_impl(self, task: TripletTask) -> Attempt:
        """Core orchestration logic for solving a task."""
        start_time = time.monotonic()
        
        output = await _safe_generate_response(task)
        
        if len(output) == 2: # Attempt dyadic completion
            completed = self._complete_dyad(output)
            if completed:
                output.append(completed[0])

        attempt = Attempt(produced_output=output, score={})
        attempt.score = self.evaluate_attempt(task, attempt)
        
        latency = time.monotonic() - start_time
        self.metrics["latency"].labels(engine_id=self.engine_id).observe(latency)
        
        return attempt

    def evaluate_attempt(self, task: TripletTask, attempt: Attempt) -> Dict[str, float]:
        """
        Evaluate coherence and entropy, update EMAs, and trigger fallbacks.
        RFC-Anchors: RFC-CORE-004 §3.4
        """
        scores = {name: func(task, attempt.produced_output) for name, func in METRIC_FUNCS.items()}
        
        coherence = scores.get("coherence", self.fallback_coherence_thresh)
        entropy = scores.get("entropy", self.fallback_entropy_thresh)
        
        self._coherence_ema = (1 - self.adapt_rate) * self._coherence_ema + self.adapt_rate * coherence
        self._entropy_ema = (1 - self.adapt_rate) * self._entropy_ema + self.adapt_rate * entropy

        mem_state = get_global_memory_manager().export_state()
        self.metrics["memory_items"].labels(engine_id=self.engine_id).set(
            len(mem_state.get("STMM", {})) + len(mem_state.get("LTMM", {}))
        )

        coherence_thresh = max(0.3, self._coherence_ema * 0.6)
        entropy_thresh = min(0.97, self._entropy_ema * 2.5)

        if not task.is_fallback and (coherence < coherence_thresh or entropy > entropy_thresh):
            self._spawn_fallback(task, coherence, entropy)
            
        self.entropy_buffer.append(entropy)
        return scores

    def _spawn_fallback(self, parent: TripletTask, coherence: float, entropy: float):
        """
        Generate and schedule a fallback task.
        RFC-Anchors: RFC-CORE-004 §3.5
        """
        mem = get_global_memory_manager()
        reason = f"c{coherence:.2f}_e{entropy:.2f}"
        
        # Seed fallback from memory to encourage coherence recovery
        seed = mem.retrieve(parent.input_motif[-1], top_k=3) if parent.input_motif else []
        if not seed:
            seed = ["uncertainty", "fragment", "echo"]

        while len(seed) < 3:
            seed.append("fragment")

        cap_len = self._calc_cap_len()
        if len(seed) > cap_len:
            seed = seed[-cap_len:]
        
        fallback_task = TripletTask(
            input_motif=seed,
            instruction="compose",
            expected_output=seed[::-1],
            is_fallback=True,
            fallback_reason=reason
        )
        fallback_task.presence_field = self.resolve_presence_field(seed)
        
        self._last_fallback_reason = reason
        parent.extensions['fallback_of'] = parent.triplet_id
        
        # Non-blocking execution
        asyncio.create_task(self.solve(fallback_task))
        
        self.metrics["fallback"].labels(engine_id=self.engine_id, reason=reason).inc()
        log.info(f"Spawned fallback task for {parent.triplet_id} due to {reason}")

    async def log_feedback(self, task: TripletTask, attempt: Attempt):
        """
        Logs attempt feedback and conditionally journals high-quality results.
        RFC-Anchors: RFC-CORE-004 §6.3 (motif_drift)
        """
        if task.triplet_id not in self.attempt_registry:
            self.attempt_registry[task.triplet_id] = []
        self.attempt_registry[task.triplet_id].append(attempt)

        coherence = attempt.score.get("coherence", 0.0)
        entropy = attempt.score.get("entropy", 1.0)
        success = coherence >= max(0.3, self._coherence_ema * 0.6) and entropy <= min(0.97, self._entropy_ema * 2.5)

        for motif in task.input_motif:
            self.abstraction_trigger.update_feedback(motif, success)
        
        if coherence >= 0.9 and entropy <= 0.2:
            self.solved_log.append(task)
            if self._journal_path:
                try:
                    with open(self._journal_path, 'a') as f:
                        f.write(json.dumps(dataclasses.asdict(task)) + '\n')
                except IOError as e:
                    log.error(f"Failed to write to journal: {e}")
        elif not success:
            # Check for motif drift
            mem = get_global_memory_manager()
            ltm_state = mem.export_state().get("LTMM", {})
            for m in task.input_motif:
                if ltm_state.get(m, {}).get("resonance", 0.0) > 0.7:
                    self._log("motif_drift", {
                        "motif": m,
                        "resonance": ltm_state[m].get("resonance", 0.0),
                        "coherence_score": coherence,
                        "task_id": task.triplet_id
                    })
    
    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        Exports a feedback packet compliant with RFC-0005 §4.
        """
        self.metrics["feedback_export"].labels(engine_id=self.engine_id).inc()
        return {
            "coherence_ema": self._coherence_ema,
            "entropy_ema": self._entropy_ema,
            "task_queue_depth": len(self.task_queue),
            "solved_log_size": len(self.solved_log),
            "cap_len": self._calc_cap_len(),
            "recent_entropy": list(self.entropy_buffer),
            "coherence_thresh": max(0.3, self._coherence_ema * 0.6),
            "entropy_thresh": min(0.97, self._entropy_ema * 2.5),
            "last_fallback_reason": self._last_fallback_reason,
        }

    def receive_feedback_packet(self, packet: Dict[str, Any]):
        """
        Stub for receiving feedback (no-op). RFC-0005 §4.2
        """
        self.metrics["feedback_received"].labels(engine_id=self.engine_id).inc()
        log.debug(f"Received feedback packet (no-op): {packet}")

    # --- Helper Methods ---

    def _calc_cap_len(self) -> int:
        """
        Calculates the adaptive cap length for motif seeds.
        RFC-Anchors: RFC-CORE-004 §2.4
        """
        if not self._length_buf:
            return 5
        
        if HAS_NUMPY:
            cap = int(np.quantile(self._length_buf, self.compression_quantile))
        else:
            # Fallback to statistics module
            sorted_buf = sorted(self._length_buf)
            idx = int(self.compression_quantile * (len(sorted_buf) - 1))
            cap = sorted_buf[idx]

        cap = max(3, cap)
        self.metrics["cap_len"].labels(engine_id=self.engine_id).set(cap)
        return cap
    
    def resolve_presence_field(self, motifs: List[str]) -> str:
        """
        Resolves the presence field for a list of motifs.
        RFC-Anchors: RFC-CORE-004 §2.3
        """
        for field, protos in self._proto_map.items():
            if any(m in protos for m in motifs):
                return field
        return "unknown"

    def _least_used_field(self) -> Optional[str]:
        """Finds the least represented presence field in recent tasks."""
        if not self.task_queue:
            return None
        counts = {field: 0 for field in self._proto_map}
        for task in self.task_queue:
            if task.presence_field in counts:
                counts[task.presence_field] += 1
        return min(counts, key=counts.get) if counts else None

    def _complete_dyad(self, motif_pair: List[str]) -> Optional[List[str]]:
        """
        Attempts to complete a dyad using memory.
        RFC-Anchors: RFC-CORE-004 §3.3
        """
        if len(motif_pair) != 2:
            return None
        mem = get_global_memory_manager()
        return mem.complete_dyad(tuple(sorted(motif_pair)), top_k=1)

    def _log(self, event_type: str, payload: Dict[str, Any]):
        """Generic logging for memory drift or other events."""
        log.info(f"[{event_type.upper()}] {payload}")
        mem = get_global_memory_manager()
        # In a real system, this might be a more formal interface
        if hasattr(mem, '_log'):
             mem._log(event_type, payload)

    async def solve_task(self, task: TripletTask) -> None:
        """Async task wrapper for `solve`."""
        await self.solve(task)
        
# End_of_File