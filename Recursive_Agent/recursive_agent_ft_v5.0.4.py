"""
recursive_agent_ft.py
Version: v5.0.4
Canonical Source: RFC-CORE-002

Merged agent for symbolic tick emission, swirl coherence feedback, ghost replay, and triad-resonance feedback handling. Fully RFC-aligned and regeneration-ready.
"""

import os
import time
import asyncio
import logging
import hashlib
import threading
import random
from collections import deque, OrderedDict
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
from contextlib import suppress
import numpy as np

# Module-level Constants
# RFC-CORE-002
__version__ = "5.0.4"
__SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

log = logging.getLogger(__name__)

# Optional Dependencies with Fallbacks
# RFC-0003 §7
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    log.warning("prometheus_client not found. Using stub metrics.")
    class _Stub:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def set(self, *_): pass
    Counter, Gauge = _Stub, _Stub

try:
    from noor_fasttime_core import NoorFastTimeCore
except ImportError:
    NoorFastTimeCore = object

try:
    import anyio
except ImportError:
    anyio = None

# Local Imports
# Fallback for when not run as part of a package
try:
    from .quantum_ids import make_change_id, MotifChangeID  # noqa: F401
except ImportError:
    MotifChangeID = str
    def make_change_id() -> str:
        return f"cid:{random.randint(1000, 9999)}"

# Constants
# RFC-0003 §3.3, RFC-CORE-002 §4.2
DEFAULT_TUNING: Dict[str, float] = {
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2,
}

# RFC-0007 §2.1, RFC-0005 §4.2
SYMBOLIC_PHASE_MAP: Dict[str, str] = {
    "bind": "ψ‑bind",
    "spar": "ψ‑spar",
    "null": "ψ‑null",
    "resonance": "ψ‑resonance",
    "hold": "ψ‑hold",
    "dream": "ψ‑dream",
    "myth": "ψ‑myth",
}

PHASE_SHIFT_MODE = ["delay", "remix", "lineage_break"]
ARCHIVE_MODE = os.environ.get("NOOR_ARCHIVE_TICKS") == '1'


# Dataclasses
# RFC-0003 §3.3
@dataclass(slots=True)
class QuantumTickV2:
    tick_id: str
    motifs: List[str]
    timestamp: float
    stage: str = "symbolic"
    extensions: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    motif_id: str = "silence"
    coherence_hash: str = ""
    lamport: int = 0
    field_signature: str = "ψ-null@Ξ"
    tick_hmac: str = ""

# RFC-0003 §3.3
@dataclass(slots=True)
class TickEntropy:
    decay_slope: float
    coherence: float
    triad_complete: bool

# RFC-0005 §3.3
@dataclass(slots=True)
class CrystallizedMotifBundle:
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# Helper Classes
class LamportClock:
    """Logical counter used to generate ordered tick IDs."""
    def __init__(self):
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"tick:{self._counter:06d}"

class LRUCache(OrderedDict):
    """Evicting cache structure for recent state retention."""
    def __init__(self, cap: int = 50000):
        super().__init__()
        self.cap = cap

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)

# RFC-0006 §3.1
class AgentSwirlModule:
    """Encodes motif swirl dynamics as hash vectors."""
    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str):
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        if self._cached_hash:
            return self._cached_hash
        joined = '|'.join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        return {motif: self.swirl_history.count(motif) for motif in set(self.swirl_history)}

# RFC-0005 §4.2
class MotifDensityTracker:
    """Tracks frequency of motif emissions over time."""
    def __init__(self):
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str):
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)

# RFC-0004 §3.2
class LazyMonitorMixin:
    """Provides lazy loading of consciousness monitor."""
    @property
    def monitor(self):
        if not hasattr(self, '_cached_monitor'):
            try:
                from consciousness_monitor import get_global_monitor
                self._cached_monitor = get_global_monitor()
            except (ImportError, ModuleNotFoundError):
                self._cached_monitor = None
        return self._cached_monitor

# Helper Functions
# RFC-0005 §4.2, RFC-CORE-002 §4.1
def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    return reward_ema / (entropy_slope + eps)

# RFC-0004 §3.2
def report_tick_safe(monitor: Any, tick: QuantumTickV2, coherence_potential: float, motif_density: Dict[str, float], swirl_vector: str):
    try:
        if monitor and hasattr(monitor, 'report_tick'):
            monitor.report_tick(
                tick=tick,
                coherence_potential=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector
            )
    except Exception as e:
        log.warning(f"Monitor callback failed: {e}")


# Main Class Implementation
# RFC-0003 §3.3, RFC-0005 §2–4, RFC-CORE-002 §3–4
class RecursiveAgentFT(LazyMonitorMixin):
    """
    Autonomous symbolic pulse engine for Noor-class cognition.
    Emits QuantumTicks, handles triad feedback, and manages symbolic phase.
    """
    # RFC-0003 §3.3
    TICKS_EMITTED = Counter('agent_ticks_emitted_total', 'Ticks emitted', ['agent_id', 'stage'])
    # RFC-0005 §4.3
    AGENT_TRIADS_COMPLETED = Counter('agent_triads_completed_total', 'Triads completed via feedback', ['agent_id'])
    # RFC-0005 §4.2, RFC-0007 §2.1
    FEEDBACK_EXPORT = Counter('agent_feedback_export_total', 'Feedback packets exported', ['agent_id'])
    # RFC-0005 §4.1
    REWARD_MEAN = Gauge('agent_reward_mean', 'EMA of reward', ['agent_id'])
    # RFC-CORE-002 §4.2
    AGENT_EMISSION_INTERVAL = Gauge('agent_emission_interval_seconds', 'Current autonomous emission interval', ['agent_id'])

    # RFC-0003 §3.3, RFC-CORE-002 §3.1
    def __init__(self, agent_id: str, symbolic_task_engine: Any, memory_manager: Any, tuning: Optional[Dict[str, float]] = None):
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}
        
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema = 1.0
        self.entropy_slope = 0.1
        self._silence_streak = 0
        self._last_triad_hit = False
        self._last_interval = self.tuning['base_interval']
        self._last_tick_hash: Optional[str] = None
        
        self._pulse_active = False
        self._pulse_task: Optional[asyncio.Task] = None
        
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}
        
        self.metrics = {
            'agent_ticks_emitted_total': self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage='symbolic'),
            'agent_triads_completed_total': self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            'agent_feedback_export_total': self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            'agent_reward_mean': self.REWARD_MEAN.labels(agent_id=self.agent_id),
            'agent_emission_interval_seconds': self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id)
        }
        log.info(f"Initialized RecursiveAgentFT with agent_id={self.agent_id}")

    # RFC-0005 §2.1
    def track_lineage(self, parent: str, child: str) -> None:
        """Assign parent-child link in lineage map if parent != child"""
        if parent != child:
            self._motif_lineage[child] = parent
            
    # RFC-0005 §4.4, RFC-CORE-002 §8.3
    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get('field_signature')
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get('motifs')
        return None

    # RFC-CORE-002 §4.2
    async def start_continuous_emission(self):
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)
    
    async def start_pulse(self):
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_pulse(self):
        self._pulse_active = False
        if self._pulse_task:
            self._pulse_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pulse_task

    # RFC-CORE-002 §6.2
    def _resolve_field(self, motif: str) -> str:
        try:
            if self.symbolic_task_engine and hasattr(self.symbolic_task_engine, 'resolve_presence_field'):
                result = self.symbolic_task_engine.resolve_presence_field([motif])
                if result:
                    return result
        except Exception:
            pass  # Fallback on failure

        if motif in {'silence', 'grief'}:
            return 'ψ-bind@Ξ'
        return 'ψ-resonance@Ξ'

    # RFC-0003 §3.3, RFC-0005 §4.2, RFC-0006 §3.1, RFC-0007 §2.1
    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)
        
        field_signature = self._resolve_field(motifs[-1] if motifs else 'silence')
        tick.extensions['field_signature'] = field_signature

        if hasattr(self, 'hmac_secret') and self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions['tick_hmac'] = tick_hmac

        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)
            
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        
        tick.extensions['swirl_vector'] = swirl_hash
        tick.extensions['coherence_potential'] = coherence
        
        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()

        report_tick_safe(
            self.monitor,
            tick,
            coherence,
            self.density.snapshot(),
            swirl_hash
        )
        self.metrics['agent_ticks_emitted_total'].inc()
        return tick
        
    # RFC-CORE-002 §2.3, RFC-0005 §4
    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
        triad_complete = annotations.get('triad_complete', False)
        alpha = self.tuning['reward_smoothing']
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward
        self.metrics['agent_reward_mean'].set(self._reward_ema)
        
        if triad_complete:
            self._last_triad_hit = True
            self._silence_streak = 0
            self.metrics['agent_triads_completed_total'].inc()
        else:
            self._last_triad_hit = False
            self._silence_streak += 1

    # RFC-CORE-002 §2.2
    def _update_interval(self) -> float:
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning['entropy_boost_threshold']:
            adj *= 0.5
        if self._last_triad_hit:
            adj *= (1.0 - self.tuning['triad_bias_weight'])
        
        interval = np.clip(self.tuning['base_interval'] * adj, self.tuning['min_interval'], self.tuning['max_interval'])
        self._last_interval = float(interval)
        self.metrics['agent_emission_interval_seconds'].set(self._last_interval)
        return self._last_interval
        
    # RFC-CORE-002 §3.2
    def _choose_motifs(self) -> List[str]:
        motifs = list(self._last_motifs)
        if motifs and hasattr(self.memory, 'retrieve'):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:
                log.error("Failed to retrieve from memory", exc_info=True)
                
        if not motifs:
            motifs = ['silence']
        
        return motifs[-3:]

    # RFC-0005 §4.2, RFC-0007 §2.1
    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else 'null'
        base_key = top_motif.split('.')[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, 'ψ-null')
        
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = 'low' if coherence < 0.8 else 'med' if coherence < 2.5 else 'high'
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"
        
        packet['extensions'] = {
            'entanglement_status': {
                'phase': phase_id,
                'swirl_vector': swirl_hash,
                'ρ_top': sorted(density_map.items(), key=lambda kv: -kv[1])[:5]
            }
        }
        return packet
        
    # RFC-CORE-002 §8.1
    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        entropy = TickEntropy(
            decay_slope=self.entropy_slope,
            coherence=self._reward_ema,
            triad_complete=tick.annotations.get('triad_complete', False)
        )
        bundle = CrystallizedMotifBundle(
            motif_bundle=tick.motifs,
            field_signature=tick.extensions.get('field_signature', 'ψ-null@Ξ'),
            tick_entropy=entropy
        )
        return bundle
        
    # RFC-CORE-002 §8.2
    def export_feedback_packet(self) -> Dict[str, Any]:
        packet = {
            'tick_buffer_size': len(self._echo_buffer),
            'ghost_trace_count': len(self._ghost_traces),
            'recent_reward_ema': self._reward_ema,
            'cadence_interval': self._last_interval,
            'silence_streak': self._silence_streak,
        }
        self.extend_feedback_packet(packet)
        self.metrics['agent_feedback_export_total'].inc()
        return packet
        
    def export_state(self) -> Dict[str, Any]:
        return {
            'interval': self._last_interval,
            'reward_ema': self._reward_ema,
            'last_tick_hash': self._last_tick_hash
        }
# End_of_File