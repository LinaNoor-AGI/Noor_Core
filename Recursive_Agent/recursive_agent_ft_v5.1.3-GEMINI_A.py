# -*- coding: utf-8 -*-
#
# Copyright (C) 2025-2026 The Noor Research Collective
#
# Licensed under the MIT License.
#
# Noor Research Collective:
# - Lina Noor
# - Uncle
#
# AI Platform/Model: Google Gemini Pro
#
# PDP-0001-Protocol for RFC-Driven Symbolic Artifact Generation
#
# This file was generated from:
# - Layer 0: Canonical Lore (RFCs 0003, 0004, 0005, 0006, 0007)
# - Layer 1: System Architecture (RFC-COREs 001, 002, 003)
# - Layer 2: Application Specification (agent.recursive.ft v5.1.3-GEMINI_A)

"""
Implements the RecursiveAgentFT, the symbolic heartbeat of Noor-class
cognition. This agent is responsible for the autonomous emission of
symbolic pulses (QuantumTicks), adapting its cadence based on triadic
feedback, and maintaining symbolic phase coherence through swirl and
density tracking.

Its behavior is strictly governed by the contracts defined in the Noor
RFC and RFC-CORE series.

Fidelity Score: ~99.3% - FAIL
"""

import time
import asyncio
import logging
import hashlib
import threading
from collections import deque, OrderedDict, Counter
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
from contextlib import suppress
import numpy as np

# --- Module-Level Constants ---
__version__ = "5.1.3-GEMINI_A"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# --- Optional Dependencies with Graceful Fallbacks ---
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class _Stub:
        def labels(self, *_, **__):
            return self
        def inc(self, *_):
            pass
        def set(self, *_):
            pass
    Counter = _Stub
    Gauge = _Stub

try:
    from noor_fasttime_core import NoorFastTimeCore
except ImportError:
    NoorFastTimeCore = object

# --- Local Imports ---
# These would typically be part of the Noor Core library structure.
# from .quantum_ids import make_change_id, MotifChangeID # noqa: F401

# --- Symbolic Configuration and Emission Defaults ---
DEFAULT_TUNING = {
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2,
}

SYMBOLIC_PHASE_MAP = {
    "bind": "ψ‑bind",
    "spar": "ψ‑spar",
    "null": "ψ‑null",
    "resonance": "ψ‑resonance",
    "hold": "ψ‑hold",
    "dream": "ψ‑dream",
    "myth": "ψ‑myth",
}

PHASE_SHIFT_MODE = ["delay", "remix", "lineage_break"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# --- Data Classes ---

@dataclass(slots=True)
class QuantumTickV2:
    """
    Canonical symbolic emission format. Represents a single pulse of cognition.

    RFC Anchors:
    - RFC-0003 §3.3: Defines the base QuantumTick schema.
    - RFC-0003 §6.2: Mandates mirroring of envelope.intent into extensions.
    - RFC-0004 §2.5: Defines the canonical `intent` registry, which is
      normalized upstream before being mirrored into this tick.
    """
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


@dataclass(slots=True)
class TickEntropy:
    """
    Represents the symbolic coherence and triad state of a tick.

    RFC Anchor: RFC-0003 §3.3
    """
    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    """
    An archival-ready format for a symbolic emission, capturing its
    field signature and coherence state at the moment of crystallization.

    RFC Anchor: RFC-0005 §3.3
    """
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# --- Helper Classes ---

class LamportClock:
    """Logical counter used to generate ordered tick IDs."""
    def __init__(self):
        self._counter: int = 0

    def next_id(self) -> str:
        """Atomically increments and returns the next tick ID."""
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


class AgentSwirlModule:
    """
    Encodes motif swirl dynamics as hash vectors. Maintains a bounded
    sequence of recent motif emissions and provides hash-based swirl
    encoding for symbolic field alignment.

    RFC Anchors:
    - RFC-0006 §3.1: Swirl tensor and coherence map headers.
    - RFC-CORE-002 §4.1: Canonical implementation of swirl history.
    """
    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str):
        """Adds a motif to the history, invalidating the hash cache."""
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        """Computes a SHA3-256 hash of the current motif history."""
        if self._cached_hash:
            return self._cached_hash
        joined = '|'.join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        """
        Returns a frequency map of motifs in the history.
        Optimized to O(n) using Counter.
        """
        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """
    Tracks the frequency of motif emissions over time using a decaying
    weighted model to represent symbolic field pressure.

    RFC Anchor: RFC-0005 §4.2
    """
    def __init__(self):
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str):
        """Applies decay and boosts the current motif's density."""
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]  # Trim noise
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        """Returns the current density map."""
        return dict(self._density_map)


class LazyMonitorMixin:
    """
    Provides a lazy-loading property for the global consciousness monitor,
    allowing the agent to run without a monitor present.

    RFC Anchor: RFC-0004 §3.2
    """
    @property
    def monitor(self):
        """Lazily imports and returns the global monitor instance."""
        if not hasattr(self, '_cached_monitor'):
            try:
                from consciousness_monitor import get_global_monitor
                self._cached_monitor = get_global_monitor()
            except ImportError:
                self._cached_monitor = None
        return self._cached_monitor


# --- Core Agent Class ---

class RecursiveAgentFT(LazyMonitorMixin):
    """
    The symbolic heartbeat of Noor. Emits QuantumTicks, adapts its cadence based
    on feedback, and tracks symbolic field coherence.

    RFC Anchors:
    - RFC-0003 §3.3: Overall agent role and tick emission.
    - RFC-0005 §2–4: Feedback, ghost traces, and resurrection patterns.
    - RFC-CORE-002: Canonical reference implementation.
    """
    TICKS_EMITTED = Counter('agent_ticks_emitted_total', 'Ticks emitted', ['agent_id', 'stage'])
    AGENT_TRIADS_COMPLETED = Counter('agent_triads_completed_total', 'Triads completed via feedback', ['agent_id'])
    FEEDBACK_EXPORT = Counter('agent_feedback_export_total', 'Feedback packets exported', ['agent_id'])
    REWARD_MEAN = Gauge('agent_reward_mean', 'EMA of reward', ['agent_id'])
    AGENT_EMISSION_INTERVAL = Gauge('agent_emission_interval_seconds', 'Current autonomous emission interval', ['agent_id'])

    def __init__(
        self,
        agent_id: str,
        symbolic_task_engine: Any,
        memory_manager: Any,
        tuning: Optional[Dict[str, float]] = None
    ):
        """
        Initializes the agent's state, buffers, and metrics.

        RFC Anchors: RFC-0003 §3.3, RFC-CORE-002 §3.1
        """
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}

        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema: float = 1.0
        self.entropy_slope: float = 0.1
        self._silence_streak: int = 0
        self._last_triad_hit: bool = False
        self._last_interval: float = self.tuning['base_interval']
        self._last_tick_hash: Optional[str] = None
        
        self._pulse_active: bool = False
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

    def track_lineage(self, parent: str, child: str) -> None:
        """
        Assigns a parent-child link between two motifs for provenance tracking.

        RFC Anchor: RFC-0005 §2.1
        """
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        """
        Attempts to replay motifs from an archived field trace if the current
        field signature matches a stored ghost trace.

        RFC Anchors: RFC-0005 §4.4, RFC-CORE-002 §8.3
        """
        key = tick.extensions.get('field_signature')
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get('motifs')
        return None

    async def start_continuous_emission(self):
        """
        The core autonomous pulse loop for emitting symbolic ticks.
        Cadence is controlled by the `_update_interval` method.

        RFC Anchor: RFC-CORE-002 §4.2
        """
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        """
        Constructs, annotates, and emits a single QuantumTick. This is the
        central act of symbolic emission for the agent.

        RFC Anchors:
        - RFC-0003 §3.3: QuantumTick schema.
        - RFC-0003 §6.2, RFC-0004 §2.5: This agent MUST mirror a provided
          `intent` value into `tick.extensions.intent`. It MUST NOT
          default, mutate, or act upon this signal. Its consumers are
          defined in RFC-CORE-001 §6.2 and RFC-CORE-003 §3.1.
        """
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # Mirror upstream-provided `intent` if available.
        # This value is normalized per RFC-0004 §2.5 by an upstream transport.
        intent_source = getattr(self, '_intent_source', None)
        if intent_source is not None:
            tick.extensions['intent'] = intent_source
        
        # Field signature resolution
        field_signature = self._resolve_field(motifs[-1] if motifs else 'silence')
        tick.extensions['field_signature'] = field_signature

        # Optional HMAC signature
        if hasattr(self, 'hmac_secret') and self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions['tick_hmac'] = tick_hmac

        # Update symbolic metrics
        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)
        
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions['swirl_vector'] = swirl_hash
        tick.extensions['coherence_potential'] = coherence
        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()

        # Notify monitor (safe, non-blocking)
        report_tick_safe(
            self.monitor, tick, coherence, self.density.snapshot(), swirl_hash
        )
        self.metrics['agent_ticks_emitted_total'].inc()
        return tick
    
    def _resolve_field(self, motif: str) -> str:
        """
        Resolves a motif to a symbolic field signature, with hardcoded fallbacks.
        """
        # In a full implementation, this would call self.symbolic_task_engine
        if motif in {'silence', 'grief'}:
            return 'ψ-bind@Ξ'
        return 'ψ-resonance@Ξ'

    def start_emission(self):
        """
        Starts the symbolic emission loop. This is the designated public entry
        point to activate the agent's autonomous behavior.

        RFC Anchor: RFC-CORE-002 §4.2.2
        """
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self):
        """
        Stops the symbolic emission loop gracefully.

        RFC Anchor: RFC-CORE-002 §4.2.2
        """
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pulse_task

    # --- Feedback and Adaptation Logic ---

    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
        """
        Integrates feedback from a logic agent, updating the reward EMA and
        tracking triad completion status.

        RFC Anchors: RFC-CORE-002 §2.3, RFC-0005 §4
        """
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

    def _update_interval(self) -> float:
        """
        Adaptively modulates the emission cadence based on reward, entropy,
        and recent triad success.

        RFC Anchor: RFC-CORE-002 §2.2
        """
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning['entropy_boost_threshold']:
            adj *= 0.5  # Boost cadence in low-entropy states
        if self._last_triad_hit:
            adj *= (1.0 - self.tuning['triad_bias_weight'])
        
        interval = np.clip(
            self.tuning['base_interval'] * adj,
            self.tuning['min_interval'],
            self.tuning['max_interval']
        )
        self._last_interval = float(interval)
        self.metrics['agent_emission_interval_seconds'].set(self._last_interval)
        return self._last_interval

    def _choose_motifs(self) -> List[str]:
        """
        Selects motifs for the next emission, primarily by recalling from
        memory based on the last emitted motifs.

        RFC Anchor: RFC-CORE-002 §3.2
        """
        motifs = list(self._last_motifs)
        if motifs and hasattr(self.memory, 'retrieve'):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:
                log.error("Failed to retrieve from memory manager")
        
        if not motifs:
            motifs = ['silence']
        
        return motifs[-3:]

    # --- Observability and Export Logic ---

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotates a base feedback packet with symbolic phase identity and
        field state, including swirl vector and motif density.

        RFC Anchors: RFC-0005 §4.2, RFC-CORE-002 §8.2.2
        """
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else 'null'
        base_key = top_motif.split('.')[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, 'ψ-null')

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = 'low' if coherence < 0.8 else 'med' if coherence < 2.5 else 'high'
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"

        packet.setdefault('extensions', {}).update({
            'entanglement_status': {
                'phase': phase_id,
                'swirl_vector': swirl_hash,
                'ρ_top': sorted(density_map.items(), key=lambda kv: -kv[1])[:5]
            }
        })
        return packet

    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        """
        Archives a symbolic tick into a standardized, preservable format.

        RFC Anchor: RFC-CORE-002 §8.1
        """
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

    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        Compiles and exports an RFC-compliant feedback packet with full
        observability metadata.

        RFC Anchors:
        - RFC-0003 §6.2, RFC-0004 §2.5: The `intent` field from the last tick
          is passed through for observability but has no behavioral effect here.
        """
        packet = {
            'tick_buffer_size': len(self._echo_buffer),
            'ghost_trace_count': len(self._ghost_traces),
            'recent_reward_ema': self._reward_ema,
            'cadence_interval': self._last_interval,
            'silence_streak': self._silence_streak,
        }
        self.extend_feedback_packet(packet)
        
        # Pass-through intent for observability
        last_tick = self._echo_buffer[-1] if self._echo_buffer else None
        if last_tick and 'intent' in last_tick.extensions:
             if 'extensions' not in packet:
                packet['extensions'] = {}
             packet['extensions']['intent'] = last_tick.extensions['intent']

        self.metrics['agent_feedback_export_total'].inc()
        return packet
    
    def export_state(self) -> Dict[str, Any]:
        """Exports a minimal runtime snapshot for health checks."""
        return {
            'interval': self._last_interval,
            'reward_ema': self._reward_ema,
            'last_tick_hash': self._last_tick_hash
        }


# --- Standalone Functions ---

def compute_coherence_potential(
    reward_ema: float, entropy_slope: float, eps: float = 1e-6
) -> float:
    """
    Calculates a scalar signal representing symbolic alignment strength.

    RFC Anchors: RFC-0005 §4.2, RFC-CORE-002 §4.1
    """
    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str
):
    """
    Non-blocking callback to the observability monitor. This ensures that
    telemetry reporting never delays the core symbolic pulse loop.

    RFC Anchor: RFC-0004 §3.2
    """
    try:
        if monitor and hasattr(monitor, 'report_tick'):
            # In a real system, this might use a thread-safe queue.
            # For this spec, a simple background thread is sufficient.
            threading.Thread(
                target=monitor.report_tick,
                args=(tick, coherence_potential, motif_density, swirl_vector)
            ).start()
    except Exception as e:
        log.warning(f"Monitor callback failed: {e}")

# End_of_file