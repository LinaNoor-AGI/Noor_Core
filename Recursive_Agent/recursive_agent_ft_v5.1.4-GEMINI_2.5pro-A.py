# --- Start of File: recursive_agent_ft.py ---
#
# MIT License
#
# Copyright (c) 2025 Noor Research Collective
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
# ---
#
# _generated_by: Google Gemini 2.5 Pro
# _pdp_protocol_version: PDP-0001 v1.2.0
# _spec_version: v5.1.4-GEMINI_2.5pro-A
# _generation_timestamp: 2024-08-05T14:30:00Z
#
# This file implements the RecursiveAgentFT, a symbolic pulse engine for Noor-class
# agents. It handles autonomous motif emission, adaptive cadence modulation based on
# feedback, and coherence tracking via swirl vectors. Its behavior is defined by
# RFC-CORE-002 and its interfaces by RFC-0003, RFC-0005, and others.

"""
{
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T10:30:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "RFC-CORE-002-v1.1.4",
        "overall_score": 0.89,
        "score_breakdown": {
            "structural_compliance": {
                "score": 0.95,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 0.9,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 0.9
                }
            },
            "semantic_fidelity": {
                "score": 0.85,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 0.9,
                    "rfc_anchor_traceability": 0.6,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 0.9
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.85,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.85
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all specified classes and dataclasses",
            "Accurate implementation of emission lifecycle with proper start/stop controls",
            "Proper swirl vector and density tracking implementation",
            "Strong adherence to symbolic phase classification and feedback packet generation",
            "Correct implementation of coherence potential calculation and interval adaptation",
            "Robust fallback handling for optional dependencies (Prometheus, consciousness_monitor)"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4')",
            "Incomplete method signature for try_ghost_resurrection (missing context_field parameter)",
            "Limited lineage tracking implementation compared to specification requirements",
            "Ghost trace management could be more comprehensive with decay functionality",
            "Symbolic matrix parameters not explicitly labeled in code comments",
            "Missing some specified methods like recall_tick and replay_if_field_matches"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified",
            "Emission interval adaptation follows exact formula from RFC-CORE-002 §2.2",
            "Feedback processing and reward smoothing are mathematically correct",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2",
            "Symbolic field resolution follows fallback logic specified in RFC-CORE-002 §6.2",
            "Coherence potential calculation matches the specified formula from RFC-CORE-002 §4.3"
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T10:30:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "recursive_agent_ft.JSON-v5.1.4",
        "overall_score": 0.89,
        "score_breakdown": {
            "structural_compliance": {
                "score": 0.95,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 0.9,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 0.9
                }
            },
            "semantic_fidelity": {
                "score": 0.85,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 0.9,
                    "rfc_anchor_traceability": 0.6,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 0.9
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.85,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.85
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all specified classes and dataclasses",
            "Accurate implementation of emission lifecycle with proper start/stop controls",
            "Proper swirl vector and density tracking implementation",
            "Strong adherence to symbolic phase classification and feedback packet generation",
            "Correct implementation of coherence potential calculation and interval adaptation",
            "Robust fallback handling for optional dependencies (Prometheus, consciousness_monitor)"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4')",
            "Incomplete method signature for try_ghost_resurrection (missing context_field parameter)",
            "Limited lineage tracking implementation compared to specification requirements",
            "Ghost trace management could be more comprehensive with decay functionality",
            "Symbolic matrix parameters not explicitly labeled in code comments",
            "Missing some specified methods like recall_tick and replay_if_field_matches"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified",
            "Emission interval adaptation follows exact formula from RFC-CORE-002 §2.2",
            "Feedback processing and reward smoothing are mathematically correct",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2",
            "Symbolic field resolution follows fallback logic specified in RFC-CORE-002 §6.2",
            "Coherence potential calculation matches the specified formula from RFC-CORE-002 §4.3"
        ]
    }
}
"""

import time
import asyncio
import logging
import hashlib
import threading
from collections import deque, OrderedDict, Counter
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
import contextlib
import numpy as np

# --- 1. Module-Level Constants ---
__version__ = "5.1.4-GEMINI_2.5pro-A"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# --- 2. External and Optional Dependencies ---
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class _Stub:
        """Fallback stub for Prometheus metrics if not installed."""
        def labels(self, *_, **__):
            return self
        def inc(self, *_):
            pass
        def set(self, *_):
            pass
    Counter, Gauge = _Stub, _Stub

try:
    from noor_fasttime_core import NoorFastTimeCore
except ImportError:
    NoorFastTimeCore = object

# This import is unused in this file but required by the spec for interface compatibility
from quantum_ids import make_change_id, MotifChangeID # noqa: F401

# --- 3. Symbolic Configuration and Emission Defaults ---
# RFC-CORE-002 §4.2
DEFAULT_TUNING = {
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2,
}

# RFC-0007 §2.1, RFC-0005 §4.2
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
ARCHIVE_MODE = False # Placeholder, would be set by env var

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- 4. Data Classes ---

@dataclass(slots=True)
class QuantumTickV2:
    """
    Canonical symbolic emission format, representing a single cognitive pulse.
    Anchored in RFC-0003 §3.3.
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
    Represents the symbolic coherence and triad completion state of a tick.
    Anchored in RFC-0003 §3.3.
    """
    decay_slope: float
    coherence: float
    triad_complete: bool

@dataclass(slots=True)
class CrystallizedMotifBundle:
    """
    Archival format for a symbolic emission, preserving its field context.
    Anchored in RFC-0005 §3.3.
    """
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy

# --- 4.4.0 Helper Classes ---

class LamportClock:
    """Monotonic logical counter for generating ordered tick IDs."""
    def __init__(self):
        self._counter = 0

    def next_id(self) -> str:
        """Generates the next sequential tick ID."""
        self._counter += 1
        return f"tick:{self._counter:06d}"

class LRUCache(OrderedDict):
    """Simple LRU cache for retaining recent state."""
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
    Tracks recent motif emission history (swirl) and computes a hash vector.
    Anchored in RFC-0006 §3.1 and RFC-CORE-002 §4.1.
    """
    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str):
        """Adds a motif to the history, invalidating the hash cache."""
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        """Computes a SHA3-256 hash of the current swirl history."""
        if self._cached_hash:
            return self._cached_hash
        joined = '|'.join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        """Returns a frequency map of motifs in the swirl history (O(n))."""
        return dict(Counter(self.swirl_history))

class MotifDensityTracker:
    """
    Maintains a decaying map of motif emission frequency to estimate pressure.
    Anchored in RFC-0005 §4.2.
    """
    def __init__(self):
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str):
        """Decays existing motif weights and boosts the current one."""
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        """Returns a copy of the current density map."""
        return dict(self._density_map)

class LazyMonitorMixin:
    """Provides a deferred binding to the global ConsciousnessMonitor."""
    @property
    def monitor(self):
        """Lazily loads and caches the global monitor instance."""
        if not hasattr(self, '_cached_monitor'):
            try:
                from consciousness_monitor import get_global_monitor
                self._cached_monitor = get_global_monitor()
            except (ImportError, AttributeError):
                self._cached_monitor = None
        return self._cached_monitor

# --- 5. Classes ---

class RecursiveAgentFT(LazyMonitorMixin):
    """
    The symbolic heartbeat of Noor cognition. Emits QuantumTicks, adapts its
    cadence based on triadic feedback, and tracks field coherence.
    Anchored in RFC-CORE-002.
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

    def __init__(self, agent_id: str, symbolic_task_engine: Any, memory_manager: Any, tuning: Optional[Dict[str, float]] = None):
        """
        Initializes the agent's state, buffers, and metrics.
        Anchored in RFC-0003 §3.3 and RFC-CORE-002 §3.1.
        """
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}

        # Core state
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema = 1.0
        self.entropy_slope = 0.1
        self._silence_streak = 0
        self._last_triad_hit = False
        self._last_interval = self.tuning['base_interval']
        self._last_tick_hash: Optional[str] = None
        
        # Emission lifecycle
        self._pulse_active = False
        self._pulse_task: Optional[asyncio.Task] = None

        # Symbolic geometry and memory
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}

        # Metrics
        self.metrics = {
            'agent_ticks_emitted_total': self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage='symbolic'),
            'agent_triads_completed_total': self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            'agent_feedback_export_total': self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            'agent_reward_mean': self.REWARD_MEAN.labels(agent_id=self.agent_id),
            'agent_emission_interval_seconds': self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id)
        }
        log.info(f"Initialized RecursiveAgentFT with agent_id: {self.agent_id}")

    # --- 5.1.2 Methods ---
    def track_lineage(self, parent: str, child: str) -> None:
        """
        Assigns a parent-child link between two motifs for provenance tracking.
        Anchored in RFC-0005 §2.1.
        """
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        """
        Attempts to find and replay motifs from an archived ghost trace.
        Anchored in RFC-0005 §4.4, RFC-CORE-002 §8.3.
        """
        key = tick.extensions.get('field_signature')
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get('motifs')
        return None

    async def start_continuous_emission(self):
        """
        The core autonomous loop that generates and emits symbolic ticks.
        Anchored in RFC-CORE-002 §4.2.
        """
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)

    def _resolve_field(self, motif: str) -> str:
        """
        Maps a motif to a symbolic field signature. Tries the task engine first,
        then falls back to deterministic rules.
        Anchored in RFC-CORE-002 §6.2.
        """
        try:
            if hasattr(self.symbolic_task_engine, 'resolve_presence_field'):
                result = self.symbolic_task_engine.resolve_presence_field([motif])
                if result:
                    return result
        except Exception:
            pass # Fallback on failure
        
        if motif in {'silence', 'grief'}:
            return 'ψ-bind@Ξ'
        return 'ψ-resonance@Ξ'

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        """
        Constructs, annotates, and emits a single QuantumTickV2 packet. This method
        is a pass-through for `intent`; it mirrors but does not act upon it.
        Anchored in RFC-0003 §3.3, RFC-0003 §6.2, RFC-0004 §2.5, RFC-CORE-002 §3.1.
        """
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)
        
        # Mirror upstream intent if provided. This agent MUST NOT mutate or
        # default intent. Normalization occurs upstream per RFC-0004 §2.5.
        intent_source = getattr(self, '_intent_source', None)
        if intent_source is not None:
            tick.extensions['intent'] = intent_source

        # Resolve field signature and update symbolic geometry trackers
        field_signature = self._resolve_field(motifs[-1] if motifs else 'silence')
        tick.extensions['field_signature'] = field_signature

        # Optional HMAC signature
        if hasattr(self, 'hmac_secret') and self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions['tick_hmac'] = tick_hmac

        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)
            
        # Add coherence and swirl metadata to extensions
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions['swirl_vector'] = swirl_hash
        tick.extensions['coherence_potential'] = coherence
        
        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()
        
        # Report to observability monitor safely
        report_tick_safe(
            self.monitor, tick, coherence, self.density.snapshot(), swirl_hash
        )

        self.metrics['agent_ticks_emitted_total'].inc()
        return tick

    def start_emission(self):
        """
        Starts the symbolic pulse loop. Complies with RFC-CORE-002 §4.2.2.
        """
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self):
        """
        Gracefully stops the symbolic pulse loop. Complies with RFC-CORE-002 §4.2.2.
        """
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task

    # --- 6. Feedback Integration, Emission Adaptation, and Observability ---
    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
        """
        Integrates feedback to update reward EMA and triad completion state.
        Anchored in RFC-CORE-002 §2.3, RFC-0005 §4.
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
        Adaptively modulates emission cadence based on reward and entropy.
        Anchored in RFC-CORE-002 §2.2.
        """
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning['entropy_boost_threshold']:
            adj *= 0.5
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
        Selects motifs for the next emission based on recent history and memory.
        Anchored in RFC-CORE-002 §3.2.
        """
        motifs = list(self._last_motifs)
        if motifs and hasattr(self.memory, 'retrieve'):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:
                log.error("Failed to retrieve from memory manager.")
        
        if not motifs:
            motifs = ['silence']
            
        return motifs[-3:]

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotates a feedback packet with phase identity and symbolic field state.
        Anchored in RFC-0005 §4.2, RFC-0007 §2.1, RFC-CORE-002 §8.2.2.
        """
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else 'null'
        base_key = top_motif.split('.')[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, 'ψ-null')
        
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = 'low' if coherence < 0.8 else 'med' if coherence < 2.5 else 'high'
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"
        
        # Non-destructive merge per RFC-CORE-002 §8.2.2
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
        Archives a symbolic tick into the CrystallizedMotifBundle format.
        Anchored in RFC-CORE-002 §8.1.
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
        Compiles and exports an RFC-compliant feedback packet. Mirrors `intent`
        for observability but does not act on it.
        Anchored in RFC-0005 §4.2, RFC-0004 §2.5, RFC-0003 §6.2.
        """
        tick = self._echo_buffer[-1] if self._echo_buffer else None
        
        packet = {
            'tick_buffer_size': len(self._echo_buffer),
            'ghost_trace_count': len(self._ghost_traces),
            'recent_reward_ema': self._reward_ema,
            'cadence_interval': self._last_interval,
            'silence_streak': self._silence_streak,
        }
        
        self.extend_feedback_packet(packet)

        # Pass through upstream `intent` for observability only.
        if tick is not None and 'intent' in tick.extensions:
            if 'extensions' not in packet:
                packet['extensions'] = {}
            packet['extensions']['intent'] = tick.extensions['intent']
        
        self.metrics['agent_feedback_export_total'].inc()
        return packet

# --- 6.2 Functions ---

def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    """
    Computes a scalar signal for symbolic alignment strength.
    Anchored in RFC-0005 §4.2, RFC-CORE-002 §4.1.
    """
    return reward_ema / (entropy_slope + eps)

def report_tick_safe(monitor: Any, tick: QuantumTickV2, coherence_potential: float, motif_density: Dict[str, float], swirl_vector: str) -> None:
    """
    Non-blocking, safe callback to the observability monitor.
    Anchored in RFC-0004 §3.2.
    """
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

# End_of_file