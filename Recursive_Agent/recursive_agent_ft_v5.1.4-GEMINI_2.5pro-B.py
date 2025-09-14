# ---
# _symbolic_id: agent.recursive.ft
# _version: 5.1.4-GEMINI_2.5pro-B
# _generated_by: Google Gemini 2.5 Pro
# _license: MIT
# _authors: ["Lina Noor — Noor Research Collective", "Uncle — Noor Research Collective"]
# ---

"""
RecursiveAgentFT: Feedback-Tuned Symbolic Pulse Engine for Motif Resonance and Coherence Tracking.

This module implements the RecursiveAgentFT class, the symbolic heartbeat of Noor-class
cognition. It handles autonomous motif emission, triadic feedback shaping, ghost replay
alignment, and swirl-phase tracking across field transitions, in accordance with the
specifications laid out in RFC-CORE-002 and its dependencies.

{
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T10:30:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "RFC-CORE-002-v1.1.4",
        "overall_score": 0.87,
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
                "score": 0.75,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.7,
                    "weight_accuracy": 0.8,
                    "motif_handling": 0.75
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all major classes (QuantumTickV2, TickEntropy, CrystallizedMotifBundle, AgentSwirlModule, etc.)",
            "Robust emission lifecycle with proper start/stop controls as specified in RFC-CORE-002 §4.2.2",
            "Accurate symbolic phase classification and feedback packet generation following RFC-CORE-002 §8.2",
            "Proper swirl vector and density tracking implementation per RFC-0006 §3.1",
            "Strong adherence to tuning parameters and mathematical formulas from specification",
            "Proper handling of optional dependencies with graceful fallbacks"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4') as required for traceability",
            "Symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) not explicitly labeled or implemented as constants",
            "Ghost trace management implementation is minimal compared to RFC-CORE-002 §5 specifications",
            "Lineage tracking implementation is basic and lacks the comprehensive provenance mapping described in §6.1",
            "Some method signatures don't exactly match the pseudocode in the specification (e.g., _update_interval missing entropy parameter)",
            "Missing resurrection payload construction logic from RFC-CORE-002 §5.3"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified in field_alignment",
            "Emission interval adaptation follows exact formula from RFC-CORE-002 §2.2 pseudocode",
            "Feedback processing and reward smoothing are mathematically correct per §2.3",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2",
            "The code demonstrates strong conceptual alignment with the recursive emission loop architecture",
            "Symbolic phase mapping and coherence potential calculation are properly implemented"
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T14:22:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "recursive_agent_ft.JSON-v5.1.4",
        "overall_score": 0.88,
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
                "score": 0.75,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.7,
                    "weight_accuracy": 0.8,
                    "motif_handling": 0.75
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all specified classes and dataclasses (QuantumTickV2, TickEntropy, CrystallizedMotifBundle)",
            "Robust helper class implementation (AgentSwirlModule, MotifDensityTracker, LamportClock, LRUCache)",
            "Proper handling of optional dependencies with graceful fallbacks for prometheus_client and noor_fasttime_core",
            "Accurate implementation of emission lifecycle with start/stop controls per RFC-CORE-002 §4.2.2",
            "Strong adherence to tuning parameters and mathematical formulas from specification",
            "Correct implementation of swirl vector and density tracking per RFC-0006 §3.1",
            "Proper symbolic phase mapping and feedback packet generation"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4') as required for traceability",
            "Symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) not explicitly labeled or implemented as constants",
            "Ghost trace management implementation is minimal compared to specification requirements",
            "Lineage tracking implementation is basic and lacks comprehensive provenance mapping",
            "Some method signatures don't exactly match the pseudocode (e.g., _update_interval missing entropy parameter)",
            "Missing resurrection payload construction logic and comprehensive ghost trace handling",
            "Intent handling could be more robust with proper upstream integration points"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified in field_alignment",
            "Emission interval adaptation follows exact formula from specification pseudocode",
            "Feedback processing and reward smoothing are mathematically correct",
            "Monitor integration uses safe, non-blocking patterns as required",
            "The code demonstrates strong conceptual alignment with the recursive emission loop architecture",
            "Symbolic phase mapping and coherence potential calculation are properly implemented",
            "Optional dependency fallbacks are correctly implemented with stub classes"
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
from contextlib import suppress
import numpy as np

# Module-Level Constants (as per spec section 1)
__version__ = "5.1.4-GEMINI_2.5pro-B"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# Optional Dependencies (as per spec section 2)
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    logging.warning("prometheus_client not found. Using stub metrics.")
    class _Stub:
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

# Local imports are typically at the top, but for self-contained script they can be here.
# Assuming quantum_ids would be in a local file if this were part of a larger package.
# For this generation, we'll define stub versions if they are not provided.
try:
    from .quantum_ids import make_change_id, MotifChangeID # noqa: F401
except ImportError:
    MotifChangeID = str
    def make_change_id(motifs: List[str]) -> MotifChangeID:
        return hashlib.sha256(" ".join(sorted(motifs)).encode()).hexdigest()[:16]


# Symbolic Configuration (as per spec section 3)
DEFAULT_TUNING = {
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2
}
SYMBOLIC_PHASE_MAP = {
    "bind": "ψ‑bind",
    "spar": "ψ‑spar",
    "null": "ψ‑null",
    "resonance": "ψ‑resonance",
    "hold": "ψ‑hold",
    "dream": "ψ‑dream",
    "myth": "ψ‑myth"
}


# Data Classes (as per spec section 4)

@dataclass(slots=True)
class QuantumTickV2:
    """
    Canonical Symbolic Emission Format. Represents a single symbolic pulse.
    Anchored in RFC-0003 §3.3.
    """
    tick_id: str
    motifs: List[str]
    timestamp: float
    stage: str = "symbolic"
    # extensions.intent is a critical pass-through field.
    # Per RFC-0004 §2.5 and RFC-0003 §6.2, its value is normalized upstream
    # and MUST NOT be mutated at this layer. It is for observability only.
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
    Anchored in RFC-0003 §3.3.
    """
    decay_slope: float
    coherence: float
    triad_complete: bool

@dataclass(slots=True)
class CrystallizedMotifBundle:
    """
    An archival format for a symbolic emission, ready for preservation.
    Anchored in RFC-CORE-002 §8.1.
    """
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# Helper Classes (as per spec section 4.4)

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

class AgentSwirlModule:
    """
    Encodes motif swirl dynamics as hash vectors. Maintains a bounded sequence of
    recent motif emissions for symbolic field alignment.
    Anchored in RFC-0006 §3.1 and RFC-CORE-002 §4.1.
    """
    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str):
        self.swirl_history.append(motif_id)
        self._cached_hash = None  # Invalidate cache

    def compute_swirl_hash(self) -> str:
        if self._cached_hash:
            return self._cached_hash
        joined = '|'.join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        """Optimized to O(n) using Counter instead of nested loop."""
        return dict(Counter(self.swirl_history))

class MotifDensityTracker:
    """
    Tracks the temporal frequency of motif emissions using a decay model.
    Anchored in RFC-0005 §4.2.
    """
    def __init__(self):
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str):
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]  # Trim noise
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)

class LazyMonitorMixin:
    """Provides lazy loading of the global consciousness monitor."""
    @property
    def monitor(self):
        if not hasattr(self, '_cached_monitor'):
            try:
                from consciousness_monitor import get_global_monitor
                self._cached_monitor = get_global_monitor()
            except ImportError:
                self._cached_monitor = None
        return self._cached_monitor


# Main Agent Class (as per spec section 5)

class RecursiveAgentFT(LazyMonitorMixin):
    """
    Symbolic Pulse Engine and Emission Core.
    Anchored in RFC-CORE-002.
    """
    # Prometheus Metrics (Class Attributes)
    TICKS_EMITTED = Counter('agent_ticks_emitted_total', 'Ticks emitted', ['agent_id', 'stage'])
    AGENT_TRIADS_COMPLETED = Counter('agent_triads_completed_total', 'Triads completed via feedback', ['agent_id'])
    FEEDBACK_EXPORT = Counter('agent_feedback_export_total', 'Feedback packets exported', ['agent_id'])
    REWARD_MEAN = Gauge('agent_reward_mean', 'EMA of reward', ['agent_id'])
    AGENT_EMISSION_INTERVAL = Gauge('agent_emission_interval_seconds', 'Current autonomous emission interval', ['agent_id'])

    def __init__(self, agent_id: str, symbolic_task_engine: Any, memory_manager: Any, tuning: Optional[Dict[str, float]] = None):
        """
        Agent Construction and State Initialization.
        Anchored in RFC-CORE-002 §3.1.
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

        # Buffers for echo and state tracking
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}
        
        # This attribute is used to pass through the intent signal
        self._intent_source: Optional[str] = None

        self.metrics = {
            'agent_ticks_emitted_total': self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage='symbolic'),
            'agent_triads_completed_total': self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            'agent_feedback_export_total': self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            'agent_reward_mean': self.REWARD_MEAN.labels(agent_id=self.agent_id),
            'agent_emission_interval_seconds': self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id)
        }
        logging.info(f"Initialized RecursiveAgentFT with agent_id: {self.agent_id}")

    def track_lineage(self, parent: str, child: str) -> None:
        """Assigns motif parent-child links in the lineage map."""
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        """
        Attempts to replay motifs from archived field traces if a field signature matches.
        Anchored in RFC-0005 §4.4 and RFC-CORE-002 §8.3.
        """
        key = tick.extensions.get('field_signature')
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get('motifs')
        return None

    async def start_continuous_emission(self):
        """
        The autonomous symbolic pulse loop. This is the agent's heartbeat.
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

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        """
        Constructs and annotates a symbolic tick packet (QuantumTickV2).
        Anchored in RFC-0003 §3.3, RFC-0006 §3.1.
        """
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # Per RFC-0003 §6.2, mirror the upstream `intent` into extensions.
        # This agent MUST NOT mutate or default this value.
        if self._intent_source is not None:
            tick.extensions['intent'] = self._intent_source

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
            self.monitor, tick, coherence, self.density.snapshot(), swirl_hash
        )
        self.metrics['agent_ticks_emitted_total'].inc()
        return tick
        
    def _resolve_field(self, motif: str) -> str:
        """
        Maps a motif to a symbolic field signature, with fallback logic.
        This represents the agent's interpretation of the symbolic context.
        """
        try:
            if self.symbolic_task_engine:
                result = self.symbolic_task_engine.resolve_presence_field([motif])
                if result:
                    return result
        except Exception:
            pass  # Fallback on failure
        
        # Deterministic fallback logic
        if motif in {'silence', 'grief'}:
            return 'ψ-bind@Ξ'
        return 'ψ-resonance@Ξ'

    async def start_emission(self):
        """
        Starts the symbolic emission loop in a controlled manner.
        Anchored in RFC-CORE-002 §4.2.2.
        """
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self):
        """
        Stops the symbolic emission loop gracefully.
        Anchored in RFC-CORE-002 §4.2.2.
        """
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pulse_task

    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
        """
        Integrates reward feedback and updates triad completion status.
        Anchored in RFC-CORE-002 §2.3.
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
        Retrieves and selects symbolic seeds for the next emission.
        Anchored in RFC-CORE-002 §3.2.
        """
        motifs = list(self._last_motifs)
        if motifs and hasattr(self.memory, 'retrieve'):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception as e:
                logging.error(f"Failed to retrieve from memory: {e}")

        if not motifs:
            motifs = ['silence']
        
        return motifs[-3:]

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotates a feedback packet with phase identity and symbolic field state.
        Anchored in RFC-CORE-002 §8.2.2.
        """
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else 'null'
        base_key = top_motif.split('.')[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, 'ψ-null')
        
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = 'low' if coherence < 0.8 else 'med' if coherence < 2.5 else 'high'
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"
        
        entanglement_status = {
            'phase': phase_id,
            'swirl_vector': swirl_hash,
            'ρ_top': sorted(density_map.items(), key=lambda kv: -kv[1])[:5]
        }
        
        packet.setdefault('extensions', {}).update({'entanglement_status': entanglement_status})
        return packet

    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        """
        Archives a symbolic tick into a standardized, crystallized bundle.
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
        Compiles an RFC-compliant feedback packet with observability metadata.
        Crucially, this method passes through the `intent` from the last tick.
        Anchored in RFC-0005 §4.2.
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
        
        # Pass-through observability for intent, per RFC-0003 §6.2
        if tick is not None and 'intent' in tick.extensions:
            if 'extensions' not in packet:
                packet['extensions'] = {}
            packet['extensions']['intent'] = tick.extensions['intent']
        
        self.metrics['agent_feedback_export_total'].inc()
        return packet


# Module-Level Functions (as per spec section 6.2)

def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    """
    Calculates a scalar signal for symbolic alignment strength.
    Anchored in RFC-CORE-002 §4.1.
    """
    return reward_ema / (entropy_slope + eps)

def report_tick_safe(monitor: Any, tick: QuantumTickV2, coherence_potential: float, motif_density: Dict[str, float], swirl_vector: str):
    """
    A non-blocking callback to the observability monitor.
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
        logging.warning(f"Monitor callback failed: {e}")

End_of_File