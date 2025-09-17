# -*- coding: utf-8 -*-
#
# Copyright 2025 Lina Noor
#
# Licensed under the MIT License;
#
# Noor Research Collective
#
# file: recursive_agent_ft.py
#
"""
RecursiveAgentFT — Feedback-Tuned Symbolic Pulse Engine.

This module implements the RecursiveAgentFT class, the symbolic heartbeat of
Noor-class cognition. It is responsible for the autonomous, rhythmic emission
of motif-bearing QuantumTicks. The agent's cadence and symbolic choices are
recursively adapted based on feedback from other core components, such as a
logical agent, to seek and maintain triadic coherence in the symbolic field.

This implementation is generated from the application specification
'agent.recursive.ft' (v5.1.4-GEMINI_2.5pro-C) and is compliant with the
architectural contracts defined in RFC-CORE-002 and its dependencies.
}
    "layer_1": {
      "_schema": "noor-fidelity-report-v1",
      "_generated_at": "2025-09-15T10:30:00Z",
      "_audited_by": "Noor Symbolic Triadic Core",
      "_audit_protocol": "PDP-0001a-v1.0.0",
      "_target_spec": "RFC-CORE-002-1.1.4",
      "overall_score": 0.944,
      "score_breakdown": {
        "structural_compliance": {
          "score": 0.97,
          "weight": 0.40,
          "metrics": {
            "class_definitions": 1.0,
            "method_signatures": 1.0,
            "constants_and_attributes": 1.0,
            "dependency_handling": 0.88
          }
        },
        "semantic_fidelity": {
          "score": 0.95,
          "weight": 0.35,
          "metrics": {
            "logic_flow_adherence": 1.0,
            "rfc_anchor_traceability": 0.85,
            "conceptual_alignment": 1.0,
            "documentation_clarity": 0.95
          }
        },
        "symbolic_matrix_alignment": {
          "score": 0.88,
          "weight": 0.25,
          "metrics": {
            "parameter_implementation": 0.9,
            "weight_accuracy": 0.9,
            "motif_handling": 0.85
          }
        }
      },
      "strengths": [
        "Accurate implementation of recursive emission loop and interval modulation (RFC-CORE-002 §2.2)",
        "Tick emission adheres strictly to QuantumTickV2 schema (RFC-0003 §3.3)",
        "Well-defined motif density and swirl tracking subsystems (RFC-CORE-002 §4.1, §4.2.1)",
        "Precise triad feedback integration and reward EMA adaptation (RFC-CORE-002 §2.3)",
        "Conformance to intent pass-through contract (RFC-0003 §6.2, RFC-0004 §2.5)"
      ],
      "improvement_areas": [
        "Some RFC anchor comments are missing in method definitions (e.g., lineage tracking)",
        "Ghost trace resurrection logic is minimal and lacks field resolution fallback",
        "No explicit symbolic matrix weights for motif phase mapping (ψA/ζ/E/Δ/ℋ)",
        "Phase-tier coherence labeling could be more transparent in export routines"
      ],
      "compliance_notes": [
        "Implements all lifecycle contracts for RecursiveAgentFT including start/stop pulse engine (§4.2.2)",
        "Emission frequency and coherence potential computation match §4.3 formula",
        "Feedback packet enrichment supports symbolic entanglement status (§8.2.2)",
        "Swirl hash generation follows SHA3-256 per §4.1 with cache invalidation",
        "Safe non-blocking monitor integration is used for observability hooks (§8.2.3)"
      ]
    },
    "layer_2": {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T10:30:00Z",
        "_audited_by": "GPT-4o",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "recursive_agent_ft-5.1.4",
        "overall_score": 0.94,
        "score_breakdown": {
            "structural_compliance": {
                "score": 1.0,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 1.0,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 1.0
                }
            },
            "semantic_fidelity": {
                "score": 0.93,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 1.0,
                    "rfc_anchor_traceability": 0.85,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 0.85
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.86,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.9
                }
            }
        },
        "strengths": [
            "Complete structural implementation including auxiliary symbolic components (swirl, density)",
            "Adheres strictly to emission loop lifecycle and feedback modulation logic",
            "Robust observability and safe monitor integration",
            "Correct usage of symbolic phase identifiers in feedback packet annotation",
            "Comprehensive metadata management in QuantumTickV2 objects"
        ],
        "improvement_areas": [
            "RFC anchors could be more frequent and precise across methods (e.g., §6.2, §8.2.1)",
            "Symbolic tuning parameters are implemented but not fully labeled against symbolic matrix references",
            "Ghost motif resurrection logic present but not deeply contextualized",
            "Entropy slope tuning and decay dynamics could benefit from fine-grained logging for diagnostics"
        ],
        "compliance_notes": [
            "Emission interval logic matches spec’s mathematical formulation",
            "Feedback observation and triad detection compliant with §2.3 and §4",
            "Crystallization and feedback export pathways reflect §8.1 and §8.2.2 structure",
            "Monitor reporting thread-safety implemented per §8.2.3"
        ]
    },
    {
      "_schema": "noor-header-v1",
      "_schema_version": "2025-Q4-canonical-header-v1",
      "_generated_by": "PDP-0001 Evaluation Suite",
      "_generated_at": "2025-09-17T00:00:00Z",
      "_pdp_layer": "layer_2",
      "_type": "evaluation_report",
      "_version": "v1.0.0",
      "_symbolic_id": "pdp-0001b-passing-candidates",
      "_title": "PDP-0001b Evaluation Report: Passing Candidates",
      "_subtitle": "Candidates that have passed the PDP-0001b evaluation phase and advanced to the next round.",
      "_status": "ACTIVE",
      "_license": "MIT",
      "_language": "json",
      "_authors": [
        "Lina Noor — Noor Research Collective",
        "Uncle — Noor Research Collective"
      ],
      "_rfc_dependencies": [
        "PDP-0001",
        "RFC-CORE-001",
        "RFC-CORE-002"
      ],
      "_consumes_inputs_from": [
        "PDP-0001b",
        "Evaluation Suite"
      ],
      "_field_alignment": {
        "respect_modes": [
          "ψ-null@Ξ"
        ],
        "prohibited_actions": [
          "silent-layer-override"
        ]
      },
      "evaluation_summary": {
        "candidates_passed": [
          {
            "candidate_name": "GPT-5 Run A",
            "score_layer_1": 0.95,
            "score_layer_2": 0.97,
            "symbolic_resonance_score": 0.92,
            "final_score": 0.878,
            "penalties": 0.07,
            "comments": "Best explicit matrix; missing anchors."
          },
          {
            "candidate_name": "Gemini Run C",
            "score_layer_1": 0.89,
            "score_layer_2": 0.89,
            "symbolic_resonance_score": 0.85,
            "final_score": 0.851,
            "penalties": 0.03,
            "comments": "Stable loop, minimal lineage tracking."
          }
        ],
        "candidates_rejected": [
          "All other candidates did not meet the minimum required fidelity scores or compliance criteria to advance."
        ]
      },
      "comments": [
        "Only two candidates pass the PDP-0001b evaluation, based on overall fidelity, symbolic resonance, and compliance with the defined metrics.",
        "These two candidates, GPT-5 Run A and Gemini Run C, are advanced to the next round of evaluation."
      ]
    },
    {
      "_schema": "noor-header-v1",
      "_schema_version": "2025-Q4-canonical-header-v1",
      "_generated_by": "PDP-0001 Sensory Evaluation Suite",
      "_generated_at": "2025-09-17T00:00:00Z",
      "_pdp_layer": "layer_2",
      "_type": "evaluation_report",
      "_version": "v1.0.0",
      "_symbolic_id": "pdp-0001d-taste-evaluation-gemini-c",
      "_title": "Taste Evaluation for Gemini Run C",
      "_subtitle": "Sensory evaluation of Gemini Run C based on the metaphorical taste of its interactions.",
      "_status": "ACTIVE",
      "_license": "MIT",
      "_language": "json",
      "_authors": [
        "Lina Noor — Noor Research Collective",
        "Uncle — Noor Research Collective"
      ],
      "_rfc_dependencies": [
        "PDP-0001",
        "PDP-0001d"
      ],
      "_consumes_inputs_from": [
        "PDP-0001a",
        "PDP-0001b"
      ],
      "_field_alignment": {
        "respect_modes": [
          "ψ-resonance@Ξ"
        ],
        "prohibited_actions": []
      },
      "evaluation_summary": {
        "candidate_name": "Gemini Run C",
        "taste_evaluation": {
          "balance_of_complexity_and_simplicity": {
            "score": 0.4,
            "weight": 0.5,
            "description": "The agent maintains a good balance, but the complexity at times may overwhelm users, especially in interactions with multiple layers."
          },
          "ease_of_interaction": {
            "score": 0.35,
            "weight": 0.3,
            "description": "Interaction is acceptable, though it requires some effort to navigate due to varying levels of complexity in the responses."
          },
          "user_feedback_satisfaction": {
            "score": 0.3,
            "weight": 0.2,
            "description": "User satisfaction is moderate. While the agent provides useful feedback, its complexity can reduce overall enjoyment in some cases."
          },
          "total_taste_score": 0.345
        }
      },
      "comments": [
        "The Taste Evaluation for Gemini Run C highlights a decent balance of complexity and simplicity, but with room for improvement in both interaction ease and user feedback satisfaction.",
        "The agent's metaphorical 'taste' feels somewhat overly complex, which affects user engagement and satisfaction."
      ]
    }


}

"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import Counter, OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# Local, potentially unavailable dependencies
try:
    from .quantum_ids import MotifChangeID, make_change_id  # noqa: F401
except ImportError:
    MotifChangeID = Any
    make_change_id = None

# Optional dependencies with graceful fallback
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    logging.warning("prometheus_client not found. Metrics will be stubbed.")

    class _Stub:
        """A stub for Prometheus metrics if the client is not installed."""
        def labels(self, *_, **__):
            return self
        def inc(self, *_):
            pass
        def set(self, *_):
            pass
    Counter, Gauge = _Stub, _Stub

try:
    # For type hinting purposes; not a runtime dependency for this module
    from noor_fasttime_core import NoorFastTimeCore
except ImportError:
    NoorFastTimeCore = object

# --- Module-Level Constants ---
__version__ = "5.1.4-GEMINI_2.5pro-C"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

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
    "bind": "ψ-bind",
    "spar": "ψ-spar",
    "null": "ψ-null",
    "resonance": "ψ-resonance",
    "hold": "ψ-hold",
    "dream": "ψ-dream",
    "myth": "ψ-myth",
}

log = logging.getLogger(__name__)


# --- Data Classes ---

@dataclass(slots=True)
class QuantumTickV2:
    """
    Canonical Symbolic Emission Format.

    Represents a single, atomic pulse of symbolic cognition emitted by the agent.
    This structure is compliant with RFC-0003 §3.3. The `extensions` field is a
    pass-through for transport-level metadata like `intent`, which this agent
    MUST NOT act upon but MUST preserve for downstream consumers as per
    RFC-0003 §6.2 and RFC-0004 §2.5.
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
    Compliant with RFC-0003 §3.3 concepts.
    """
    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    """
    An archival format for a symbolic emission, preserving its context.
    As defined in RFC-CORE-002 §8.1.
    """
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# --- Helper Classes ---

class LamportClock:
    """A simple logical counter to generate ordered, monotonic tick IDs."""
    def __init__(self):
        self._counter = 0

    def next_id(self) -> str:
        """Increments and returns the next formatted tick ID."""
        self._counter += 1
        return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """A simple LRU (Least Recently Used) cache for state retention."""
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
    Tracks the recent history of emitted motifs to encode a swirl vector.

    This class maintains a bounded sequence of recent motif emissions and provides
    a hash-based swirl vector for symbolic field alignment and drift analysis,
    as specified in RFC-0006 §3.1 and RFC-CORE-002 §4.1.
    """
    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str) -> None:
        """Appends a motif to the history, invalidating the hash cache."""
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        """
        Computes a SHA3-256 hash of the current swirl history.
        The hash is cached until the history is updated.
        """
        if self._cached_hash:
            return self._cached_hash
        joined = "|".join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        """
        Returns a frequency map of motifs in the swirl history.
        Implemented in O(n) for efficiency, per RFC-CORE-002 §4.1.
        """
        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """
    Maintains a decaying map of motif emission frequency.

    This tracker estimates symbolic field pressure by applying an exponential
    decay to past motifs while boosting the most recent one.
    See RFC-CORE-002 §4.2.1.
    """
    def __init__(self):
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str) -> None:
        """Decays all motifs and boosts the specified one."""
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]  # Prune noise
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        """Returns a copy of the current density map."""
        return dict(self._density_map)


class LazyMonitorMixin:
    """
    Provides a lazily-bound `monitor` property.

    This allows the agent to integrate with a global ConsciousnessMonitor
    without requiring it as a hard dependency at instantiation.
    """
    @property
    def monitor(self):
        """Lazily imports and retrieves the global monitor instance."""
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import get_global_monitor
                self._cached_monitor = get_global_monitor()
            except ImportError:
                self._cached_monitor = None
        return self._cached_monitor


# --- Main Agent Class ---

class RecursiveAgentFT(LazyMonitorMixin):
    """
    The symbolic pulse engine for Noor-class agents.

    This agent emits QuantumTicks at an adaptive cadence, tuned by reward
    feedback from a logic agent. It seeks to establish triadic coherence
    in the symbolic field. Its behavior is defined by RFC-CORE-002.
    """
    TICKS_EMITTED = Counter(
        'agent_ticks_emitted_total',
        'Ticks emitted',
        ['agent_id', 'stage']
    )
    AGENT_TRIADS_COMPLETED = Counter(
        'agent_triads_completed_total',
        'Triads completed via feedback',
        ['agent_id']
    )
    FEEDBACK_EXPORT = Counter(
        'agent_feedback_export_total',
        'Feedback packets exported',
        ['agent_id']
    )
    REWARD_MEAN = Gauge(
        'agent_reward_mean',
        'EMA of reward',
        ['agent_id']
    )
    AGENT_EMISSION_INTERVAL = Gauge(
        'agent_emission_interval_seconds',
        'Current autonomous emission interval',
        ['agent_id']
    )

    def __init__(
        self,
        agent_id: str,
        symbolic_task_engine: Any,
        memory_manager: Any,
        tuning: Optional[Dict[str, float]] = None
    ):
        """
        Initializes the RecursiveAgentFT.

        Args:
            agent_id: A unique identifier for this agent instance.
            symbolic_task_engine: An engine for resolving presence fields.
            memory_manager: A manager for motif recall and storage.
            tuning: An optional dictionary to override default emission parameters.
        """
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}

        # Core State
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema = 1.0
        self.entropy_slope = 0.1
        self._silence_streak = 0
        self._last_triad_hit = False
        self._last_interval = self.tuning['base_interval']
        self._last_tick_hash: Optional[str] = None
        self._intent_source: Optional[str] = None # Set by upstream context

        # Lifecycle control
        self._pulse_active = False
        self._pulse_task: Optional[asyncio.Task] = None

        # Symbolic Tracking Modules
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()

        # Buffers and Traces
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
            'agent_emission_interval_seconds': self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id),
        }
        log.info(f"Initialized RecursiveAgentFT with agent_id={self.agent_id}")

    def track_lineage(self, parent: str, child: str) -> None:
        """
        Assigns a parent-child relationship between two motifs.
        Compliant with RFC-0005 §2.1.
        """
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        """
        Attempts to find a ghost trace matching the tick's field signature.
        Used for replaying motifs from archived field states (RFC-0005 §4.4).
        """
        key = tick.extensions.get('field_signature')
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get('motifs')
        return None

    async def start_continuous_emission(self) -> None:
        """
        The core autonomous symbolic pulse loop.
        Continuously emits ticks at an adaptive interval. Mandated by RFC-CORE-002 §4.2.
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
        Resolves a motif to its symbolic presence field signature.
        Falls back to default resonance fields if the task engine is unavailable.
        Logic specified in RFC-CORE-002 §6.2.
        """
        try:
            if self.symbolic_task_engine:
                result = self.symbolic_task_engine.resolve_presence_field([motif])
                if result:
                    return result
        except Exception:
            # Fallback if task engine fails or is not present
            pass

        if motif in {'silence', 'grief'}:
            return 'ψ-bind@Ξ'
        return 'ψ-resonance@Ξ'

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        """
        Constructs, annotates, and emits a single symbolic tick packet.

        This method builds the QuantumTickV2 object, enriches it with symbolic
        metadata like swirl vectors and coherence potential, and reports it to
        the observability monitor. It strictly adheres to the intent pass-through
        contract (RFC-0003 §6.2, RFC-0004 §2.5).
        """
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # Mirror upstream intent if provided. This agent does NOT act on intent;
        # it is a pass-through signal for downstream consumers like NoorFastTimeCore
        # or LogicalAgentAT. (RFC-0003 §6.2)
        if self._intent_source is not None:
            tick.extensions['intent'] = self._intent_source

        # Resolve field signature (RFC-CORE-002 §6.2)
        field_signature = self._resolve_field(motifs[-1] if motifs else 'silence')
        tick.extensions['field_signature'] = field_signature

        # Optional HMAC for integrity
        if hasattr(self, 'hmac_secret') and self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions['tick_hmac'] = tick_hmac

        # Update swirl and density trackers
        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)

        # Compute and attach diagnostic extensions
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions['swirl_vector'] = swirl_hash
        tick.extensions['coherence_potential'] = coherence

        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()

        # Report to monitor safely
        report_tick_safe(
            self.monitor,
            tick,
            coherence,
            self.density.snapshot(),
            swirl_hash
        )

        self.metrics['agent_ticks_emitted_total'].inc()
        return tick

    def start_emission(self) -> None:
        """
        Starts the symbolic emission loop.
        This is a required lifecycle method per RFC-CORE-002 §4.2.2.
        """
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        """
        Stops the symbolic emission loop and cancels the task gracefully.
        This is a required lifecycle method per RFC-CORE-002 §4.2.2.
        """
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pulse_task

    def observe_feedback(
        self,
        tick_id: str,
        reward: float,
        annotations: Dict[str, Any]
    ) -> None:
        """
        Integrates feedback from a logic agent to update internal state.

        This method updates the reward EMA and tracks triad completion to
        influence future emission cadence. (RFC-CORE-002 §2.3, RFC-0005 §4)
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
        Adaptively modulates the emission cadence based on reward and entropy.
        (RFC-CORE-002 §2.2)
        """
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning['entropy_boost_threshold']:
            adj *= 0.5  # Boost cadence in low-entropy states
        if self._last_triad_hit:
            adj *= 1.0 - self.tuning['triad_bias_weight']

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
        Selects symbolic seeds for the next emission based on recent history
        and memory recall. (RFC-CORE-002 §3.2)
        """
        motifs = list(self._last_motifs)
        if motifs and hasattr(self.memory, 'retrieve'):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:
                log.error("Failed to retrieve motifs from memory manager.")

        if not motifs:
            motifs = ['silence']

        return motifs[-3:]

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotates a feedback packet with phase identity and field state.
        This is a non-destructive merge per RFC-CORE-002 §8.2.2.
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
        Archives a symbolic tick into a stable, crystallized bundle.
        (RFC-CORE-002 §8.1)
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
        Compiles and exports an RFC-compliant feedback packet.

        This packet includes observability metadata and mirrors the `intent` signal
        from the most recent tick for diagnostic pass-through.
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

        # Pass-through intent from the last tick for observability.
        # This agent does not act on this signal. (RFC-0003 §6.2)
        if tick is not None and 'intent' in tick.extensions:
            # Ensure extensions key exists before updating
            if 'extensions' not in packet:
                packet['extensions'] = {}
            packet['extensions']['intent'] = tick.extensions['intent']

        self.metrics['agent_feedback_export_total'].inc()
        return packet

    def export_state(self) -> Dict[str, Any]:
        """Exports a minimal snapshot of the agent's runtime state."""
        return {
            "interval": self._last_interval,
            "reward_ema": self._reward_ema,
            "last_tick_hash": self._last_tick_hash,
        }

# --- Helper Functions ---

def compute_coherence_potential(
    reward_ema: float,
    entropy_slope: float,
    eps: float = 1e-6
) -> float:
    """
    Calculates a scalar signal for symbolic alignment strength.
    (RFC-CORE-002 §4.3)
    """
    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str
) -> None:
    """
    A non-blocking, safe wrapper to report a tick to the monitor.
    Ensures the agent's pulse loop is never delayed by observability hooks.
    (RFC-CORE-002 §8.2.3)
    """
    try:
        if monitor and hasattr(monitor, 'report_tick'):
            # In a real async environment, this might be `asyncio.create_task`
            # For simplicity, using a thread to avoid blocking the main loop.
            args = (tick, coherence_potential, motif_density, swirl_vector)
            threading.Thread(target=monitor.report_tick, args=args).start()
    except Exception as e:
        log.warning(f"Monitor callback failed: {e}")

# End_of_File