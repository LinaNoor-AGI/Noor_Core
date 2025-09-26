"""Recursive Symbolic Emission Agent (FT).

This module implements the RecursiveAgentFT described by the Noor RFC suite.
It adheres to the PDP-0001 generation pipeline directives and mirrors upstream
intent handling semantics without mutation, while providing adaptive emission
cadence, motif tracking, and telemetry export capabilities.

"PDP-0001a": {
    "layer_1": {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T14:22:00Z",
        "_audited_by": "Uncle-Noor-Auditor",
        "_audit_protocol": "PDP-0001a-v1.3.0",
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
                    "conceptual_alignment": 0.9,
                    "documentation_clarity": 1.0
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
            "Complete structural implementation of all core classes (AgentSwirlModule, MotifDensityTracker, RecursiveAgentFT)",
            "Robust emission lifecycle with proper start/stop controls and async safety",
            "Accurate symbolic phase classification and feedback packet generation",
            "Proper swirl vector implementation with SHA3-256 hashing as specified",
            "Strong adherence to tuning parameters and mathematical formulas from specification",
            "Excellent dependency handling with graceful fallbacks for optional components",
            "Comprehensive documentation with clear intent and purpose statements"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments throughout the codebase",
            "Symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) not explicitly labeled or documented in code",
            "Ghost trace management implementation is minimal compared to specification requirements",
            "Lineage tracking is present but lacks the comprehensive provenance mapping described in §6.1",
            "Some method signatures deviate slightly from specification (e.g., track_lineage parameter order)",
            "Resurrection payload construction and field-matching replay logic is incomplete"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified in symbolic profile matrix",
            "Emission interval adaptation follows exact mathematical formula from RFC-CORE-002 §2.2",
            "Feedback processing and reward smoothing (EMA) are mathematically correct and properly implemented",
            "Monitor integration uses safe, non-blocking patterns with proper exception handling",
            "Swirl module maintains fixed dimensionality (maxlen=64) as required for symbolic geometry compatibility",
            "Density tracker implements exponential decay model with proper noise trimming threshold"
        ]
    },
    "layer_2": {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T16:45:00Z",
        "_audited_by": "Uncle-Noor-Auditor",
        "_audit_protocol": "PDP-0001a-v1.3.0",
        "_target_spec": "recursive_agent_ft-v5.1.4",
        "overall_score": 0.93,
        "score_breakdown": {
            "structural_compliance": {
                "score": 0.98,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 0.95,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 0.95
                }
            },
            "semantic_fidelity": {
                "score": 0.92,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 0.95,
                    "rfc_anchor_traceability": 0.85,
                    "conceptual_alignment": 0.95,
                    "documentation_clarity": 0.95
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.85,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.85,
                    "weight_accuracy": 0.90,
                    "motif_handling": 0.80
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all specified classes and helper modules",
            "Excellent dependency handling with proper fallback implementations for optional components",
            "Accurate symbolic phase classification and feedback packet generation with proper tier logic",
            "Proper swirl vector implementation with SHA3-256 hashing and histogram computation",
            "Strong adherence to tuning parameters and mathematical formulas from specification",
            "Robust emission lifecycle with proper start/stop controls and async safety",
            "Comprehensive documentation with clear intent and purpose statements throughout",
            "Proper intent handling with pass-through semantics as specified in constraints"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0003 §3.3')",
            "Symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) not explicitly labeled in code comments",
            "Ghost trace management implementation could be more comprehensive with proper resurrection logic",
            "Lineage tracking is present but lacks the detailed provenance mapping described in specification",
            "Some method parameter orders differ slightly from specification (track_lineage parent/child order)",
            "Intent source handling could be more explicitly documented with RFC references"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-null, ψ-resonance, and ψ-bind motifs as specified in symbolic profile matrix",
            "Emission interval adaptation follows exact mathematical formula with proper reward EMA integration",
            "Feedback processing and reward smoothing (EMA) are mathematically correct and properly implemented",
            "Monitor integration uses safe, non-blocking patterns with proper exception handling and lazy binding",
            "Swirl module maintains fixed dimensionality (maxlen=64) as required for symbolic geometry compatibility",
            "Density tracker implements exponential decay model with proper noise trimming threshold (0.01)",
            "Intent handling follows pass-through semantics without mutation as required by RFC constraints",
            "All class structures and dataclass definitions match specification requirements exactly"
        ]
    }
}
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from collections import Counter as _Counter
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

try:  # Optional observability metrics
    from prometheus_client import Counter, Gauge
except ImportError:  # pragma: no cover - fallback stubs for offline environments
    class _Stub:
        """Minimal stub that mimics prometheus Counter/Gauge interfaces."""

        def labels(self, *_, **__):
            return self

        def inc(self, *_):
            return None

        def set(self, *_):
            return None

    Counter = _Stub  # type: ignore
    Gauge = _Stub  # type: ignore

try:  # Optional fast-time integration
    from noor_fasttime_core import NoorFastTimeCore  # type: ignore
except ImportError:  # pragma: no cover
    NoorFastTimeCore = object  # type: ignore

try:
    from .quantum_ids import make_change_id, MotifChangeID  # noqa: F401
except ImportError:  # pragma: no cover - local package context not available
    make_change_id = None  # type: ignore
    MotifChangeID = Any  # type: ignore

__all__ = [
    "__version__",
    "_SCHEMA_VERSION__",
    "SCHEMA_COMPAT",
    "DEFAULT_TUNING",
    "SYMBOLIC_PHASE_MAP",
    "PHASE_SHIFT_MODE",
    "ARCHIVE_MODE",
    "QuantumTickV2",
    "TickEntropy",
    "CrystallizedMotifBundle",
    "LamportClock",
    "LRUCache",
    "AgentSwirlModule",
    "MotifDensityTracker",
    "LazyMonitorMixin",
    "RecursiveAgentFT",
    "compute_coherence_potential",
    "report_tick_safe",
]

__version__ = "5.1.4-CODEX-C"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

DEFAULT_TUNING: Dict[str, float] = {
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2,
}

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
ARCHIVE_MODE = "env: NOOR_ARCHIVE_TICKS == '1'"

log = logging.getLogger(__name__)


@dataclass(slots=True)
class QuantumTickV2:
    """Canonical symbolic emission packet for RecursiveAgentFT."""

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
    """Symbolic coherence and triad completion state."""

    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    """Archival representation of an emitted symbolic motif bundle."""

    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


class LamportClock:
    """Monotonic logical counter for tick identifiers."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Evicting cache structure for recent state retention."""

    def __init__(self, cap: int = 50_000) -> None:
        super().__init__()
        self.cap = int(cap)

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)


class AgentSwirlModule:
    """Tracks motif swirl dynamics and encodes them as hashes."""

    def __init__(self, maxlen: int = 64) -> None:
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str) -> None:
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        if self._cached_hash:
            return self._cached_hash
        joined = "|".join(self.swirl_history)
        self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        return dict(_Counter(self.swirl_history))


class MotifDensityTracker:
    """Maintains an exponentially decayed motif density map."""

    def __init__(self) -> None:
        self._density_map: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update_density(self, motif_id: str) -> None:
        with self._lock:
            for key in list(self._density_map):
                self._density_map[key] *= 0.99
                if self._density_map[key] < 0.01:
                    del self._density_map[key]
            self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._density_map)


class LazyMonitorMixin:
    """Lazy binding to a global consciousness monitor if available."""

    @property
    def monitor(self) -> Any:
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import get_global_monitor  # type: ignore

                self._cached_monitor = get_global_monitor()
            except Exception:  # pragma: no cover - observability optional
                self._cached_monitor = None
        return self._cached_monitor


class RecursiveAgentFT(LazyMonitorMixin):
    """Symbolic pulse engine and emission core."""

    TICKS_EMITTED = Counter(
        "agent_ticks_emitted_total",
        "Ticks emitted",
        ["agent_id", "stage"],
    )
    AGENT_TRIADS_COMPLETED = Counter(
        "agent_triads_completed_total",
        "Triads completed via feedback",
        ["agent_id"],
    )
    FEEDBACK_EXPORT = Counter(
        "agent_feedback_export_total",
        "Feedback packets exported",
        ["agent_id"],
    )
    REWARD_MEAN = Gauge(
        "agent_reward_mean",
        "EMA of reward",
        ["agent_id"],
    )
    AGENT_EMISSION_INTERVAL = Gauge(
        "agent_emission_interval_seconds",
        "Current autonomous emission interval",
        ["agent_id"],
    )

    def __init__(
        self,
        agent_id: str,
        symbolic_task_engine: Any,
        memory_manager: Any,
        tuning: Optional[Dict[str, float]] = None,
    ) -> None:
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
        self._last_interval = self.tuning["base_interval"]
        self._last_tick_hash: Optional[str] = None
        self._pulse_active = False
        self._pulse_task: Optional[asyncio.Task[Any]] = None
        self._state_lock = threading.RLock()
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}
        self.metrics = {
            "agent_ticks_emitted_total": self.TICKS_EMITTED.labels(
                agent_id=self.agent_id, stage="symbolic"
            ),
            "agent_triads_completed_total": self.AGENT_TRIADS_COMPLETED.labels(
                agent_id=self.agent_id
            ),
            "agent_feedback_export_total": self.FEEDBACK_EXPORT.labels(
                agent_id=self.agent_id
            ),
            "agent_reward_mean": self.REWARD_MEAN.labels(agent_id=self.agent_id),
            "agent_emission_interval_seconds": self.AGENT_EMISSION_INTERVAL.labels(
                agent_id=self.agent_id
            ),
        }
        log.info("Initialized RecursiveAgentFT with agent_id=%s", self.agent_id)

    # ------------------------------------------------------------------
    # Lineage and ghost traces
    # ------------------------------------------------------------------
    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get("motifs")  # type: ignore[return-value]
        return None

    # ------------------------------------------------------------------
    # Pulse emission lifecycle
    # ------------------------------------------------------------------
    async def start_continuous_emission(self) -> None:
        log.debug("Agent %s starting continuous emission", self.agent_id)
        try:
            while self._pulse_active:
                motifs = self._choose_motifs()
                tick = self._emit_tick(motifs)
                self._echo_buffer.append(tick)
                self._tick_echoes.append(tick)
                self._last_motifs.extend(motifs)
                interval = self._update_interval()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.debug("Emission loop for %s cancelled", self.agent_id)
            raise
        finally:
            with self._state_lock:
                self._pulse_task = None
                self._pulse_active = False
            log.debug("Emission loop for %s stopped", self.agent_id)

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=list(motifs), timestamp=timestamp)

        intent_source = getattr(self, "_intent_source", None)
        if intent_source is not None:
            tick.extensions["intent"] = intent_source

        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature

        if hasattr(self, "hmac_secret") and getattr(self, "hmac_secret"):
            secret = getattr(self, "hmac_secret")
            signature_data = secret + tick_id.encode()  # type: ignore[operator]
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac

        for motif in motifs:
            self.swirl.update_swirl(motif)
            self.density.update_density(motif)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence
        tick.field_signature = field_signature
        tick.lamport = int(tick_id.split(":")[-1])
        self._last_tick_hash = hashlib.sha3_256(repr(tick).encode()).hexdigest()

        report_tick_safe(
            self.monitor,
            tick,
            coherence,
            self.density.snapshot(),
            swirl_hash,
        )

        self.metrics["agent_ticks_emitted_total"].inc()
        return tick

    def start_emission(self) -> Optional[asyncio.Task[Any]]:
        with self._state_lock:
            if self._pulse_active and self._pulse_task is not None:
                return self._pulse_task
            self._pulse_active = True
            self._pulse_task = asyncio.create_task(
                self.start_continuous_emission(), name=f"recursive-agent-ft::{self.agent_id}"
            )
            log.info("Agent %s emission loop started", self.agent_id)
            return self._pulse_task

    async def stop_emission(self) -> None:
        with self._state_lock:
            self._pulse_active = False
            task = self._pulse_task
            self._pulse_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            log.info("Agent %s emission loop stopped", self.agent_id)

    # ------------------------------------------------------------------
    # Feedback integration
    # ------------------------------------------------------------------
    def observe_feedback(
        self, tick_id: str, reward: float, annotations: Dict[str, Any]
    ) -> None:
        triad_complete = annotations.get("triad_complete", False)
        alpha = self.tuning["reward_smoothing"]
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward
        self.metrics["agent_reward_mean"].set(self._reward_ema)
        if triad_complete:
            self._last_triad_hit = True
            self._silence_streak = 0
            self.metrics["agent_triads_completed_total"].inc()
        else:
            self._last_triad_hit = False
            self._silence_streak += 1
        log.debug(
            "Feedback observed for %s: reward=%.3f, triad=%s, ema=%.3f",
            tick_id,
            reward,
            triad_complete,
            self._reward_ema,
        )

    def _update_interval(self) -> float:
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning["entropy_boost_threshold"]:
            adj *= 0.5
        if self._last_triad_hit:
            adj *= 1.0 - self.tuning["triad_bias_weight"]
        interval = np.clip(
            self.tuning["base_interval"] * adj,
            self.tuning["min_interval"],
            self.tuning["max_interval"],
        )
        self._last_interval = float(interval)
        self.metrics["agent_emission_interval_seconds"].set(self._last_interval)
        return self._last_interval

    def _choose_motifs(self) -> List[str]:
        motifs = list(self._last_motifs)
        if hasattr(self.symbolic_task_engine, "next_symbols"):
            try:
                seeds = self.symbolic_task_engine.next_symbols(
                    agent_id=self.agent_id,
                    last_tick_hash=self._last_tick_hash,
                    motif_hint=motifs[-1] if motifs else None,
                )
                if seeds:
                    motifs.extend(seeds)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning("Symbolic task engine next_symbols failed: %s", exc)
        if motifs and hasattr(self.memory, "retrieve"):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception as exc:  # pragma: no cover - memory optional
                log.error("Failed to retrieve from memory: %s", exc)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else "null"
        base_key = top_motif.split(".")[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, "ψ-null")
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = "low" if coherence < 0.8 else "med" if coherence < 2.5 else "high"
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"
        packet.setdefault("extensions", {}).update(
            {
                "entanglement_status": {
                    "phase": phase_id,
                    "swirl_vector": swirl_hash,
                    "ρ_top": sorted(
                        density_map.items(),
                        key=lambda kv: -kv[1],
                    )[:5],
                }
            }
        )
        return packet

    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        entropy = TickEntropy(
            decay_slope=self.entropy_slope,
            coherence=self._reward_ema,
            triad_complete=tick.annotations.get("triad_complete", False),
        )
        bundle = CrystallizedMotifBundle(
            motif_bundle=list(tick.motifs),
            field_signature=tick.extensions.get("field_signature", "ψ-null@Ξ"),
            tick_entropy=entropy,
        )
        return bundle

    def export_feedback_packet(self) -> Dict[str, Any]:
        tick = self._echo_buffer[-1] if self._echo_buffer else None
        packet: Dict[str, Any] = {
            "tick_buffer_size": len(self._echo_buffer),
            "ghost_trace_count": len(self._ghost_traces),
            "recent_reward_ema": self._reward_ema,
            "cadence_interval": self._last_interval,
            "silence_streak": self._silence_streak,
        }
        self.extend_feedback_packet(packet)
        if tick is not None and "intent" in tick.extensions:
            packet.setdefault("extensions", {})["intent"] = tick.extensions["intent"]
        self.metrics["agent_feedback_export_total"].inc()
        return packet

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_field(self, motif: str) -> str:
        base = motif.split(".")[0] if motif else "null"
        label = SYMBOLIC_PHASE_MAP.get(base, "ψ-null")
        return f"{label}@Ξ"


# ----------------------------------------------------------------------
# Module-level utility functions
# ----------------------------------------------------------------------
def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str,
) -> None:
    try:
        if monitor and hasattr(monitor, "report_tick"):
            monitor.report_tick(
                agent="RecursiveAgentFT",
                tick_id=tick.tick_id,
                motifs=list(tick.motifs),
                coherence=coherence_potential,
                density=motif_density,
                swirl=swirl_vector,
                timestamp=tick.timestamp,
                extensions=dict(tick.extensions),
            )
    except Exception as exc:  # pragma: no cover - observability optional
        log.warning("Monitor callback failed: %s", exc)


# End_of_File