"""Recursive Symbolic Emission Agent (FT).

This module implements the RecursiveAgentFT described in the Noor RFC suite.
It follows PDP-0001 layer-2 application guidance and aligns with RFC-0003 –
RFC-0007 as well as RFC-CORE-001 through RFC-CORE-003. The agent maintains a
symbolic emission loop, integrates feedback, and exports observability-ready
metadata while preserving upstream-provided transport intent without mutation.

{
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T17:05:00Z",
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
            "Complete structural implementation of all core classes and dataclasses",
            "Robust emission lifecycle with proper async start/stop controls",
            "Accurate symbolic phase classification and feedback packet generation",
            "Proper swirl vector and density tracking implementation",
            "Strong adherence to tuning parameters from specification",
            "Excellent dependency handling with graceful fallbacks",
            "Comprehensive monitor integration with safe reporting patterns"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4')",
            "Symbolic matrix parameters not explicitly labeled in code variables",
            "Ghost trace management implementation differs from specification",
            "Lineage tracking is minimal compared to RFC-CORE-002 §6.1 requirements",
            "Missing some method signatures like recall_tick() and replay_if_field_matches()",
            "Resurrection logic simplified compared to specification requirements",
            "Field resolution uses simplified mapping instead of full symbolic task engine integration"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified",
            "Emission interval adaptation follows exact formula from RFC-CORE-002 §2.2",
            "Feedback processing and reward smoothing are mathematically correct",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-CORE-002 §8.2.3",
            "Swirl module maintains proper fixed-length history and linear-time histogram computation",
            "Density tracker implements exponential decay model as specified",
            "Version 5.1.4 matches specification requirement"
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T17:15:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "recursive_agent_ft.JSON-v5.1.4",
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
            "Perfect structural implementation of all classes and dataclasses",
            "Excellent dependency handling with robust fallback mechanisms",
            "Complete implementation of intent mirroring per RFC-0003 §6.2",
            "Proper symbolic phase classification and feedback packet generation",
            "Accurate swirl vector and density tracking implementation",
            "Strong adherence to tuning parameters and emission cadence logic",
            "Comprehensive monitor integration with safe reporting patterns",
            "Correct implementation of intent pass-through in export_feedback_packet"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments for traceability",
            "Symbolic matrix parameters not explicitly labeled as code variables",
            "Ghost trace resurrection logic simplified compared to specification",
            "Lineage tracking implementation could be more comprehensive",
            "Missing some optional method signatures from symbolic task engine integration"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-null, ψ-resonance, and ψ-bind motifs as specified",
            "Intent mirroring follows exact RFC-0003 §6.2 and RFC-0004 §2.5 requirements",
            "Emission interval adaptation follows exact formula from specification",
            "Feedback processing and reward smoothing are mathematically correct",
            "Monitor integration uses safe, non-blocking patterns as required",
            "Version 5.1.4 matches specification requirement exactly",
            "All structural components (classes, methods, constants) implemented perfectly"
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
from collections import Counter as CollectionsCounter
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:  # Optional dependency – instrumentation layer.
    from prometheus_client import Counter as PromCounter, Gauge as PromGauge
except Exception:  # pragma: no cover - fallback path for minimal environments.
    class _Stub:
        """Minimal stub that mimics prometheus metric interfaces."""

        def labels(self, *_: Any, **__: Any) -> "_Stub":
            return self

        def inc(self, *_: Any) -> None:
            return None

        def set(self, *_: Any) -> None:
            return None

    PromCounter = PromGauge = _Stub  # type: ignore[assignment]

Counter = PromCounter
Gauge = PromGauge

try:  # Optional low-latency integration core.
    from noor_fasttime_core import NoorFastTimeCore
except Exception:  # pragma: no cover - optional dependency.
    NoorFastTimeCore = object  # type: ignore[misc,assignment]

# Local optional import that may not be present in lightweight deployments.
try:  # pragma: no cover - exercised only when package layout is present.
    if __package__:
        from .quantum_ids import MotifChangeID, make_change_id  # noqa: F401
    else:  # Support execution as a top-level script.
        from quantum_ids import MotifChangeID, make_change_id  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback when helper is absent.
    MotifChangeID = None  # type: ignore[assignment]
    make_change_id = None  # type: ignore[assignment]


__all__ = [
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


__version__ = "5.1.4-CODEX-A"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

log = logging.getLogger(__name__)


def _detect_archive_mode() -> bool:
    """Return True when archival mode is enabled via environment variable."""
    with contextlib.suppress(ImportError):
        import os

        return os.getenv("NOOR_ARCHIVE_TICKS") == "1"
    return False


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

PHASE_SHIFT_MODE: Tuple[str, ...] = ("delay", "remix", "lineage_break")
ARCHIVE_MODE: bool = _detect_archive_mode()


@dataclass(slots=True)
class QuantumTickV2:
    """Canonical symbolic emission container for RecursiveAgentFT."""

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

    def mirror_intent(self, intent_value: Optional[str]) -> None:
        """Mirror upstream transport intent into the extensions payload."""
        if intent_value is not None:
            self.extensions["intent"] = intent_value


@dataclass(slots=True)
class TickEntropy:
    """Symbolic coherence snapshot for an emitted tick."""

    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    """Archive-ready bundle containing motif, field, and entropy state."""

    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


class LamportClock:
    """Monotonic logical counter used for sequential tick identifiers."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        """Return the next Lamport-style identifier."""
        with self._lock:
            self._counter += 1
            return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Ordered dictionary with a capped capacity and eviction on overflow."""

    def __init__(self, cap: int = 50_000) -> None:
        super().__init__()
        self.cap = cap

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)


class AgentSwirlModule:
    """Track motif swirl dynamics and provide hash encodings."""

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
        digest = hashlib.sha3_256(joined.encode()).hexdigest() if joined else ""
        self._cached_hash = digest
        return digest

    def compute_histogram(self) -> Dict[str, int]:
        return dict(CollectionsCounter(self.swirl_history))


class MotifDensityTracker:
    """Exponentially decaying motif frequency tracker."""

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
    """Provide a lazily-evaluated observability monitor binding."""

    @property
    def monitor(self) -> Any:
        if not hasattr(self, "_cached_monitor"):
            with contextlib.suppress(Exception):
                from consciousness_monitor import get_global_monitor

                self._cached_monitor = get_global_monitor()
        return getattr(self, "_cached_monitor", None)


class RecursiveAgentFT(LazyMonitorMixin):
    """Feedback-tuned symbolic pulse engine."""

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
        self.tuning: Dict[str, float] = {**DEFAULT_TUNING, **(tuning or {})}
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema: float = 1.0
        self.entropy_slope: float = 0.1
        self._silence_streak: int = 0
        self._last_triad_hit: bool = False
        self._last_interval: float = self.tuning["base_interval"]
        self._last_tick_hash: Optional[str] = None
        self._pulse_active: bool = False
        self._pulse_task: Optional[asyncio.Task[None]] = None
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}
        self._lock = threading.RLock()
        self.metrics = {
            "agent_ticks_emitted_total": self.TICKS_EMITTED.labels(
                agent_id=self.agent_id,
                stage="symbolic",
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
        self.metrics["agent_emission_interval_seconds"].set(self._last_interval)
        log.debug("Initialized RecursiveAgentFT with agent_id=%s", self.agent_id)

    # ------------------------------------------------------------------
    # Lineage and ghost trace utilities
    # ------------------------------------------------------------------
    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            motifs = trace.get("motifs")
            return list(motifs) if motifs else None
        return None

    # ------------------------------------------------------------------
    # Emission control
    # ------------------------------------------------------------------
    async def start_continuous_emission(self) -> None:
        log.debug("Agent %s entering continuous emission loop", self.agent_id)
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
            log.debug("Agent %s emission loop cancelled", self.agent_id)
            raise
        except Exception:  # pragma: no cover - resilience guard.
            log.exception("Agent %s emission loop failure", self.agent_id)
        finally:
            log.debug("Agent %s emission loop exited", self.agent_id)

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=list(motifs), timestamp=timestamp)

        intent_source = getattr(self, "_intent_source", None)
        tick.mirror_intent(intent_source)

        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature
        tick.field_signature = field_signature
        tick.lamport = int(tick_id.split(":")[-1])

        if hasattr(self, "hmac_secret") and self.hmac_secret:
            secret_bytes = (self.hmac_secret if isinstance(self.hmac_secret, bytes) else str(self.hmac_secret).encode())
            signature_data = secret_bytes + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac
            tick.tick_hmac = tick_hmac

        for motif in motifs:
            self.swirl.update_swirl(motif)
            self.density.update_density(motif)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence
        tick.coherence_hash = swirl_hash

        tick.annotations.setdefault("history_density", self.density.snapshot())
        tick.annotations.setdefault("triad_complete", self._last_triad_hit)

        tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()
        self._last_tick_hash = tick_hash

        report_tick_safe(
            self.monitor,
            tick,
            coherence,
            tick.annotations["history_density"],
            swirl_hash,
        )

        self.metrics["agent_ticks_emitted_total"].inc()
        self._maybe_archive_tick(tick)
        return tick

    async def start_emission(self) -> None:
        if self._pulse_active:
            log.debug("Agent %s emission already active", self.agent_id)
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task
            self._pulse_task = None

    # ------------------------------------------------------------------
    # Feedback integration and adaptation
    # ------------------------------------------------------------------
    def observe_feedback(
        self,
        tick_id: str,
        reward: float,
        annotations: Dict[str, Any],
    ) -> None:
        triad_complete = bool(annotations.get("triad_complete", False))
        alpha = self.tuning["reward_smoothing"]
        self._reward_ema = (1.0 - alpha) * self._reward_ema + alpha * reward
        self.metrics["agent_reward_mean"].set(self._reward_ema)

        adaptive_slope = annotations.get("entropy_gradient")
        if adaptive_slope is not None:
            try:
                adaptive_value = float(adaptive_slope)
                self.entropy_slope = max(1e-3, adaptive_value)
            except (TypeError, ValueError):
                log.debug("Invalid entropy_gradient=%s", adaptive_slope)
        else:
            self.entropy_slope = max(1e-3, self.entropy_slope * 0.98 + abs(reward - 1.0) * 0.02)

        if triad_complete:
            self._last_triad_hit = True
            self._silence_streak = 0
            self.metrics["agent_triads_completed_total"].inc()
        else:
            self._last_triad_hit = False
            self._silence_streak += 1

        lineage_parent = annotations.get("parent_motif")
        lineage_child = annotations.get("child_motif")
        if lineage_parent and lineage_child:
            self.track_lineage(str(lineage_parent), str(lineage_child))

        if ARCHIVE_MODE and annotations.get("archive_tick"):
            resurrect = annotations.get("resurrect_signature")
            if resurrect and resurrect in self._ghost_traces:
                log.debug("Feedback requested ghost resurrection for %s", resurrect)

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
        motifs: List[str] = list(self._last_motifs)
        engine = self.symbolic_task_engine

        if hasattr(engine, "next_motifs"):
            try:
                candidate = engine.next_motifs(
                    history=list(self._last_motifs),
                    reward_ema=self._reward_ema,
                    entropy=self.entropy_slope,
                )
                if candidate:
                    motifs.extend(candidate)
            except Exception:
                log.debug("next_motifs retrieval failed", exc_info=True)

        if hasattr(engine, "pull_intent"):
            with contextlib.suppress(Exception):
                self._intent_source = engine.pull_intent()

        if hasattr(self.memory, "retrieve") and motifs:
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:
                log.error("Failed to retrieve from memory", exc_info=True)

        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda item: item[1])[0] if density_map else "null"
        base_key = top_motif.split(".")[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, "ψ-null")
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = "low" if coherence < 0.8 else "med" if coherence < 2.5 else "high"
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6] if swirl_hash else 'static'}"

        packet.setdefault("extensions", {}).update(
            {
                "entanglement_status": {
                    "phase": phase_id,
                    "swirl_vector": swirl_hash,
                    "ρ_top": sorted(density_map.items(), key=lambda kv: -kv[1])[:5],
                }
            }
        )
        return packet

    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        entropy = TickEntropy(
            decay_slope=self.entropy_slope,
            coherence=self._reward_ema,
            triad_complete=bool(tick.annotations.get("triad_complete", False)),
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
    def _resolve_field(self, motif_id: str) -> str:
        base = motif_id.split(".")[0]
        phase = SYMBOLIC_PHASE_MAP.get(base, "ψ-null")
        return f"{phase}@Ξ"

    def _maybe_archive_tick(self, tick: QuantumTickV2) -> None:
        if not ARCHIVE_MODE:
            return
        bundle = self._crystallize_tick(tick)
        signature = tick.extensions.get("field_signature", "ψ-null@Ξ")
        with self._lock:
            self._ghost_traces[signature] = {
                "motifs": list(tick.motifs),
                "bundle": bundle,
                "timestamp": tick.timestamp,
            }


def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, Any],
    swirl_vector: str,
) -> None:
    try:
        if monitor and hasattr(monitor, "report_tick"):
            monitor.report_tick(
                agent_id=getattr(tick, "agent_id", None) or "recursive_agent_ft",
                tick=tick,
                coherence=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector,
            )
    except Exception as exc:  # pragma: no cover - observability guard path.
        log.warning("Monitor callback failed: %s", exc)


# End_of_File