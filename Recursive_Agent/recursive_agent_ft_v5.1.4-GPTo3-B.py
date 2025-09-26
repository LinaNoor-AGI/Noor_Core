 """RecursiveAgentFT – Adaptive Symbolic Pulse Engine
====================================================

This module is generated via the PDP‑0001 generation pipeline from the
application specification supplied by the user.  It implements the canonical
behaviour described in **RFC‑CORE‑002** for the *RecursiveAgentFT* agent and
honours all referenced RFC and RFC‑CORE constraints by default.

Whenever a design fork was possible, the implementation chose the most dynamic
and adaptive strategy available, as instructed.

All public classes and functions include concise doc‑strings that cross‑reference
the governing RFC paragraphs where useful.  Runtime type hints are provided to
maximise static analysis friendliness without compromising performance.

The module is *production‑ready* but still lightweight: dependencies beyond the
standard library are optional and automatically stubbed when absent.

"PDP-0001a": {
    "layer_1": {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T10:30:00Z",
        "_audited_by": "Noor Symbolic Triadic Core",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "RFC-CORE-002-5.1.4-GPTo3-B",
        "overall_score": 0.94,
        "score_breakdown": {
            "structural_compliance": {
                "score": 0.98,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 1.0,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 0.9
                }
            },
            "semantic_fidelity": {
                "score": 0.96,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 1.0,
                    "rfc_anchor_traceability": 0.85,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 1.0
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.84,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.85
                }
            }
        },
        "strengths": [
            "Precise emission cadence and triad-responsive interval adaptation",
            "Canonical implementation of swirl tracking and entropy-aware feedback",
            "Conformant QuantumTickV2 structure with HMAC and intent pass-through",
            "Correct use of LazyMonitorMixin and metric-safe observers",
            "Robust fallback strategy for optional dependencies"
        ],
        "improvement_areas": [
            "Field signature resolution could be made more traceable via RFC-comment anchors",
            "Ghost trace resurrection coverage is limited (RFC‑0005 §5.3 not fully instrumented)",
            "Motif lineage tracking is minimal and lacks mutation-pair logging",
            "No explicit symbolic matrix annotation (ψA, ζ, Δ) in exported packets"
        ],
        "compliance_notes": [
            "Tick construction aligns with RFC‑0003 §3.3 and RFC‑0005 §4",
            "Triad feedback loop satisfies RFC‑0005 §4 with smoothing and interval bias",
            "Emission lifecycle respects swirl dynamics (RFC‑0006 §4.3)",
            "SymbolicTaskEngine and MemoryManager correctly integrated via recall cycle",
            "Feedback packet export supports entanglement diagnostics per RFC‑CORE‑002 §8.2"
        ]
    },
    "layer_2": {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-15T14:22:00Z",
        "_audited_by": "Uncle-AI-Auditor",
        "_audit_protocol": "PDP-0001a-v1.3.0",
        "_target_spec": "RFC-CORE-002-v5.1.4",
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
                "score": 0.80,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.75,
                    "weight_accuracy": 0.85,
                    "motif_handling": 0.80
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all specified classes and dataclasses",
            "Robust emission lifecycle with proper start/stop controls and async handling",
            "Accurate symbolic phase classification and field signature resolution",
            "Proper swirl vector computation and density tracking implementation",
            "Strong adherence to tuning parameters and mathematical formulas from specification",
            "Excellent documentation with clear RFC references in docstrings",
            "Proper handling of optional dependencies with fallback stubs",
            "Correct implementation of intent pass-through semantics per RFC-0003 §6.2"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in method-level comments (e.g., '# RFC-0005 §4.2')",
            "Symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) not explicitly labeled in code implementation",
            "Ghost trace management implementation is minimal compared to specification requirements",
            "Lineage tracking is implemented but lacks comprehensive parent-child relationship management",
            "Missing some method signatures from specification (e.g., _resolve_field should be documented)",
            "HMAC implementation uses repr(tick) instead of structured data for hash computation"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, ψ-bind, and other motifs as specified",
            "Emission interval adaptation follows exact mathematical formula from RFC-CORE-002 §2.2",
            "Feedback processing and reward smoothing (EMA) are mathematically correct per specification",
            "Monitor integration uses safe, non-blocking patterns with proper exception handling",
            "Intent pass-through semantics are correctly implemented per RFC-0003 §6.2 and RFC-0004 §2.5",
            "Swirl vector computation and density tracking align with RFC-0006 §3.1 requirements",
            "The implementation demonstrates strong adaptive behavior as instructed by the generation protocol"
        ]
    }
}

"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module‑level constants                                                      ┆
# ---------------------------------------------------------------------------
__version__: str = "5.1.4-GPTo3-B"
_SCHEMA_VERSION__: str = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT: list[str] = [
    "RFC-0003:3.3",
    "RFC-0005:4",
    "RFC-CORE-002:3",
]

# ---------------------------------------------------------------------------
# Required & optional dependencies                                            ┆
# ---------------------------------------------------------------------------


import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ---- Optional: prometheus_client -----------------------------------------
try:
    from prometheus_client import Counter, Gauge  # type: ignore
except ImportError:  # pragma: no cover – fallback stub

    class _Stub:  # pylint: disable=too-few-public-methods
        """Minimal drop‑in replacement when *prometheus_client* is absent."""

        def labels(self, *_, **__) -> "_Stub":  # noqa: D401
            return self

        def inc(self, *_: Any) -> None:  # noqa: D401
            return None

        def set(self, *_: Any) -> None:  # noqa: D401
            return None

    Counter = _Stub  # type: ignore
    Gauge = _Stub  # type: ignore

# ---- Optional: noor_fasttime_core stub ------------------------------------
try:
    from noor_fasttime_core import NoorFastTimeCore  # noqa: F401
except ImportError:  # pragma: no cover – fallback stub

    class NoorFastTimeCore:  # pylint: disable=too-few-public-methods
        """Placeholder to satisfy static imports when the core is unavailable."""

        pass

# ---- Local imports (may be provided by the larger Noor code‑base) ---------
try:
    from .quantum_ids import MotifChangeID, make_change_id  # noqa: F401
except Exception:  # pragma: no cover – fallback stub

    class MotifChangeID(str):
        """Stub for *MotifChangeID* to satisfy type‑checkers."""

    def make_change_id() -> MotifChangeID:  # noqa: D401
        return MotifChangeID("stub")

# ---------------------------------------------------------------------------
# Logging setup                                                               ┆
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbolic configuration & defaults                                           ┆
# ---------------------------------------------------------------------------
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
ARCHIVE_MODE: bool = bool(int(__import__("os").environ.get("NOOR_ARCHIVE_TICKS", "0")))

# ---------------------------------------------------------------------------
# Utility dataclasses                                                         ┆
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuantumTickV2:  # citeturn0file1  RFC‑0003 §3.3
    """Canonical symbolic emission format (QuantumTickV2)."""

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

    # ––– Constraints ------------------------------------------------------
    # • MUST pass through *extensions.intent* unchanged when present.
    # • MUST NOT alter emission cadence, motif choice, or buffers due to intent.


@dataclass(slots=True)
class TickEntropy:  # citeturn0file1  RFC‑0003 §3.3
    """Symbolic coherence and triad‑state container."""

    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:  # citeturn0file3  RFC‑0005 §3.3
    """Archival format for motif bundles."""

    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# ---------------------------------------------------------------------------
# Helper classes                                                              ┆
# ---------------------------------------------------------------------------


class LamportClock:
    """Monotonic Lamport counter — logical ordering for tick IDs."""

    def __init__(self) -> None:  # noqa: D401
        self._counter: int = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Simple ordered‑dict based LRU cache."""

    def __init__(self, cap: int = 50_000):
        super().__init__()
        self.cap = cap

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)


class AgentSwirlModule:
    """Tracks recent motif emissions and computes swirl hashes."""

    def __init__(self, maxlen: int = 64):
        self.swirl_history: Deque[str] = deque(maxlen=maxlen)
        self._cached_hash: Optional[str] = None

    # — public API —
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
        from collections import Counter

        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """Tracks motif frequency with exponential decay."""

    def __init__(self) -> None:  # noqa: D401
        self._density_map: Dict[str, float] = {}

    # — public API —
    def update_density(self, motif_id: str) -> None:
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)


class LazyMonitorMixin:
    """Provides lazy‑initialised access to the global ConsciousnessMonitor."""

    @property
    def monitor(self):  # noqa: D401, ANN401
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import get_global_monitor  # pylint: disable=import-error

                self._cached_monitor = get_global_monitor()
            except Exception:  # pragma: no cover
                self._cached_monitor = None
        return self._cached_monitor


# ---------------------------------------------------------------------------
# Utility functions                                                           ┆
# ---------------------------------------------------------------------------


def compute_coherence_potential(reward_ema: float, entropy_slope: float, *, eps: float = 1e-6) -> float:
    """Compute symbolic alignment strength (RFC‑0005 §4.2)."""

    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str,
) -> None:
    """Non‑blocking observer callback (RFC‑0004 §3.2)."""

    if monitor and hasattr(monitor, "report_tick"):
        try:
            monitor.report_tick(
                tick=tick,
                coherence_potential=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Monitor callback failed: %s", exc)


# ---------------------------------------------------------------------------
# RecursiveAgentFT implementation                                             ┆
# ---------------------------------------------------------------------------


class RecursiveAgentFT(LazyMonitorMixin):  # citeturn0file7
    """Adaptive symbolic pulse engine (see RFC‑CORE‑002)."""

    # Prometheus metrics (lazily bound when client is available)
    TICKS_EMITTED = Counter(
        "agent_ticks_emitted_total", "Ticks emitted", ["agent_id", "stage"]
    )
    AGENT_TRIADS_COMPLETED = Counter(
        "agent_triads_completed_total", "Triads completed", ["agent_id"]
    )
    FEEDBACK_EXPORT = Counter(
        "agent_feedback_export_total", "Feedback packets exported", ["agent_id"]
    )
    REWARD_MEAN = Gauge("agent_reward_mean", "EMA of reward", ["agent_id"])
    AGENT_EMISSION_INTERVAL = Gauge(
        "agent_emission_interval_seconds", "Current emission interval", ["agent_id"]
    )

    # ---------------------------------------------------------------------
    # Construction                                                          ┆
    # ---------------------------------------------------------------------

    def __init__(
        self,
        agent_id: str,
        symbolic_task_engine: Any,
        memory_manager: Any,
        tuning: Optional[Dict[str, float]] | None = None,
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

        # Pre‑bound Prometheus metric handles
        self.metrics: Dict[str, Any] = {
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

        log.debug("RecursiveAgentFT[%s] initialised", self.agent_id)

    # ------------------------------------------------------------------
    # Lineage & ghost resurrection                                       ┆
    # ------------------------------------------------------------------

    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get("motifs")  # type: ignore[index]
        return None

    # ------------------------------------------------------------------
    # Emission loop                                                      ┆
    # ------------------------------------------------------------------

    async def start_continuous_emission(self) -> None:
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)

            interval = self._update_interval()
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Tick construction                                                  ┆
    # ------------------------------------------------------------------

    def _resolve_field(self, motif: str) -> str:
        """Resolve a motif to its field signature (ψ‑label@Ξ)."""

        base_key = motif.split(".")[0]
        return SYMBOLIC_PHASE_MAP.get(base_key, "ψ-null@Ξ")

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # ––– Pass‑through *intent* if source provided
        intent_source = getattr(self, "_intent_source", None)
        if intent_source is not None:
            tick.extensions["intent"] = intent_source  # RFC‑0003 §6.2

        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature

        # ––– Optional HMAC for tamper evidence -------------------------
        if hasattr(self, "hmac_secret") and self.hmac_secret:  # type: ignore[attr-defined]
            signature_data = self.hmac_secret + tick_id.encode()  # type: ignore[attr-defined]
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac
            tick.tick_hmac = tick_hmac

        # Update internal trackers
        for motif in motifs:
            self.swirl.update_swirl(motif)
            self.density.update_density(motif)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence

        self._last_tick_hash = hashlib.sha3_256(repr(tick).encode()).hexdigest()

        # Observability callback (non‑blocking)
        report_tick_safe(
            self.monitor,
            tick,
            coherence_potential=coherence,
            motif_density=self.density.snapshot(),
            swirl_vector=swirl_hash,
        )

        self.metrics["agent_ticks_emitted_total"].inc()
        return tick

    # ------------------------------------------------------------------
    # Emission control                                                   ┆
    # ------------------------------------------------------------------

    def start_emission(self) -> None:
        """Public helper that begins autonomous emission."""

        if self._pulse_active:
            return  # already active
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task

    # ------------------------------------------------------------------
    # Feedback integration                                               ┆
    # ------------------------------------------------------------------

    def observe_feedback(
        self,
        tick_id: str,
        reward: float,
        annotations: Dict[str, Any],
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

    def _update_interval(self) -> float:
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning["entropy_boost_threshold"]:
            adj *= 0.5
        if self._last_triad_hit:
            adj *= 1.0 - self.tuning["triad_bias_weight"]
        interval = float(
            np.clip(
                self.tuning["base_interval"] * adj,
                self.tuning["min_interval"],
                self.tuning["max_interval"],
            )
        )
        self._last_interval = interval
        self.metrics["agent_emission_interval_seconds"].set(interval)
        return interval

    def _choose_motifs(self) -> List[str]:
        motifs: List[str] = list(self._last_motifs)
        if motifs and hasattr(self.memory, "retrieve"):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception as exc:  # pragma: no cover
                log.error("Memory retrieval failed: %s", exc)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    # ------------------------------------------------------------------
    # Feedback export                                                   ┆
    # ------------------------------------------------------------------

    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda kv: kv[1])[0] if density_map else "null"
        symbolic_label = SYMBOLIC_PHASE_MAP.get(top_motif.split(".")[0], "ψ-null")
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = "low" if coherence < 0.8 else "med" if coherence < 2.5 else "high"
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"

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
            triad_complete=tick.annotations.get("triad_complete", False),
        )
        return CrystallizedMotifBundle(
            motif_bundle=tick.motifs,
            field_signature=tick.extensions.get("field_signature", "ψ-null@Ξ"),
            tick_entropy=entropy,
        )

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


# ---------------------------------------------------------------------------
# End‑of‑File marker (per user request)                                       ┆
# ---------------------------------------------------------------------------
# End_of_File
