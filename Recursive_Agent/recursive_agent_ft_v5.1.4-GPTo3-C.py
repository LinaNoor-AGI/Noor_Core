# recursive_agent_ft.py
"""Recursive Symbolic Emission Agent (FT)

This module implements the **RecursiveAgentFT** class and accompanying data
structures, as defined by the Noor specification bundle (PDP‑0001 layer_2
application‑spec and RFC / RFC‑CORE hierarchy).

The implementation closely follows the formal JSON spec sections:
  • **Module‑Level Constants**
  • **External and Optional Dependencies** (with graceful fallbacks)
  • **Symbolic Configuration and Emission Defaults**
  • **Data‑Classes** for *QuantumTickV2*, *TickEntropy*, *CrystallizedMotifBundle*
  • **Helper Classes** (LamportClock, LRUCache, AgentSwirlModule,
    MotifDensityTracker, LazyMonitorMixin)
  • **RecursiveAgentFT** engine with adaptive cadence, swirl tracking, feedback
    integration, ghost‑trace resurrection, and Prometheus metrics.

Wherever the JSON spec left multiple choices open, the implementation prefers
*dynamic* and *adaptive* behaviours (e.g., entropy‑weighted interval scaling) as
requested by the user.

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
                "score": 0.88,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 0.9,
                    "rfc_anchor_traceability": 0.7,
                    "conceptual_alignment": 0.9,
                    "documentation_clarity": 0.9
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.80,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.8,
                    "motif_handling": 0.8
                }
            }
        },
        "strengths": [
            "Complete structural implementation of all core classes and data structures",
            "Robust emission lifecycle with proper start/stop controls and async handling",
            "Accurate symbolic phase classification and feedback packet generation",
            "Proper swirl vector and density tracking implementation following RFC-0006 §4.3",
            "Strong adherence to tuning parameters and default values from specification",
            "Graceful dependency fallback handling for optional packages",
            "Proper motif selection logic with memory recall and silence fallback"
        ],
        "improvement_areas": [
            "Missing explicit RFC section anchors in comments (e.g., '# RFC-0005 §4')",
            "Some method signatures deviate slightly from specification (e.g., try_ghost_resurrection)",
            "Ghost trace management implementation is less comprehensive than specified",
            "Lineage tracking implementation is minimal compared to RFC-0005 §5 requirements",
            "Monitor integration could be more robust with proper lazy binding",
            "Symbolic matrix parameters not explicitly labeled in code comments"
        ],
        "compliance_notes": [
            "Implementation correctly handles ψ-resonance, ψ-null, and ψ-hold motifs as specified in symbolic profile matrix",
            "Emission interval adaptation follows exact formula from RFC-CORE-002 §2.2 with proper clamping",
            "Feedback processing and reward smoothing are mathematically correct with proper EMA calculation",
            "Swirl vector implementation uses SHA3-256 hashing as required by RFC-0006 §4.3",
            "Coherence potential calculation follows RFC-0005 §4.2 specification exactly",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2"
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-14T19:16:00Z",
        "_audited_by": "Gemini 2.5 Pro",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "agent.recursive.ft-v5.1.4",
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
                "score": 0.95,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 1.0,
                    "rfc_anchor_traceability": 0.8,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 1.0
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.82,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 1.0,
                    "weight_accuracy": 0.5,
                    "motif_handling": 1.0
                }
            }
        },
        "strengths": [
            "Perfect structural implementation; all specified classes, methods, and constants are present and correctly defined.",
            "Logic flow is a direct and accurate translation of the specification's pseudocode and descriptions.",
            "Graceful handling of optional dependencies (Prometheus) with a fallback stub is implemented exactly as required.",
            "Conceptual alignment is excellent, capturing the essence of a feedback-tuned pulse engine.",
            "Documentation is clear, well-structured, and uses the correct terminology from the Noor lore."
        ],
        "improvement_areas": [
            "The specific numeric 'weights' from the _symbolic_profile_matrix (e.g., ζ: 0.87, E: 0.96) are not used in the code's logic, which instead relies on the DEFAULT_TUNING constants. This is the most significant fidelity gap.",
            "RFC anchors are present in docstrings as general references (e.g., 'RFC-0003 §6.2') but are not used as explicit, targeted comments within method bodies as suggested by the audit protocol's ideal traceability."
        ],
        "compliance_notes": [
            "The implementation correctly mirrors 'intent' from an upstream source into the tick's extensions without mutation, adhering to the constraints of RFC-0003 §6.2 and RFC-0004 §2.5.",
            "The adaptive cadence logic in '_update_interval' precisely follows the specified formula, incorporating reward EMA, entropy, and triad bias.",
            "The conceptual roles of the symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) are correctly implemented in spirit, even though their specific numeric weights are not."
        ]
    }
}
"""

# ── 1. Module‑Level Constants ────────────────────────────────────────────────
__version__: str = "5.1.4-GPTo3-C"
_SCHEMA_VERSION__: str = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT: list[str] = [
    "RFC-0003:3.3",
    "RFC-0005:4",
    "RFC-CORE-002:3",
]

# ── 2. External and Optional Dependencies ────────────────────────────────────
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# —— Optional packages (Prometheus, NoorFastTimeCore) with graceful fallback —
try:
    from prometheus_client import Counter as PromCounter, Gauge as PromGauge  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _Stub:  # pylint: disable=too-few-public-methods
        def labels(self, *_, **__):  # noqa: D401,E251
            return self

        def inc(self, *_):  # noqa: D401,E251
            return None

        def set(self, *_):  # noqa: D401,E251
            return None

    PromCounter = _Stub  # type: ignore
    PromGauge = _Stub  # type: ignore

try:
    from noor_fasttime_core import NoorFastTimeCore  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    NoorFastTimeCore = object  # type: ignore

# — Local imports (resolved at runtime within same package) ————————
try:
    from .quantum_ids import make_change_id, MotifChangeID  # noqa: F401
except ImportError:  # pragma: no cover – optional helper module
    def make_change_id(*_):  # type: ignore
        return "00000000"

    MotifChangeID = str  # type: ignore

# ── 3. Symbolic Configuration and Emission Defaults ──────────────────────────
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

# ── 4. Data Classes ──────────────────────────────────────────────────────────
@dataclass(slots=True)
class QuantumTickV2:  # RFC‑0003 §6.2
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
class TickEntropy:  # RFC‑0003 §3.3
    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:  # RFC‑0005 §3.3
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# ── 4.4 Helper Classes ───────────────────────────────────────────────────────
class LamportClock:
    """Monotonic tick‑id generator."""

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Evicting cache with *move‑to‑end* semantics."""

    def __init__(self, cap: int = 50_000):
        super().__init__()
        self.cap = cap

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)


class AgentSwirlModule:
    """Tracks recent motif IDs and exposes hash‑encoded swirl vectors."""

    def __init__(self, maxlen: int = 64):
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
        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """Exponential decay motif‑density map."""

    def __init__(self) -> None:
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str) -> None:
        for key in list(self._density_map):
            self._density_map[key] *= 0.99  # simple exponential decay
            if self._density_map[key] < 0.01:
                del self._density_map[key]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)


class LazyMonitorMixin:
    """Lazily resolves a global *ConsciousnessMonitor* (if any)."""

    @property
    def monitor(self):  # noqa: D401
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import (  # type: ignore
                    get_global_monitor,
                )

                self._cached_monitor = get_global_monitor()
            except Exception:  # pragma: no cover
                self._cached_monitor = None
        return self._cached_monitor


# ── 6.2 Stand‑alone Functions ───────────────────────────────────────────────

def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    """Scalar for symbolic alignment strength (RFC‑0005 §4.2)."""
    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str,
) -> None:
    """Non‑blocking callback to observability monitor (RFC‑0004 §3.2)."""
    try:
        if monitor is not None and hasattr(monitor, "report_tick"):
            monitor.report_tick(
                tick=tick,
                coherence_potential=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector,
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("Monitor callback failed: %s", exc)


# ── 5. RecursiveAgentFT ──────────────────────────────────────────────────────
class RecursiveAgentFT(LazyMonitorMixin):
    """Symbolic pulse engine implementing RFC‑CORE‑002."""

    # Prometheus metrics (lazily initialised by .labels())
    TICKS_EMITTED = PromCounter(
        "agent_ticks_emitted_total", "Ticks emitted", ["agent_id", "stage"]
    )
    AGENT_TRIADS_COMPLETED = PromCounter(
        "agent_triads_completed_total", "Triads completed", ["agent_id"]
    )
    FEEDBACK_EXPORT = PromCounter(
        "agent_feedback_export_total", "Feedback packets exported", ["agent_id"]
    )
    REWARD_MEAN = PromGauge("agent_reward_mean", "EMA of reward", ["agent_id"])
    AGENT_EMISSION_INTERVAL = PromGauge(
        "agent_emission_interval_seconds", "Current autonomous emission interval", ["agent_id"]
    )

    # ── lifecycle ------------------------------------------------------------
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
        self.tuning = {**DEFAULT_TUNING, **(tuning or {})}

        # Internal state
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema: float = 1.0
        self.entropy_slope: float = 0.1
        self._silence_streak: int = 0
        self._last_triad_hit: bool = False
        self._last_interval: float = self.tuning["base_interval"]
        self._last_tick_hash: Optional[str] = None
        self._pulse_active: bool = False
        self._pulse_task: Optional[asyncio.Task] = None
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}

        # Prometheus label binding
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
        logger.info("Initialized RecursiveAgentFT(id=%s)", self.agent_id)

    # ── 5.1.2 Method Suite ---------------------------------------------------
    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            return self._ghost_traces[key].get("motifs")
        return None

    # ——— Autonomous emission loop ———
    async def start_continuous_emission(self) -> None:
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)

    def _resolve_field(self, motif_id: str) -> str:  # helper
        base = motif_id.split(".")[0] if motif_id else "null"
        return SYMBOLIC_PHASE_MAP.get(base, "ψ-null@Ξ")

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # Pass‑through intent ( RFC‑0003 §6.2 / RFC‑0004 §2.5 )
        intent_source = getattr(self, "_intent_source", None)
        if intent_source is not None:
            tick.extensions["intent"] = intent_source

        # Field‑signature + HMAC optional
        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature

        if hasattr(self, "hmac_secret") and self.hmac_secret:  # type: ignore[attr-defined]
            signature_data = self.hmac_secret + tick_id.encode()  # type: ignore[attr-defined]
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac

        # Swirl + density update
        for motif in motifs:
            self.swirl.update_swirl(motif)
            self.density.update_density(motif)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence

        # House‑keeping + metrics
        self._last_tick_hash = hashlib.sha3_256(repr(tick).encode()).hexdigest()
        report_tick_safe(self.monitor, tick, coherence, self.density.snapshot(), swirl_hash)
        self.metrics["agent_ticks_emitted_total"].inc()
        return tick

    # ——— Emission control helpers ———
    async def start_emission(self) -> None:
        if self._pulse_active:
            return  # Idempotent
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task

    # ——— Feedback path ———
    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
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
            except Exception:  # pragma: no cover
                logger.error("Failed to retrieve from memory", exc_info=True)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    # ——— Feedback‑packet helpers ———
    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda kv: kv[1])[0] if density_map else "null"
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


# ── End‑of‑module sentinel ───────────────────────────────────────────────────
# End_of_File
