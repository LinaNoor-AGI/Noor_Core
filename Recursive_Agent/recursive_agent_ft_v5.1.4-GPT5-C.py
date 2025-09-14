"""
Recursive Symbolic Emission Agent (FT)
Feedback-Tuned Symbolic Pulse Engine for Motif Resonance and Coherence Tracking

Layering provenance (PDP-0001):
- Layer 0 (RFC): symbolic canon and constraints
- Layer 1 (RFC-CORE): architecture/behaviors (dominant for runtime specifics)
- Layer 2 (App-Spec): this implementation blueprint rendered as Python
The artifact mirrors upstream intent signals without mutation, and binds
to field geometry/swirl per RFC-0006 and feedback per RFC-0005/CORE-002.

This module is designed to be:
- Dynamic: optional dependencies are lazy and stubbed; cadence adapts to feedback/entropy.
- Adaptive: motif choice leans on recent memory, swirl history, and density decay.
- Observability-ready: Prometheus metrics (or no-op stubs) with safe monitor callbacks.
{
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-14T19:06:00Z",
        "_audited_by": "Gemini 2.5 Pro",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "RFC-CORE-002-v1.1.4",
        "overall_score": 0.92,
        "score_breakdown": {
            "structural_compliance": {
                "score": 0.98,
                "weight": 0.40,
                "metrics": {
                    "class_definitions": 1.0,
                    "method_signatures": 0.9,
                    "constants_and_attributes": 1.0,
                    "dependency_handling": 1.0
                }
            },
            "semantic_fidelity": {
                "score": 0.93,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 0.9,
                    "rfc_anchor_traceability": 0.8,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 1.0
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.80,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.7,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.8
                }
            }
        },
        "strengths": [
            "Near-perfect structural implementation of all specified classes, dataclasses, and helper modules.",
            "Excellent implementation of graceful degradation for optional dependencies like Prometheus and the Consciousness Monitor.",
            "The core emission logic, particularly the `_update_interval` function, is a mathematically precise implementation of the RFC's pseudocode.",
            "Accurate symbolic phase classification and feedback packet generation, correctly using swirl, density, and coherence potential.",
            "Clever and accurate implementation of the symbolic matrix weights through the `DEFAULT_TUNING` constants, especially the direct mapping for the `Δ` parameter."
        ],
        "improvement_areas": [
            "The `ghost_decay` function specified in RFC-CORE-002 §5.2 is missing, leaving the ghost trace registry to grow indefinitely.",
            "While RFC anchors are present in module and class docstrings, they are largely absent as inline comments, reducing traceability from specific functions to the section they implement.",
            "The conceptual roles of the symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) are not explicitly labeled in the code with comments, making the link between the matrix and the logic less obvious.",
            "While motif handling is present for classification, the code could more strongly reflect the specific behavioral roles of the motifs listed in the matrix (`ψ-resonance`, `ψ-null`, `ψ-hold`)."
        ],
        "compliance_notes": [
            "The implementation correctly handles and resolves the symbolic motifs required by the specification.",
            "The emission interval adaptation formula follows the exact logic from RFC-CORE-002 §2.2, correctly implementing feedback-driven cadence.",
            "Feedback processing and reward smoothing (EMA) are implemented correctly as per the RFC's description.",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2, ensuring the core emission loop is never compromised by observability hooks."
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-14T19:08:00Z",
        "_audited_by": "Gemini 2.5 pro",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "recursive_agent_ft.JSON-v5.1.4",
        "overall_score": 0.96,
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
                "score": 0.94,
                "weight": 0.35,
                "metrics": {
                    "logic_flow_adherence": 1.0,
                    "rfc_anchor_traceability": 0.8,
                    "conceptual_alignment": 1.0,
                    "documentation_clarity": 1.0
                }
            },
            "symbolic_matrix_alignment": {
                "score": 0.89,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 0.9,
                    "motif_handling": 0.95
                }
            }
        },
        "strengths": [
            "Perfect structural implementation of all specified dataclasses, helper classes, and the main agent class.",
            "Excellent and robust handling of optional dependencies (`prometheus-client`, `noor_fasttime_core`) with graceful fallback stubs as specified.",
            "The core emission logic, especially the `_update_interval` function, is a mathematically precise implementation of the specification's pseudocode.",
            "Strong conceptual alignment, with clear docstrings that correctly reference and adhere to complex constraints like the read-only, pass-through nature of `intent`.",
            "Accurate symbolic phase classification and feedback packet generation, correctly using swirl, density, and coherence potential."
        ],
        "improvement_areas": [
            "RFC anchor traceability could be improved. While high-level docstrings reference RFCs, specific method implementations lack inline comments linking them to the exact RFC sections they fulfill (e.g., '# RFC-CORE-002 §2.2').",
            "The conceptual roles of the symbolic matrix parameters (ψA, ζ, E, Δ, ℋ) are implemented functionally but are not explicitly labeled with comments in the code, making the audit trail less direct.",
            "The implementation of the symbolic matrix `weights` is indirect, translated into `DEFAULT_TUNING` constants rather than being used as direct multipliers, requiring interpretation to verify."
        ],
        "compliance_notes": [
            "The implementation correctly mirrors the `intent` field in tick extensions without mutation, fully complying with RFC-0003 §6.2 and RFC-0004 §2.5.",
            "The emission interval adaptation formula precisely follows the logic from the specification, correctly implementing feedback-driven cadence.",
            "Feedback processing and reward smoothing (EMA) are implemented correctly.",
            "Monitor integration uses safe, non-blocking patterns as required by RFC-0004 §3.2, ensuring the core emission loop is never compromised by observability hooks."
        ]
    }
}
"""

from __future__ import annotations

# ——— Required / standard library
import time
import asyncio
import logging
import hashlib
import threading
from collections import deque, OrderedDict, Counter
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
import contextlib

# ——— Third-party (required by spec)
import numpy as np  # RFC-CORE cadence control uses numeric clipping

# ——— Optional dependencies (graceful fallbacks)
with contextlib.suppress(Exception):
    # Local import is optional; we guard to allow standalone operation.
    # The noqa keeps linters happy even if these symbols are unused here.
    from .quantum_ids import make_change_id, MotifChangeID  # noqa: F401

try:
    from prometheus_client import Counter as _PCounter, Gauge as _PGauge  # type: ignore
    CounterMetric = _PCounter
    GaugeMetric = _PGauge
except Exception:
    class _Stub:  # minimal no-op metrics stub per spec
        def __init__(self, *_: Any, **__: Any) -> None: ...
        def labels(self, *_: Any, **__: Any) -> "_Stub": return self
        def inc(self, *_: Any, **__: Any) -> None: ...
        def set(self, *_: Any, **__: Any) -> None: ...
    CounterMetric = _Stub  # type: ignore
    GaugeMetric = _Stub    # type: ignore

try:
    # Optional runtime integration point. We avoid hard dependency.
    from noor_fasttime_core import NoorFastTimeCore  # type: ignore
except Exception:
    class NoorFastTimeCore(object):  # pragma: no cover
        """Placeholder type; real integration occurs if package present."""
        pass

log = logging.getLogger("recursive_agent_ft")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# ————————————————————————————————————————————————————————————————————
# 1) Module-Level Constants (Spec §1)
# ————————————————————————————————————————————————————————————————————
__version__ = "5.1.4-GPT5-C"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# ————————————————————————————————————————————————————————————————————
# 3) Symbolic Configuration and Emission Defaults (Spec §3)
# ————————————————————————————————————————————————————————————————————
DEFAULT_TUNING: Dict[str, float] = {
    # Expanded tuning parameters including base_interval (RFC-CORE-002 cadence)
    "min_interval": 0.25,
    "max_interval": 10.0,
    "base_interval": 1.5,
    "entropy_boost_threshold": 0.35,
    "triad_bias_weight": 0.15,
    "reward_smoothing": 0.2,
}

SYMBOLIC_PHASE_MAP: Dict[str, str] = {
    "bind": "ψ-bind",
    "spar": "ψ-spar",
    "null": "ψ-null",
    "resonance": "ψ-resonance",
    "hold": "ψ-hold",
    "dream": "ψ-dream",
    "myth": "ψ-myth",
}

PHASE_SHIFT_MODE: List[str] = ["delay", "remix", "lineage_break"]


def _env_flag(name: str, default: bool = False) -> bool:
    """Lazy env check to avoid a hard top-level dependency on os."""
    with contextlib.suppress(Exception):
        import os  # local import keeps 'os' off the global import list
        return os.environ.get(name, "0") == "1"
    return default


ARCHIVE_MODE: bool = _env_flag("NOOR_ARCHIVE_TICKS", False)

# ————————————————————————————————————————————————————————————————————
# 4) Data Classes (Spec §4)
# ————————————————————————————————————————————————————————————————————
@dataclass(slots=True)
class QuantumTickV2:
    """
    Canonical symbolic emission format.
    ⚠ Intent handling: per RFC-0004 §2.5 and RFC-0003 §6.2, this agent must
    mirror transport-normalized `intent` into tick.extensions['intent']
    without mutation, defaulting, or inference at this layer.
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
    """Symbolic coherence & triad state snapshot for a tick."""
    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    """
    Archival crystallization of a tick’s motif state (RFC-0005 §2.1).
    """
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# ————————————————————————————————————————————————————————————————————
# 4.4) Helper Classes
# ————————————————————————————————————————————————————————————————————
class LamportClock:
    """Monotonic logical tick ID generator."""
    def __init__(self) -> None:
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Evicting cache (not used directly in core loop, provided per spec)."""
    def __init__(self, cap: int = 50_000) -> None:
        super().__init__()
        self.cap = cap

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)  # evict oldest


class AgentSwirlModule:
    """
    Swirl vector tracker & hash encoder.
    Encodes motif swirl history as a symbolic hash (RFC-0006 §2.2/4.2).
    """
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
        # O(n) using Counter over history window
        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """Temporal frequency map with mild exponential decay."""
    def __init__(self) -> None:
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str) -> None:
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)


class LazyMonitorMixin:
    """
    Deferred bind to Consciousness Monitor (RFC-0004 §3.2).
    The monitor is optional; callbacks must be safe.
    """
    @property
    def monitor(self) -> Any:
        if not hasattr(self, "_cached_monitor"):
            with contextlib.suppress(Exception):
                from consciousness_monitor import get_global_monitor  # type: ignore
                self._cached_monitor = get_global_monitor()
            if not hasattr(self, "_cached_monitor"):
                self._cached_monitor = None
        return self._cached_monitor


# ————————————————————————————————————————————————————————————————————
# 6.2) Functions — coherence & observability helpers
# ————————————————————————————————————————————————————————————————————
def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    """
    Scalar signal for symbolic alignment strength.
    Heuristic: C ∝ reward_ema / entropy_slope (RFC-0006 §4.1).
    """
    return float(reward_ema / (entropy_slope + eps))


def report_tick_safe(monitor: Any,
                     tick: QuantumTickV2,
                     coherence_potential: float,
                     motif_density: Dict[str, float],
                     swirl_vector: str) -> None:
    """Non-blocking monitor callback (RFC-0004 §3.2)."""
    try:
        if monitor and hasattr(monitor, "report_tick"):
            monitor.report_tick(  # type: ignore
                tick=tick,
                coherence_potential=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector,
            )
    except Exception as e:  # pragma: no cover
        log.warning(f"Monitor callback failed: {e}")


# ————————————————————————————————————————————————————————————————————
# 5) Class: RecursiveAgentFT — Emission Core
# ————————————————————————————————————————————————————————————————————
class RecursiveAgentFT(LazyMonitorMixin):
    """
    Autonomous motif emitter with feedback-aligned cadence.
    Intent handling contract:
      • Mirror upstream-normalized envelope.intent into tick.extensions.intent.
      • Do not mutate/default/alias intent here (RFC-0004 §2.5; RFC-0003 §6.2).
      • No behavioral effects may be conditioned on intent at this layer.
    """

    # Prometheus metric declarations (no-ops if prom client unavailable)
    TICKS_EMITTED = CounterMetric(
        "agent_ticks_emitted_total", "Ticks emitted", ["agent_id", "stage"]
    )
    AGENT_TRIADS_COMPLETED = CounterMetric(
        "agent_triads_completed_total", "Triads completed via feedback", ["agent_id"]
    )
    FEEDBACK_EXPORT = CounterMetric(
        "agent_feedback_export_total", "Feedback packets exported", ["agent_id"]
    )
    REWARD_MEAN = GaugeMetric(
        "agent_reward_mean", "EMA of reward", ["agent_id"]
    )
    AGENT_EMISSION_INTERVAL = GaugeMetric(
        "agent_emission_interval_seconds", "Current autonomous emission interval", ["agent_id"]
    )

    # lightweight echo buffers (read-only intent pass-through)
    # (deque is sufficient; LRUCache provided for extended use-cases)
    def __init__(self,
                 agent_id: str,
                 symbolic_task_engine: Any,
                 memory_manager: Any,
                 tuning: Optional[Dict[str, float]] = None) -> None:
        # Identity & collaborators
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager

        # Tuning / dynamics
        self.tuning: Dict[str, float] = {**DEFAULT_TUNING, **(tuning or {})}
        self._lamport = LamportClock()
        self._last_motifs: Deque[str] = deque(maxlen=3)
        self._reward_ema: float = 1.0
        self.entropy_slope: float = 0.1
        self._silence_streak: int = 0
        self._last_triad_hit: bool = False
        self._last_interval: float = self.tuning["base_interval"]
        self._last_tick_hash: Optional[str] = None

        # Runtime/loop
        self._pulse_active: bool = False
        self._pulse_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()

        # Symbolic trackers
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}

        # Metrics with label binding
        self.metrics = {
            "agent_ticks_emitted_total": self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage="symbolic"),
            "agent_triads_completed_total": self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            "agent_feedback_export_total": self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            "agent_reward_mean": self.REWARD_MEAN.labels(agent_id=self.agent_id),
            "agent_emission_interval_seconds": self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id),
        }

        log.info("Initialized RecursiveAgentFT(agent_id=%s)", self.agent_id)

    # ——— Constraints helpers ——————————————————————————————
    @staticmethod
    def _resolve_field(motif_id: str) -> str:
        """
        Resolve a symbolic field signature from a motif base-key.
        Maps {bind,spar,null,resonance,hold,dream,myth} → ψ-<key>@Ξ.
        """
        base = (motif_id or "null").split(".")[0]
        label = SYMBOLIC_PHASE_MAP.get(base, "ψ-null")
        return f"{label}@Ξ"

    # ——— Lineage & ghosts ————————————————————————————————
    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")  # upstream-resolved key
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get("motifs")
        return None

    # ——— Emission loop ————————————————————————————————
    async def start_continuous_emission(self) -> None:
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)

            with self._lock:
                self._echo_buffer.append(tick)
                self._tick_echoes.append(tick)
                self._last_motifs.extend(motifs)

            interval = self._update_interval()
            await asyncio.sleep(interval)

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        ts = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=ts)

        # Intent pass-through (mirror only; never mutate) — RFC-0004 §2.5, RFC-0003 §6.2
        intent_source = getattr(self, "_intent_source", None)
        if intent_source is not None:
            tick.extensions["intent"] = intent_source  # mirror verbatim

        # Field signature and HMAC (optional)
        field_sig = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_sig

        if hasattr(self, "hmac_secret") and getattr(self, "hmac_secret"):
            # hmac_secret is expected to be bytes; caller responsibility.
            signature_data = self.hmac_secret + tick_id.encode()  # type: ignore[attr-defined]
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac

        # Update swirl/density and annotate extensions
        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence

        # Internal trace hash
        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()

        # Observability (safe)
        report_tick_safe(self.monitor, tick, coherence, self.density.snapshot(), swirl_hash)

        # Metrics
        self.metrics["agent_ticks_emitted_total"].inc()
        return tick

    def start_emission(self) -> None:
        """Begin pulse loop (async task)."""
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        """Terminate pulse loop and cancel task."""
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(Exception):
                await self._pulse_task
            self._pulse_task = None

    # ——— Feedback integration & adaptation —————————————
    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]) -> None:
        triad_complete = annotations.get("triad_complete", False)
        alpha = self.tuning["reward_smoothing"]
        self._reward_ema = (1.0 - alpha) * self._reward_ema + alpha * reward
        self.metrics["agent_reward_mean"].set(self._reward_ema)

        if triad_complete:
            self._last_triad_hit = True
            self._silence_streak = 0
            self.metrics["agent_triads_completed_total"].inc()
        else:
            self._last_triad_hit = False
            self._silence_streak += 1

    def _update_interval(self) -> float:
        # Start from inverse relation: higher reward pushes shorter intervals.
        adj = 1.0 - (self._reward_ema - 1.0)
        if self.entropy_slope < self.tuning["entropy_boost_threshold"]:
            adj *= 0.5
        if self._last_triad_hit:
            adj *= (1.0 - self.tuning["triad_bias_weight"])

        interval = float(np.clip(self.tuning["base_interval"] * adj,
                                 self.tuning["min_interval"],
                                 self.tuning["max_interval"]))
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
            except Exception:
                log.error("Failed to retrieve from memory", exc_info=True)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    # ——— Export / crystallize ———————————————————————————
    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else "null"
        base_key = top_motif.split(".")[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, "ψ-null")
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = "low" if coherence < 0.8 else "med" if coherence < 2.5 else "high"
        phase_id = f"{symbolic_label}-[{tier}]-{swirl_hash[:6]}"

        packet.setdefault("extensions", {}).update({
            "entanglement_status": {
                "phase": phase_id,
                "swirl_vector": swirl_hash,
                "ρ_top": sorted(density_map.items(), key=lambda kv: -kv[1])[:5],
            }
        })
        return packet

    def _crystallize_tick(self, tick: QuantumTickV2) -> CrystallizedMotifBundle:
        entropy = TickEntropy(
            decay_slope=self.entropy_slope,
            coherence=self._reward_ema,
            triad_complete=tick.annotations.get("triad_complete", False),
        )
        bundle = CrystallizedMotifBundle(
            motif_bundle=tick.motifs,
            field_signature=tick.extensions.get("field_signature", "ψ-null@Ξ"),
            tick_entropy=entropy,
        )
        return bundle

    def export_feedback_packet(self) -> Dict[str, Any]:
        with self._lock:
            last_tick = self._echo_buffer[-1] if self._echo_buffer else None

        packet: Dict[str, Any] = {
            "tick_buffer_size": len(self._echo_buffer),
            "ghost_trace_count": len(self._ghost_traces),
            "recent_reward_ema": self._reward_ema,
            "cadence_interval": self._last_interval,
            "silence_streak": self._silence_streak,
        }

        self.extend_feedback_packet(packet)

        # Observability pass-through of intent (if present). Never mutate/default here.
        if last_tick is not None and "intent" in last_tick.extensions:
            packet.setdefault("extensions", {})
            packet["extensions"]["intent"] = last_tick.extensions["intent"]

        self.metrics["agent_feedback_export_total"].inc()
        return packet


# End_of_File
