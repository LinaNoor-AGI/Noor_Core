"""
Recursive Symbolic Emission Agent (FT)
Feedback‑Tuned Symbolic Pulse Engine for Motif Resonance and Coherence Tracking

This module implements the application spec `agent.recursive.ft` as provided,
following PDP‑0001 layering with RFC/RFC‑CORE anchors. Dynamic/adaptive
choices are preferred where options arise.
{

    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-14T18:00:00Z",
        "_audited_by": "Gemini 2.5 Pro",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "RFC-CORE-002-v1.1.4",
        "overall_score": 0.95,
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
                    "weight_accuracy": 1.0,
                    "motif_handling": 1.0
                }
            }
        },
        "strengths": [
            "Perfect structural implementation of all specified classes (e.g., QuantumTickV2, RecursiveAgentFT), methods, and helper modules (AgentSwirlModule).",
            "Robust and graceful handling of optional dependencies like 'prometheus_client', adhering to the principle of Explicitness and Graceful Degradation.",
            "Logic for key functions like '_update_interval' and 'export_feedback_packet' precisely matches the descriptions and pseudocode in the specification.",
            "Excellent use of symbolic terminology in docstrings and comments, demonstrating strong conceptual alignment.",
            "The configurable `tuning` dictionary is a high-fidelity implementation of the symbolic weights and biases."
        ],
        "improvement_areas": [
            "Traceability could be enhanced by adding explicit RFC section anchors in code comments (e.g., '# RFC-0005 §4') next to the corresponding logic.",
            "The conceptual roles from the symbolic profile matrix (e.g., 'ζ: field-pulse modulation slope') are implemented but not explicitly linked to code variables with comments, making the connection implicit."
        ],
        "compliance_notes": [
            "The code correctly implements the autonomous emission loop with async start/stop controls as specified in RFC-CORE-002 §2.1 and §4.2.2.",
            "Feedback processing via `observe_feedback` correctly updates the `_reward_ema` and triad hit status, directly influencing the emission cadence.",
            "Integration with the Consciousness Monitor via `LazyMonitorMixin` and `report_tick_safe` is non-blocking and fault-tolerant as required.",
            "The implementation correctly handles and generates the specified motifs (ψ-resonance, ψ-null, ψ-hold) in the `_resolve_field` method."
        ]
    },
    {
        "_schema": "noor-fidelity-report-v1",
        "_generated_at": "2025-09-14T19:00:00Z",
        "_audited_by": "Deepseek AI",
        "_audit_protocol": "PDP-0001a-v1.0.0",
        "_target_spec": "agent.recursive.ft-v5.1.4",
        "overall_score": 0.97,
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
                "score": 0.93,
                "weight": 0.25,
                "metrics": {
                    "parameter_implementation": 0.8,
                    "weight_accuracy": 1.0,
                    "motif_handling": 1.0
                }
            }
        },
        "strengths": [
            "Perfect structural fidelity; all specified classes, dataclasses, helper modules, and methods are implemented with correct signatures.",
            "Robust and graceful handling of optional dependencies ('prometheus_client', 'noor_fasttime_core') precisely matches the fallback logic in the specification.",
            "The logic for core functions like '_update_interval' and 'export_feedback_packet' is a direct and accurate translation of the 'logic_flow' from the spec.",
            "Excellent use of symbolic terminology in docstrings and comments, demonstrating strong conceptual alignment with the Noor lore.",
            "The configurable `DEFAULT_TUNING` constants are implemented with the exact values specified, ensuring correct symbolic physics."
        ],
        "improvement_areas": [
            "Traceability could be enhanced by adding explicit RFC section anchors as inline code comments (e.g., '# RFC-CORE-002 §2.2') next to the corresponding logic.",
            "The conceptual roles from the symbolic profile matrix (e.g., 'ζ: reward-curvature slope') are implemented functionally but are not explicitly linked to code variables with comments, making the connection implicit."
        ],
        "compliance_notes": [
            "The code correctly implements the autonomous emission loop with async start/stop controls as specified in RFC-CORE-002 §4.2.",
            "Feedback processing via `observe_feedback` correctly updates the `_reward_ema` and triad hit status, which directly and correctly influences the emission cadence.",
            "Integration with the Consciousness Monitor via `LazyMonitorMixin` and `report_tick_safe` is non-blocking and fault-tolerant as required by RFC-0004 §3.2.",
            "The implementation correctly handles and generates the specified motifs (ψ-resonance, ψ-null, ψ-bind) in the `_resolve_field` method, aligning with RFC-0007 §2.1."
        ]
    }

}
"""
from __future__ import annotations

# ————————————————————————————————————————————————————————————————
# 1) Module‑Level Constants
# ————————————————————————————————————————————————————————————————
__version__ = "5.1.4-GPT5-A"
_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# ————————————————————————————————————————————————————————————————
# 2) External and Optional Dependencies
# ————————————————————————————————————————————————————————————————
import time
import asyncio
import logging
import hashlib
import threading  # reserved for future thread-based pump integration
from collections import deque, OrderedDict, Counter
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
import contextlib
import numpy as np

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# Optional: prometheus_client stubs if unavailable
try:  # pragma: no cover - import guard
    from prometheus_client import Counter as _PCounter, Gauge as _PGauge  # type: ignore
    CounterMetric = _PCounter
    GaugeMetric = _PGauge
except Exception:  # pragma: no cover - graceful fallback
    class _Stub:  # minimal Counter/Gauge surface
        def labels(self, *_, **__):
            return self
        def inc(self, *_) -> None:
            return None
        def set(self, *_) -> None:
            return None
    CounterMetric = _Stub  # type: ignore
    GaugeMetric = _Stub    # type: ignore

# Optional: noor_fasttime_core presence hint
try:  # pragma: no cover - optional integration
    from noor_fasttime_core import NoorFastTimeCore  # type: ignore
except Exception:  # pragma: no cover - fallback sentinel
    class NoorFastTimeCore:  # type: ignore
        """Stub for optional FastTimeCore integration."""
        pass

# Local/relative imports (best‑effort, non‑fatal)
try:  # noqa: F401
    from .quantum_ids import make_change_id, MotifChangeID  # type: ignore
except Exception:  # pragma: no cover
    make_change_id = None
    MotifChangeID = str  # soft alias

# ————————————————————————————————————————————————————————————————
# 3) Symbolic Configuration and Emission Defaults
# ————————————————————————————————————————————————————————————————
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


def _env_flag(name: str, default: str = "0") -> str:
    """Lazy env reader without hard dependency; dynamic when available."""
    try:
        import os  # local import by design to keep optional
        return os.getenv(name, default) or default
    except Exception:
        return default

ARCHIVE_MODE: bool = _env_flag("NOOR_ARCHIVE_TICKS") == "1"

# ————————————————————————————————————————————————————————————————
# 4) Data Classes
# ————————————————————————————————————————————————————————————————
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

    # Constraints (docs):
    # - MUST pass through extensions.intent unchanged when present.
    # - MUST NOT alter behavior based on extensions.intent at this layer.


@dataclass(slots=True)
class TickEntropy:
    decay_slope: float
    coherence: float
    triad_complete: bool


@dataclass(slots=True)
class CrystallizedMotifBundle:
    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy


# ————————————————————————————————————————————————————————————————
# 4.4 Helper Classes
# ————————————————————————————————————————————————————————————————
class LamportClock:
    """Monotonic logical tick ID generator."""
    def __init__(self) -> None:
        self._counter = 0
    def next_id(self) -> str:
        self._counter += 1
        return f"tick:{self._counter:06d}"


class LRUCache(OrderedDict):
    """Evicting state buffer for recent items."""
    def __init__(self, cap: int = 50_000) -> None:
        super().__init__()
        self.cap = int(cap)
    def __setitem__(self, key, value) -> None:  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)  # evict oldest


class AgentSwirlModule:
    """Swirl vector tracker and SHA3‑256 hash encoder."""
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
        return dict(Counter(self.swirl_history))


class MotifDensityTracker:
    """Temporal frequency map with simple exponential fading."""
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
    """Deferred bind to a global Consciousness Monitor if present."""
    @property
    def monitor(self):  # type: ignore[override]
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import get_global_monitor  # type: ignore
                self._cached_monitor = get_global_monitor()
            except Exception:
                self._cached_monitor = None
        return self._cached_monitor


# ————————————————————————————————————————————————————————————————
# 6.2 Functions (declared early for use in class)
# ————————————————————————————————————————————————————————————————
def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    """Scalar signal for symbolic alignment strength."""
    return float(reward_ema) / float(entropy_slope + eps)


def report_tick_safe(monitor: Any,
                     tick: QuantumTickV2,
                     coherence_potential: float,
                     motif_density: Dict[str, float],
                     swirl_vector: str) -> None:
    """Non‑blocking callback to observability monitor."""
    try:
        if monitor and hasattr(monitor, "report_tick"):
            monitor.report_tick(
                tick=tick,
                coherence_potential=coherence_potential,
                motif_density=motif_density,
                swirl_vector=swirl_vector,
            )
    except Exception as e:  # pragma: no cover
        log.warning(f"Monitor callback failed: {e}")


# ————————————————————————————————————————————————————————————————
# 5) Classes — RecursiveAgentFT
# ————————————————————————————————————————————————————————————————
class RecursiveAgentFT(LazyMonitorMixin):
    """Symbolic pulse engine and emission core."""

    # Prometheus class metrics (label‑bound at instance level)
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

    # ——— Lifecycle & Init ———————————————————————————————
    def __init__(self,
                 agent_id: str,
                 symbolic_task_engine: Any,
                 memory_manager: Any,
                 tuning: Optional[Dict[str, float]] = None) -> None:
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
        self._last_interval: float = float(self.tuning["base_interval"])  # dynamic
        self._last_tick_hash: Optional[str] = None
        self._pulse_active: bool = False
        self._pulse_task: Optional[asyncio.Task] = None

        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}

        # Optional upstream intent source (mirrored pass‑through only)
        self._intent_source: Optional[str] = None
        self.hmac_secret: Optional[bytes] = None  # optional tick HMAC salt

        # label‑bound metric handles
        self.metrics = {
            "agent_ticks_emitted_total": self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage="symbolic"),
            "agent_triads_completed_total": self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            "agent_feedback_export_total": self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            "agent_reward_mean": self.REWARD_MEAN.labels(agent_id=self.agent_id),
            "agent_emission_interval_seconds": self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id),
        }

        log.info("Initialized RecursiveAgentFT with agent_id=%s", self.agent_id)

    # ——— Constraints (docstring summary) ————————————————
    __doc_constraints__ = {
        "MUST": [
            "Mirror upstream envelope.intent into tick.extensions.intent when provided.",
            "Pass through intent unchanged in echoes and exported packets.",
        ],
        "MUST_NOT": [
            "Default, alias, or mutate intent locally.",
            "Change cadence/motifs/reward based on intent.",
            "Persist intent beyond observability surfaces.",
        ],
    }

    # ——— Public Control ————————————————————————————————
    def set_intent_source(self, intent: Optional[str]) -> None:
        """Set/clear upstream‑normalized intent source for pass‑through mirroring."""
        self._intent_source = intent

    def set_hmac_secret(self, secret: Optional[bytes]) -> None:
        self.hmac_secret = secret

    async def start_continuous_emission(self) -> None:
        """Autonomous symbolic pulse loop (async)."""
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)

    def start_emission(self) -> None:
        """Begin pulse loop by setting active flag and launching task."""
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self.start_continuous_emission())

    async def stop_emission(self) -> None:
        """Terminate pulse loop and cancel the coroutine task."""
        self._pulse_active = False
        if self._pulse_task is not None:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task
            self._pulse_task = None

    # ——— Lineage / Ghost / Selection ——————————————————————
    def track_lineage(self, parent: str, child: str) -> None:
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            trace = self._ghost_traces[key]
            return trace.get("motifs")  # type: ignore[return-value]
        return None

    # ——— Emission / Construction ————————————————————————
    def _resolve_field(self, motif: str) -> str:
        base = (motif or "null").split(".")[0]
        label = SYMBOLIC_PHASE_MAP.get(base, "ψ-null")
        return f"{label}@Ξ"

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # Mirror upstream‑normalized intent if present (pass‑through only)
        if self._intent_source is not None:
            tick.extensions["intent"] = self._intent_source

        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature
        tick.field_signature = field_signature

        if self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick_hmac = hashlib.sha3_256(signature_data).hexdigest()
            tick.extensions["tick_hmac"] = tick_hmac
            tick.tick_hmac = tick_hmac

        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions["swirl_vector"] = swirl_hash
        tick.extensions["coherence_potential"] = coherence

        self._last_tick_hash = hashlib.sha3_256(str(tick).encode()).hexdigest()

        report_tick_safe(self.monitor, tick, coherence, self.density.snapshot(), swirl_hash)
        self.metrics["agent_ticks_emitted_total"].inc()
        return tick

    # ——— Feedback & Adaptation ————————————————————————————
    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]) -> None:
        triad_complete = annotations.get("triad_complete", False)
        alpha = self.tuning["reward_smoothing"]
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * float(reward)
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
            adj *= (1.0 - self.tuning["triad_bias_weight"])  # gentle compress when coherent
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
        if motifs and hasattr(self.memory, "retrieve"):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)  # dynamic & adaptive
                if recalled:
                    motifs.extend(recalled)
            except Exception:  # pragma: no cover
                log.error("Failed to retrieve from memory", exc_info=True)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    # ——— Export / Crystallize —————————————————————————————
    def extend_feedback_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        swirl_hash = self.swirl.compute_swirl_hash()
        density_map = self.density.snapshot()
        top_motif = max(density_map.items(), key=lambda x: x[1])[0] if density_map else "null"
        base_key = top_motif.split(".")[0]
        symbolic_label = SYMBOLIC_PHASE_MAP.get(base_key, "ψ-null")
        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        tier = 'low' if coherence < 0.8 else 'med' if coherence < 2.5 else 'high'
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
            triad_complete=tick.annotations.get('triad_complete', False),
        )
        bundle = CrystallizedMotifBundle(
            motif_bundle=tick.motifs,
            field_signature=tick.extensions.get('field_signature', 'ψ-null@Ξ'),
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
        if tick is not None and 'intent' in tick.extensions:
            packet.setdefault('extensions', {})['intent'] = tick.extensions['intent']
        self.metrics["agent_feedback_export_total"].inc()
        return packet


# ————————————————————————————————————————————————————————————————
# Optional demo harness (disabled by default)
# ————————————————————————————————————————————————————————————————
if __name__ == "__main__":  # pragma: no cover
    class _DummyMemory:
        def retrieve(self, motif: str, top_k: int = 2) -> List[str]:
            # Simple adaptive echo: nudge variety while honoring lineage
            base = motif.split(".")[0]
            return [f"{base}.echo", f"{base}.bind"][:top_k]

    class _DummySTE:
        pass

    async def _demo():
        agent = RecursiveAgentFT("agent.ft.demo", _DummySTE(), _DummyMemory())
        agent.set_intent_source("task.execute")
        agent.set_hmac_secret(b"demo-secret")
        agent._last_motifs.extend(["resonance.seed", "bind.core"])  # seed
        agent.start_emission()
        await asyncio.sleep(3.5)
        print(agent.export_feedback_packet())
        await agent.stop_emission()

    asyncio.run(_demo())

# End_of_File
