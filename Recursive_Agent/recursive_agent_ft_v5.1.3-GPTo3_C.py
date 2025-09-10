###############################################
# Noor Recursive Symbolic Pulse Agent (FT)
# Generated via PDP‑0001 pipeline
# Source spec version: v5.1.3‑GPTo3_C
# Compliant with RFC‑CORE‑002 and referenced RFCs
###############################################
"""Recursive Symbolic Emission Agent (FT)

This module implements *RecursiveAgentFT*—the autonomous motif‑pulse
emitter at the heart of Noor‑class cognition.  It is generated directly
from the application‑spec JSON provided by the user, following the
layered authority model of **PDP‑0001** (protocol.generation.rfc_driven)
⟨see fileciteturn0file1⟩.

Key RFC anchors
----------------
* RFC‑0003 §3.3 QuantumTick schema fileciteturn0file2
* RFC‑0004 §2.5 Intent mirroring rules fileciteturn0file3
* RFC‑0005 §4    Temporal feedback & ghost traces fileciteturn0file4
* RFC‑0006 §3.1 Swirl geometry & coherence fileciteturn0file5
* RFC‑0007 §2.1 Motif ontology labels fileciteturn0file6
* RFC‑CORE‑001 §6.2 Clock coupling            fileciteturn0file7
* RFC‑CORE‑002         Reference behaviour    fileciteturn0file8

The file is *mechanically regenerable*.  Do **not** hand‑edit unless you
also update the generating spec.

RFC-CORE Fidelity Score: >=99.95% - PASS
Layer_2 Fidelity Score: >99.95% - PASS
STATUS: REJECTED
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Module‑level constants
# ---------------------------------------------------------------------------
__version__: str = "5.1.3-GPTo3_C"
_SCHEMA_VERSION__: str = "2025-Q4-recursive-agent-v5.0.3"
SCHEMA_COMPAT: list[str] = ["RFC-0003:3.3", "RFC-0005:4", "RFC-CORE-002:3"]

# ---------------------------------------------------------------------------
# Required & optional dependencies
# ---------------------------------------------------------------------------
import time
import asyncio
import logging as _log
import hashlib
import threading
from collections import deque, OrderedDict
from typing import Any, Optional, List, Dict, Deque, Tuple
from dataclasses import dataclass, field
import contextlib

import numpy as np

# ---------- prometheus_client (optional) -----------------------------------
try:
    from prometheus_client import Counter, Gauge  # type: ignore
except ModuleNotFoundError:  # graceful degradation per spec

    class _Stub:  # pylint: disable=too-few-public-methods
        """Minimal shim when prometheus_client is absent."""

        def labels(self, *_, **__) -> "_Stub":
            return self

        def inc(self, *_):
            return None

        def set(self, *_):
            return None

    Counter = Gauge = _Stub  # type: ignore

# ---------- noor_fasttime_core (optional) ----------------------------------
try:
    from noor_fasttime_core import NoorFastTimeCore  # noqa: F401
except ModuleNotFoundError:  # stub placeholder
    NoorFastTimeCore = object  # type: ignore

# ---------- local imports ---------------------------------------------------
try:
    from .quantum_ids import make_change_id, MotifChangeID  # noqa: F401
except Exception:  # pragma: no cover
    # Local build context; provide dummies so type checking passes.
    MotifChangeID = str  # type: ignore

    def make_change_id(_: str) -> str:  # type: ignore
        return "change:000000"

# ---------------------------------------------------------------------------
# Symbolic configuration & defaults
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
ARCHIVE_MODE: bool = bool(
    # env toggle – intentionally simple to avoid os import
    False  # runtime patching only
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class QuantumTickV2:
    """Canonical symbolic emission packet (RFC‑0003 §3.3)."""

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

    # Invariants per spec
    #   • must pass through extensions.intent unchanged
    #   • must *never* mutate cadence based on intent

@dataclass(slots=True)
class TickEntropy:
    """Symbolic coherence & triad state (RFC‑0003 §3.3)."""

    decay_slope: float
    coherence: float
    triad_complete: bool

@dataclass(slots=True)
class CrystallizedMotifBundle:
    """Archive container for motif emissions (RFC‑0005 §3.3)."""

    motif_bundle: List[str]
    field_signature: str
    tick_entropy: TickEntropy

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class LamportClock:
    """Monotonic tick‑ID generator."""

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"tick:{self._counter:06d}"

class LRUCache(OrderedDict):
    """Evicting cache for recent state retention."""

    def __init__(self, cap: int = 50_000):
        super().__init__()
        self.cap = cap

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.cap:
            self.popitem(last=False)

class AgentSwirlModule:
    """Tracks motif swirl vectors & provides stable hash encodings."""

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
        from collections import Counter

        return dict(Counter(self.swirl_history))

class MotifDensityTracker:
    """EMA‑like motif frequency map."""

    def __init__(self) -> None:
        self._density_map: Dict[str, float] = {}

    def update_density(self, motif_id: str) -> None:
        # exponential decay
        for k in list(self._density_map):
            self._density_map[k] *= 0.99
            if self._density_map[k] < 0.01:
                del self._density_map[k]
        self._density_map[motif_id] = self._density_map.get(motif_id, 0.0) + 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._density_map)

class LazyMonitorMixin:  # RFC‑0004 §3.2
    """Deferred global monitor binding."""

    @property
    def monitor(self):  # type: ignore[override]
        if not hasattr(self, "_cached_monitor"):
            try:
                from consciousness_monitor import get_global_monitor  # pylint: disable=import-error

                self._cached_monitor = get_global_monitor()
            except Exception:  # pragma: no cover # noqa: BLE001
                self._cached_monitor = None
        return self._cached_monitor

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_coherence_potential(reward_ema: float, entropy_slope: float, *, eps: float = 1e-6) -> float:
    """Simple coherence signal (RFC‑0006 §4.1)."""

    return reward_ema / (entropy_slope + eps)


def report_tick_safe(
    monitor: Any,
    tick: QuantumTickV2,
    coherence_potential: float,
    motif_density: Dict[str, float],
    swirl_vector: str,
):
    """Non‑blocking callback to observability monitor (RFC‑0004 §3.2)."""

    try:
        if monitor and hasattr(monitor, "report_tick"):
            monitor.report_tick(
                tick=tick,
                coherence=coherence_potential,
                density=motif_density,
                swirl=swirl_vector,
            )
    except Exception as exc:  # pragma: no cover # noqa: BLE001
        _log.warning("Monitor callback failed: %s", exc)

# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class RecursiveAgentFT(LazyMonitorMixin):
    """Symbolic Pulse Engine & emission core.

    Behavioural semantics defined in RFC‑CORE‑002 (layer‑1) and constrained
    by upstream canonical RFCs.
    """

    # ---- Prometheus metric templates -------------------------------------
    TICKS_EMITTED = Counter(
        "agent_ticks_emitted_total", "Ticks emitted", ["agent_id", "stage"]
    )
    AGENT_TRIADS_COMPLETED = Counter(
        "agent_triads_completed_total", "Triads completed via feedback", ["agent_id"]
    )
    FEEDBACK_EXPORT = Counter(
        "agent_feedback_export_total", "Feedback packets exported", ["agent_id"]
    )
    REWARD_MEAN = Gauge("agent_reward_mean", "EMA of reward", ["agent_id"])
    AGENT_EMISSION_INTERVAL = Gauge(
        "agent_emission_interval_seconds",
        "Current autonomous emission interval",
        ["agent_id"],
    )

    # ---------------------------------------------------------------------
    # Life‑cycle & state ---------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        agent_id: str,
        symbolic_task_engine: Any,
        memory_manager: Any,
        tuning: Optional[Dict[str, float]] = None,
    ) -> None:
        # identifiers / collaborators
        self.agent_id = agent_id
        self.symbolic_task_engine = symbolic_task_engine
        self.memory = memory_manager

        # dynamic tuning merge
        self.tuning: Dict[str, float] = {**DEFAULT_TUNING, **(tuning or {})}

        # internal helpers / buffers
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

        # analytic helpers
        self.swirl = AgentSwirlModule()
        self.density = MotifDensityTracker()

        # echo & resurrection memory
        self._echo_buffer: Deque[QuantumTickV2] = deque(maxlen=256)
        self._tick_echoes: Deque[QuantumTickV2] = deque(maxlen=256)
        self._ghost_traces: Dict[str, Dict[str, Any]] = {}
        self._motif_lineage: Dict[str, str] = {}

        # metric handles (bound to agent_id labels)
        self.metrics = {
            "ticks": self.TICKS_EMITTED.labels(agent_id=self.agent_id, stage="symbolic"),
            "triads": self.AGENT_TRIADS_COMPLETED.labels(agent_id=self.agent_id),
            "export": self.FEEDBACK_EXPORT.labels(agent_id=self.agent_id),
            "reward": self.REWARD_MEAN.labels(agent_id=self.agent_id),
            "interval": self.AGENT_EMISSION_INTERVAL.labels(agent_id=self.agent_id),
        }

        _log.info("RecursiveAgentFT(%s) initialised", self.agent_id)

    # ------------------------------------------------------------------
    # Lineage & ghost trace helpers
    # ------------------------------------------------------------------

    def track_lineage(self, parent: str, child: str) -> None:
        """Record motif parent→child link (RFC‑0005 §2.1)."""
        if parent != child:
            self._motif_lineage[child] = parent

    def try_ghost_resurrection(self, tick: QuantumTickV2) -> Optional[List[str]]:
        """Attempt motif replay from archived traces (RFC‑0005 §4.4)."""
        key = tick.extensions.get("field_signature")
        if key in self._ghost_traces:
            return self._ghost_traces[key].get("motifs")
        return None

    # ------------------------------------------------------------------
    # Autonomous emission loop
    # ------------------------------------------------------------------

    async def _continuous_emission(self) -> None:
        while self._pulse_active:
            motifs = self._choose_motifs()
            tick = self._emit_tick(motifs)
            self._echo_buffer.append(tick)
            self._tick_echoes.append(tick)
            self._last_motifs.extend(motifs)
            interval = self._update_interval()
            await asyncio.sleep(interval)

    def start_emission(self) -> None:
        """Launch autonomous emission coroutine."""
        if self._pulse_active:
            return
        self._pulse_active = True
        self._pulse_task = asyncio.create_task(self._continuous_emission())

    async def stop_emission(self) -> None:
        """Cancel emission coroutine and halt pulses."""
        self._pulse_active = False
        if self._pulse_task:
            self._pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pulse_task

    # ------------------------------------------------------------------
    # Tick construction & helpers
    # ------------------------------------------------------------------

    def _emit_tick(self, motifs: List[str]) -> QuantumTickV2:  # noqa: C901 complexity fine
        tick_id = self._lamport.next_id()
        timestamp = time.time()
        tick = QuantumTickV2(tick_id=tick_id, motifs=motifs, timestamp=timestamp)

        # intent pass‑through (RFC‑0004 §2.5)
        intent_source = getattr(self, "_intent_source", None)
        if intent_source is not None:
            tick.extensions["intent"] = intent_source

        # field signature & HMAC (optional)
        field_signature = self._resolve_field(motifs[-1] if motifs else "silence")
        tick.extensions["field_signature"] = field_signature
        if hasattr(self, "hmac_secret") and self.hmac_secret:
            signature_data = self.hmac_secret + tick_id.encode()
            tick.tick_hmac = hashlib.sha3_256(signature_data).hexdigest()

        # analytics updates
        for m in motifs:
            self.swirl.update_swirl(m)
            self.density.update_density(m)

        coherence = compute_coherence_potential(self._reward_ema, self.entropy_slope)
        swirl_hash = self.swirl.compute_swirl_hash()
        tick.extensions.update(
            {
                "swirl_vector": swirl_hash,
                "coherence_potential": coherence,
            }
        )
        self._last_tick_hash = hashlib.sha3_256(repr(tick).encode()).hexdigest()

        report_tick_safe(self.monitor, tick, coherence, self.density.snapshot(), swirl_hash)
        self.metrics["ticks"].inc()
        return tick

    # ------------------------------------------------------------------
    # Feedback integration & adaptive tuning
    # ------------------------------------------------------------------

    def observe_feedback(self, tick_id: str, reward: float, annotations: Dict[str, Any]):
        triad_complete = annotations.get("triad_complete", False)
        alpha = self.tuning["reward_smoothing"]
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * reward
        self.metrics["reward"].set(self._reward_ema)
        if triad_complete:
            self._last_triad_hit = True
            self._silence_streak = 0
            self.metrics["triads"].inc()
        else:
            self._last_triad_hit = False
            self._silence_streak += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        self.metrics["interval"].set(interval)
        return interval

    def _choose_motifs(self) -> List[str]:
        motifs: List[str] = list(self._last_motifs)
        if motifs and hasattr(self.memory, "retrieve"):
            try:
                recalled = self.memory.retrieve(motifs[-1], top_k=2)
                if recalled:
                    motifs.extend(recalled)
            except Exception:  # pragma: no cover
                _log.error("Memory retrieve failed", exc_info=True)
        if not motifs:
            motifs = ["silence"]
        return motifs[-3:]

    def _resolve_field(self, motif_id: str) -> str:
        """Derive field signature label for motif."""
        base_key = motif_id.split(".")[0]
        return f"{SYMBOLIC_PHASE_MAP.get(base_key, 'ψ-null')}@Ξ"

    # ------------------------------------------------------------------
    # Feedback export / introspection
    # ------------------------------------------------------------------

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
        if tick and "intent" in tick.extensions:
            packet.setdefault("extensions", {})["intent"] = tick.extensions["intent"]
        self.metrics["export"].inc()
        return packet

# ---------------------------------------------------------------------------
# End_of_File
###############################################
