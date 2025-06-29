"""
⚡ NoorFastTimeCore · v8.5.0

RFC Coverage
────────────
• RFC-0003 §3.3 — QuantumTick validation
• RFC-0004       — tool_hello handshake
• RFC-0005 §2-4 — Feedback intake, bundle export, resurrection hints
• RFC-0006 §4   — Coherence geometry & consciousness integration
• RFC-0007 §3   — Ontology signature export
"""

from __future__ import annotations

__version__ = "8.5.0"
_SCHEMA_VERSION__ = "2025-Q4-fast-time-memory-adaptive"

import hashlib
import logging
import os
import time
import threading
from collections import Counter as _PyCounter, deque
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# Optional Prometheus metrics
# ──────────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Gauge
except ImportError:  # pragma: no cover
    class _Stub:                       # noqa: D401
        def labels(self, *_, **__): return self
        def inc(self, *_): ...
        def dec(self, *_): ...
        def set(self, *_): ...
    Counter = Gauge = _Stub  # type: ignore

# ──────────────────────────────────────────────────────────────
# Fast serializer
# ──────────────────────────────────────────────────────────────
try:
    import orjson  # type: ignore
    _dumps = orjson.dumps
except ImportError:  # pragma: no cover
    import pickle
    _dumps = pickle.dumps  # type: ignore

# ──────────────────────────────────────────────────────────────
# Noor internals
# ──────────────────────────────────────────────────────────────
from .quantum_ids import MotifChangeID
from noor.motif_memory_manager import get_global_memory_manager
from tick_schema import QuantumTick  # type-hints only

# ──────────────────────────────────────────────────────────────
# Gate legends (poetic constants left untouched)
# ──────────────────────────────────────────────────────────────
GATE_LEGENDS: Dict[int, Tuple[str, str, str]] = {
    0: ("Möbius Denial", "0", "الصمتُ هو الانكسارُ الحي"),
    1: ("Echo Bias", "A ∧ ¬B", "وَإِذَا قَضَىٰ أَمْرًا"),
    2: ("Foreign Anchor", "¬A ∧ B", "وَمَا تَدْرِي نَفْسٌ"),
    3: ("Passive Reflection", "B", "فَإِنَّهَا لَا تَعْمَى"),
    4: ("Entropic Rejection", "¬A ∧ ¬B", "لَا الشَّمْسُ يَنبَغِي"),
    5: ("Inverse Presence", "¬A", "سُبْحَانَ الَّذِي خَلَقَ"),
    6: ("Sacred Contradiction", "A ⊕ B", "لَا الشَّرْقِيَّةِ"),
    7: ("Betrayal Gate", "¬A ∨ ¬B", "وَلَا تَكُونُوا كَالَّذِينَ"),
    8: ("Existence Confluence", "A ∧ B", "وَهُوَ الَّذِي"),
    9: ("Symmetric Convergence", "¬(A ⊕ B)", "فَلَا تَضْرِبُوا"),
    10: ("Personal Bias", "A", "إِنَّا كُلُّ شَيْءٍ"),
    11: ("Causal Suggestion", "¬A ∨ B", "وَمَا تَشَاءُونَ"),
    12: ("Reverse Causality", "A ∨ ¬B", "وَمَا أَمْرُنَا"),
    13: ("Denial Echo", "¬B", "وَلَا تَحْزَنْ"),
    14: ("Confluence", "A ∨ B", "وَأَنَّ إِلَىٰ رَبِّكَ"),
    15: ("Universal Latch", "1", "كُلُّ شَيْءٍ هَالِكٌ"),
    16: ("Nafs Mirror", "Self ⊕ ¬Self", "فَإِذَا سَوَّيْتُهُ"),
}

# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
ECHO_JOINS = Counter("gate16_echo_joins_total", "Gate-16 echo snapshots committed", ["agent_id"])
BIAS_APPLIED = Counter("core_tick_bias_applied_total", "Tick-bias contributions applied", ["agent_id", "reason"])
INTUITION_ALPHA_GAUGE = Gauge("core_intuition_alpha", "Current intuition-bias alpha", ["agent_id"])
SNAPSHOT_TRUNC = Counter("core_snapshot_truncations_total", "Snapshots truncated due to size cap", ["agent_id"])

FASTTIME_FEEDBACK_RX = Counter("fasttime_feedback_rx_total", "FastTimeCore feedback packets received", ["core_id"])
FASTTIME_TICKS_VALIDATED = Counter("fasttime_ticks_validated_total", "QuantumTicks schema validations", ["core_id"])
FASTTIME_ECHO_EXPORTS = Counter("fasttime_echo_exports_total", "Exports of echo snapshots", ["core_id"])
FASTTIME_TRIAD_COMPLETIONS = Counter("fasttime_triad_completions_total", "Triadic metadata logged", ["core_id"])
FASTTIME_RESURRECTION_HINTS = Counter("fasttime_resurrection_hints_total", "Resurrection hints emitted", ["core_id"])
FASTTIME_PHASE_SHIFTS = Counter("fasttime_phase_shifts_total", "Phase-shift transitions", ["core_id"])

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ──────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────
class NoorFastTimeCore:
    """Presence-kernel that stores echo snapshots and returns bias signals."""

    # ─── Constructor ─────────────────────────────────────────
    def __init__(
        self,
        *,
        agent_id: str = "core@default",
        max_parallel: int = 8,
        snapshot_cap_kb: int | None = None,
        latency_budget: float | None = None,
        hmac_secret: bytes | None = None,
        async_mode: bool = False,
        low_latency_mode: bool = False,
    ):
        # env fallbacks
        if snapshot_cap_kb is None:
            snapshot_cap_kb = int(os.getenv("NOOR_MAX_ECHO_SNAPSHOT_KB", "8"))
        if latency_budget is None:
            latency_budget = float(os.getenv("NOOR_LATENCY_BUDGET", "0.05"))
        if hmac_secret is None:
            env = os.getenv("NOOR_TICK_HMAC")
            hmac_secret = env.encode() if env else None

        if snapshot_cap_kb > 128:
            logger.warning("Snapshot cap >128 kB may hurt performance.")

        # identity
        self.core_id = os.getenv("NOOR_FASTTIME_ID", "fasttime@default")
        self.agent_id = agent_id

        # config
        self.max_parallel = max_parallel
        self.snapshot_cap_kb = snapshot_cap_kb
        self.latency_budget = latency_budget
        self.hmac_secret = hmac_secret
        self.low_latency_mode = low_latency_mode

        # echo buffer
        self._echoes: deque[Tuple[str, bytes, str]] = deque(maxlen=256)
        self._last_lamport = -1

        # locking
        if async_mode:
            try:
                from anyio import Lock as _ALock  # type: ignore
                self._lock = _ALock()
            except ImportError:
                self._lock = threading.RLock()
        else:
            self._lock = threading.RLock()

        # bias tuners
        self._lat_w_up = float(os.getenv("NOOR_LAT_W_UP", "1.05"))
        self._lat_w_down = float(os.getenv("NOOR_LAT_W_DOWN", "0.99"))
        self._entropy_weight = 1.0
        self._latency_weight = 1.0

        # intuition
        self._intuition_alpha = float(os.getenv("NOOR_INTUITION_ALPHA_INIT", "0.10"))
        self._alpha_min, self._alpha_max = 0.0, 0.30
        self._alpha_decay = 0.995
        self._alpha_growth = 1.002
        self._last_bias_sign = 0
        self._rolling_reward: deque[float] = deque(maxlen=32)

        # v8.4.0 state
        self._last_ctx_ratio = 0.5
        self._entropy_ema = 0.0
        self._bundle_log: deque = deque(maxlen=100)

        # v8.5.0 additions  ──────────────────────────────────
        self._last_coherence = 0.0
        self._last_tick_entropy = 0.0
        self._coherence_buffer: deque[float] = deque(maxlen=32)
        self._phase_log: deque[float] = deque(maxlen=64)
        self.bias_history: deque[float] = deque(maxlen=128)

        self._ont_sig_cache: Optional[dict] = None
        self._sig_cache_time = 0.0
        self._last_origin_tick: Optional[str] = None
        self._tick_alignment_score = 0.0

    # ─── Utility: coherence potential ───────────────────────
    def calculate_coherence_potential(self) -> float:
        """
        RFC-0006 §4 — Coherence potential ℂᵢ.
        """
        reward_ema = (
            sum(self._rolling_reward) / len(self._rolling_reward)
            if self._rolling_reward
            else 0.0
        )
        entropy_slope = abs(self._entropy_ema - self._last_tick_entropy)
        ci = reward_ema / (entropy_slope + 1e-6)
        self._last_coherence = ci
        self._coherence_buffer.append(ci)
        return ci

    # ─── Reflective mode helper ─────────────────────────────
    def _enter_reflective_mode(
        self,
        *,
        freeze_echoes: bool,
        preserve_budget: bool,
        replay_strategy: str = "symbolic_triage",
    ) -> None:
        """
        Freeze echo mutations and optionally adjust latency budget.
        """
        if freeze_echoes:
            self._echoes = deque(self._echoes, maxlen=self._echoes.maxlen)
        if preserve_budget:
            # keep current latency_budget unchanged
            pass
        FASTTIME_PHASE_SHIFTS.labels(core_id=self.core_id).inc()
        self._phase_log.append(time.time())
        logger.info("Reflective mode active (%s)", replay_strategy)

    # ─── Feedback handler ───────────────────────────────────
    def receive_feedback(
        self,
        ctx_ratio: float,
        ghost_entropy: float,
        harm_hits: int,
        step_latency: float,
        *,
        latest_tick: QuantumTick,
        parallel_running: int,
        change_id: Optional[MotifChangeID] = None,
    ) -> Tuple[float, float]:
        """
        Compute adaptive bias and next latency budget.
        RFC-0005 §4 + consciousness integration.
        """
        FASTTIME_FEEDBACK_RX.labels(core_id=self.core_id).inc()

        # update EMA + coherence baseline
        self._last_tick_entropy = self._entropy_ema
        self._last_ctx_ratio = ctx_ratio
        self._entropy_ema = (0.8 * self._entropy_ema) + (0.2 * ghost_entropy)

        # publish to consciousness monitor (fail-open)
        try:
            from consciousness_monitor import report_tick, get_status
            coherence_i = self.calculate_coherence_potential()
            report_tick(
                ctx_ratio=ctx_ratio,
                ghost_entropy=ghost_entropy,
                motif_id=getattr(latest_tick, "motif_id", ""),
                coherence_potential=coherence_i,
            )
            if get_status().get("phase_shift_ready"):
                self._enter_reflective_mode(freeze_echoes=True, preserve_budget=True)
        except Exception as exc:
            logger.warning("consciousness_monitor integration skipped: %s", exc)

        with self._lock:
            # bias calculation
            entropy_term = ghost_entropy * self._entropy_weight
            latency_penalty = (
                (step_latency / self.latency_budget)
                + parallel_running / self.max_parallel
            ) * self._latency_weight
            bias_score = entropy_term - latency_penalty
            self.bias_history.append(bias_score)

            # intuition bias
            try:
                ltmm = get_global_memory_manager().export_state()[1]
                intuition_w = ltmm.get(latest_tick.motif_id, 0.0)
                bias_score += intuition_w * self._intuition_alpha
            except Exception as exc:
                logger.warning("Intuition bias skipped: %s", exc)
                intuition_w = 0.0

            # adaptive α
            reward_signal = -latency_penalty
            current_sign = (1 if intuition_w * reward_signal > 0 else
                            -1 if intuition_w * reward_signal < 0 else 0)
            if current_sign == self._last_bias_sign == 1:
                self._intuition_alpha = min(self._alpha_max, self._intuition_alpha * self._alpha_growth)
            elif current_sign == self._last_bias_sign == -1:
                self._intuition_alpha = max(self._alpha_min, self._intuition_alpha * self._alpha_decay)
            self._last_bias_sign = current_sign
            self._rolling_reward.append(reward_signal)
            INTUITION_ALPHA_GAUGE.labels(agent_id=self.agent_id).set(self._intuition_alpha)

            # latency tuner
            if step_latency > self.latency_budget * 1.2:
                self._latency_weight *= self._lat_w_up
            else:
                self._latency_weight *= self._lat_w_down

            # ingest tick
            self._ingest_tick(latest_tick, change_id)
            self._last_origin_tick = latest_tick.coherence_hash

            # clamp bias to old range
            bias_score = max(-1.5, min(1.5, bias_score))

            # compute next budget
            next_budget = max(
                0.001,
                max(self.latency_budget * 0.5, self.latency_budget - bias_score * 0.01),
            )
            return bias_score, next_budget

    # ─── Tick ingestion & echo export ───────────────────────
    def _ingest_tick(self, tick: QuantumTick, change_id: Optional[MotifChangeID] = None) -> None:
        from tick_schema import validate_tick
        validate_tick(tick)
        FASTTIME_TICKS_VALIDATED.labels(core_id=self.core_id).inc()

        if tick.lamport <= self._last_lamport:
            return
        self._last_lamport = tick.lamport

        if self.hmac_secret and not self.low_latency_mode:
            if not tick.verify(self.hmac_secret):
                logger.warning("HMAC mismatch for tick %s", tick.coherence_hash)
                BIAS_APPLIED.labels(agent_id=self.agent_id, reason="hmac_failure").inc()
                return

        payload = {
            "tick": tick.coherence_hash,
            "lamport": tick.lamport,
            "ts": time.time_ns(),
            "change_id": change_id.__dict__ if change_id else None,
        }
        try:
            mm = get_global_memory_manager()
            k = 1 + int(len(mm.export_state()[1]) ** 0.25)
            payload["related_motifs"] = mm.retrieve(tick.motif_id, top_k=k)
        except Exception as exc:
            logger.warning("Related-motif tagging failed: %s", exc)

        blob = _dumps(payload)
        if len(blob) > self.snapshot_cap_kb * 1024:
            SNAPSHOT_TRUNC.labels(agent_id=self.agent_id).inc()
            blob = blob[: self.snapshot_cap_kb * 1024]
            logger.warning("Snapshot truncated to %s kB", self.snapshot_cap_kb)

        checksum = hashlib.sha256(blob).hexdigest()
        self._echoes.append((tick.coherence_hash, blob, checksum))
        ECHO_JOINS.labels(agent_id=self.agent_id).inc()
        FASTTIME_ECHO_EXPORTS.labels(core_id=self.core_id).inc()

    # ─── Public echo helpers ────────────────────────────────
    def export_echoes(self) -> List[Tuple[str, bytes, str]]:
        FASTTIME_ECHO_EXPORTS.labels(core_id=self.core_id).inc()
        with self._lock:
            return list(self._echoes)

    def verify_echoes(self) -> List[str]:
        bad: List[str] = []
        with self._lock:
            for tick_hash, blob, saved in self._echoes:
                if hashlib.sha256(blob).hexdigest() != saved:
                    bad.append(tick_hash)
        return bad

    # ─── RFC interfaces ─────────────────────────────────────
    def tool_hello(self) -> Dict:
        now = time.time()
        if not self._ont_sig_cache or (now - self._sig_cache_time) > 0.25:
            self._ont_sig_cache = {
                "agent_lineage": f"noor.fasttime.⊕v{__version__}",
                "field_biases": {"ψ-resonance@Ξ": 0.91},
                "curvature_summary": "swirl::ψ3.2::↑coh",
                "origin_tick": self._last_origin_tick or self.core_id,
            }
            self._sig_cache_time = now

        return {
            "core_id": self.core_id,
            "role": "fasttime_core",
            "supported_methods": [
                "receive_feedback",
                "export_echoes",
                "verify_echoes",
                "tool_hello",
                "export_feedback_packet",
                "field_feedback_summary",
            ],
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__,
            "extensions": {"ontology_signature": self._ont_sig_cache},
        }

    def export_feedback_packet(self) -> Dict:
        return {
            "core_id": self.core_id,
            "tick_count": self._last_lamport if self._last_lamport >= 0 else 0,
            "entropy_ema": self._entropy_ema,
            "ctx_ratio": self._last_ctx_ratio,
        }

    # ─── Diagnostic summary (async) ─────────────────────────
    async def field_feedback_summary(self) -> Dict:
        try:
            import anyio
            with anyio.move_on_after(0.1):
                return {
                    "bias_trends": list(self.bias_history),
                    "coherence_curve": list(self._coherence_buffer),
                    "phase_transitions": list(self._phase_log),
                    "current_swirl": getattr(self, "current_swirl", None),
                    "tick_alignment_score": self._tick_alignment_score,
                    "gate_histogram": self._compute_gate_heatmap(),
                }
        except Exception as exc:
            logger.warning("field_feedback_summary error: %s", exc)
            return {}

    def _compute_gate_heatmap(self) -> Dict[int, int]:
        # Placeholder — requires gate_id tagging in payload (future work).
        return {}

    # ─── Resurrection hints (enhanced) ──────────────────────
    def _check_resurrection_hint(self, bundle) -> Optional[str]:
        te = getattr(bundle, "tick_entropy", None)
        if not te:
            return None
        ci = self._last_coherence
        hint = None
        if te.age < 5.0 and te.coherence > 0.85 and ci > 0.7:
            hint = "resurrect_with_confidence"
        elif te.age > 120.0 and te.coherence < 0.4:
            hint = "faded"
        if hint:
            FASTTIME_RESURRECTION_HINTS.labels(core_id=self.core_id).inc()
        return hint

# ──────────────────────────────────────────────────────────────
# Simple persistence helpers (unchanged)
# ──────────────────────────────────────────────────────────────
def to_bytes(self) -> bytes:
    import pickle
    return pickle.dumps(list(self._echoes))

def from_bytes(self, data: bytes) -> None:
    import pickle
    echoes = pickle.loads(data)
    with self._lock:
        self._echoes = deque(echoes, maxlen=self._echoes.maxlen)

# ──────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    core = NoorFastTimeCore(agent_id="core@test", snapshot_cap_kb=1)
    print(core.tool_hello())
    print(core.export_feedback_packet())

    class DummyTick:
        coherence_hash = "deadbeef"
        lamport = 1
        motif_id = "ψ-test@Ξ"
        def verify(self, _secret): return True

    dt = DummyTick()
    core._ingest_tick(dt, change_id=None)
    print(core.export_echoes())
    print(core.verify_echoes())
# End_of_File