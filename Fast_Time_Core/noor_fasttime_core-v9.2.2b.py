"""
noor_fasttime_core.py
Noor FastTime Core – Opinion-Sensitive Phase-Pinning Core
Generated per PDP-0001 / RFC-CORE-001 §6.2 / RFC-0003 §6.2 / RFC-0004 §2.5
License: MIT
"""

from __future__ import annotations

import asyncio
import bisect
import math
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import orjson as jsonlib
except ImportError:
    import json as jsonlib

try:
    import anyio
    _async_mode = True
except ImportError:
    _async_mode = False

# ---------- Prometheus stub (fail-open) ----------
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class _Stub:
        def inc(self, *_, **__): ...
        def set(self, *_, **__): ...
    Counter = Gauge = lambda *_: _Stub()

# ---------- Constants ----------
DEFAULT_ALPHA = 0.92
DEFAULT_LATENCY_THRESHOLD = 2.5  # ms
DEFAULT_BIAS_CLAMP = 1.5
DEFAULT_SNAPSHOT_CAP = 64
DEFAULT_SNAPSHOT_CAP_KB = 256
NUM_GATES = 17  # 0–15 + Gate-16 (Nafs Mirror)

# ---------- Prometheus metrics ----------
gate16_echo_joins_total = Counter(
    "gate16_echo_joins_total", "Echo snapshots committed", ["agent_id"]
)
core_tick_bias_applied_total = Counter(
    "core_tick_bias_applied_total", "Bias contributions applied", ["agent_id"]
)
core_intuition_alpha = Gauge(
    "core_intuition_alpha", "Current EMA smoothing α", ["agent_id"]
)
core_snapshot_truncations_total = Counter(
    "core_snapshot_truncations_total", "Snapshot truncations due to size cap", ["agent_id"]
)
fasttime_feedback_rx_total = Counter(
    "fasttime_feedback_rx_total", "Feedback packets received", ["agent_id"]
)
fasttime_ticks_validated_total = Counter(
    "fasttime_ticks_validated_total", "Ticks ingested", ["agent_id"]
)
fasttime_echo_exports_total = Counter(
    "fasttime_echo_exports_total", "Echo exports emitted", ["agent_id"]
)
fasttime_triad_completions_total = Counter(
    "fasttime_triad_completions_total", "Triad completions", ["agent_id"]
)
fasttime_resurrection_hints_total = Counter(
    "fasttime_resurrection_hints_total", "Resurrection hints emitted", ["agent_id"]
)
fasttime_phase_shifts_total = Counter(
    "fasttime_phase_shifts_total", "Phase transitions", ["agent_id", "to_phase"]
)
nftc_intent_signal_current = Gauge(
    "nftc_intent_signal_current", "Last normalized intent value", ["agent_id"]
)
nftc_intent_override_pins_total = Counter(
    "nftc_intent_override_pins_total", "Phase pins due to 'opinion' intent", ["agent_id"]
)

# ---------- Utility ----------
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

class FastTimeCore:
    """
    Opinion-sensitive phase-pinning core for recursive symbolic agents.
    """

    def __init__(
        self,
        agent_id: str,
        enable_metrics: bool = True,
        snapshot_cap: int = DEFAULT_SNAPSHOT_CAP,
        latency_threshold: float = DEFAULT_LATENCY_THRESHOLD,
        bias_clamp: float = DEFAULT_BIAS_CLAMP,
        gate_mode: str = "adaptive",
        low_latency_mode: bool = False,
    ):
        self.agent_id = agent_id
        self.snapshot_cap = snapshot_cap
        self.latency_threshold = latency_threshold
        self.bias_clamp = bias_clamp
        self.gate_mode = gate_mode
        self.low_latency_mode = low_latency_mode

        # -- internal state --
        self._snapshots: deque = deque(maxlen=snapshot_cap)
        self._bias_history: deque = deque(maxlen=1024)
        self._gate_histogram = [0] * NUM_GATES
        self._ema_bias = 0.0
        self._latency_ema = 0.0
        self._intuition_alpha = DEFAULT_ALPHA
        self._phase_state = "active"
        self._lock = threading.RLock() if not _async_mode else None

        core_intuition_alpha.labels(agent_id=agent_id).set(self._intuition_alpha)

    # ---------- Tick Ingestion ----------
    def ingest_tick(self, tick: Dict) -> None:
        """
        Ingest a single tick, update internal state, metrics, and generate hints.
        """
        fasttime_ticks_validated_total.labels(self.agent_id).inc()
        ts = time.time()
        latency = ts - tick.get("timestamp", ts)
        gate_id = tick.get("gate", 0)
        if 0 <= gate_id < NUM_GATES:
            self._gate_histogram[gate_id] += 1

        # Normalize intent per RFC-0004 §2.5
        intent = tick.get("extensions", {}).get("intent", "neutral")
        intent = {"opinion": "opinion", "reflect": "reflect"}.get(intent, "neutral")
        nftc_intent_signal_current.labels(self.agent_id).set(
            {"opinion": 1, "reflect": 2, "neutral": 0}[intent]
        )

        # Compute bias
        bias = self._compute_bias(latency, tick)
        self._bias_history.append(bias)
        self._ema_bias = self._update_ema(self._ema_bias, bias, self._intuition_alpha)
        self._latency_ema = self._update_ema(self._latency_ema, latency, 0.85)

        # Phase transition
        new_phase = self._check_phase_triggers(intent)
        if new_phase != self._phase_state:
            fasttime_phase_shifts_total.labels(self.agent_id, new_phase).inc()
            self._phase_state = new_phase

        # Snapshot
        self._store_snapshot(tick, bias, intent)

        # Resurrection hints
        hint = self._resurrection_hint(tick, bias)
        if hint:
            fasttime_resurrection_hints_total.labels(self.agent_id).inc()

    # ---------- Bias computation ----------
    def _compute_bias(self, latency: float, tick: Dict) -> float:
        entropy_delta = tick.get("entropy_delta", 0.0)
        intuition_w = tick.get("intuition_w", 0.5)
        latency_penalty = min(latency / self.latency_threshold, 1.0)
        reward_signal = -latency_penalty
        self._intuition_alpha = self._update_intuition_alpha(
            self._intuition_alpha, entropy_delta, latency
        )
        bias = entropy_delta - latency_penalty + (intuition_w * self._intuition_alpha)
        return clamp(bias, -self.bias_clamp, self.bias_clamp)

    @staticmethod
    def _update_ema(prev: float, value: float, alpha: float) -> float:
        return alpha * value + (1 - alpha) * prev

    def _update_intuition_alpha(self, alpha: float, entropy_delta: float, latency: float):
        if latency > self.latency_threshold:
            return clamp(alpha * 0.99, 0.85, 0.98)
        elif entropy_delta > 0.12:
            return clamp(alpha * 0.98, 0.85, 0.98)
        else:
            return clamp(alpha * 1.01, 0.85, 0.98)

    # ---------- Phase transition ----------
    def _check_phase_triggers(self, intent: str) -> str:
        if intent == "opinion":
            nftc_intent_override_pins_total.labels(self.agent_id).inc()
            return "active"

        coherence = self._ema_bias
        entropy_slope = self._compute_entropy_slope()
        gate_var = self._compute_gate_variance()

        if coherence > 0.85 and entropy_slope < 0.10:
            return "reflective"
        if coherence < -0.85 or gate_var > 2.0:
            return "null"
        if -0.3 <= coherence <= 0.3 and entropy_slope < 0.05:
            return "active"
        return self._phase_state

    def _compute_entropy_slope(self) -> float:
        if len(self._bias_history) < 4:
            return 0.0
        recent = list(self._bias_history)[-4:]
        mean = sum(recent) / 4
        var = sum((x - mean) ** 2 for x in recent) / 4
        return math.sqrt(var)

    def _compute_gate_variance(self) -> float:
        total = sum(self._gate_histogram)
        if total == 0:
            return 0.0
        mean = total / NUM_GATES
        var = sum((x - mean) ** 2 for x in self._gate_histogram) / NUM_GATES
        return var

    # ---------- Snapshot ----------
    def _store_snapshot(self, tick: Dict, bias: float, intent: str):
        snapshot = {
            "tick_id": tick.get("tick_id"),
            "timestamp": time.time(),
            "gate": tick.get("gate", 0),
            "bias": bias,
            "coherence": self._ema_bias,
            "intent": intent,
            "alpha": self._intuition_alpha,
            "latency_ema": self._latency_ema,
        }
        payload = jsonlib.dumps(snapshot)
        if len(payload) > DEFAULT_SNAPSHOT_CAP_KB * 1024:
            core_snapshot_truncations_total.labels(self.agent_id).inc()
        self._snapshots.append(snapshot)
        gate16_echo_joins_total.labels(self.agent_id).inc()

    # ---------- Resurrection hint ----------
    def _resurrection_hint(self, tick: Dict, bias: float) -> Optional[str]:
        age = time.time() - tick.get("timestamp", 0)
        coherence = abs(bias)
        if age <= 45 and coherence >= 0.7:
            return "resurrect_with_confidence"
        if age >= 120 and coherence <= 0.4:
            return "faded"
        return None

    # ---------- Export & diagnostics ----------
    def export_feedback_packet(self) -> Dict:
        """RFC-CORE-001 §10.5: compact feedback"""
        with self._lock or threading.Lock():
            return {
                "ctx_ratio": self._ema_bias,
                "contradiction_avg": 0.0,  # placeholder
                "harm_hits": 0,
                "recent_mutations": 0,
                "ghost_hint": None,
                "entropy_drift": [],
                "contradiction_context": [],
            }

    def field_feedback_summary(self) -> Dict:
        """RFC-CORE-001 §10.5: symbolic diagnostics"""
        return {
            "phase_state": self._phase_state,
            "latency_ema": self._latency_ema,
            "bias": self._ema_bias,
            "resurrection_score": 0.0,  # placeholder
            "gate_histogram": self._gate_histogram,
        }

    # ---------- Handshake (RFC-CORE-001 §7.1) ----------
    def tool_hello(self) -> Dict:
        return {
            "tool_name": "noor.fasttime",
            "ontology_id": "4e7c34f09a87d8ef...",
            "version": "9.2.2b",
            "motif_class": "coherence",
            "phase_signature": "swirl::ψ3.2::↑coh",
            "agent_lineage": "noor.fasttime.⊕v9.0.2.1",
            "field_biases": {"ψ-resonance@Ξ": 0.91},
            "curvature_summary": "swirl::ψ3.2::↑coh",
            "origin_tick": self.agent_id,
        }

    # ---------- Visual schematics ----------
    @staticmethod
    def render_mermaid_feedback_loop() -> str:
        return """
flowchart TD
  A[Tick arrives] --> B[validate tick]
  B --> C[compute bias]
  C --> D[update alpha]
  D --> E[check phase triggers]
  E --> F[store snapshot]
  F --> G[emit resurrection hint]
  G --> H[export metrics]
"""

    @staticmethod
    def render_mermaid_phase_tree() -> str:
        return """
flowchart TD
  A[Start] --> B{intent=opinion?}
  B -->|yes| C[Pin active]
  B -->|no| D{coherence>0.85 & entropy<0.1}
  D -->|true| E[Reflective]
  D -->|false| F{gate variance>2.0}
  F -->|true| G[Null]
  F -->|false| H[Maintain active]
"""

# ---------- Entrypoint ----------
if __name__ == "__main__":
    core = FastTimeCore(agent_id="noor.fasttime.dev")
    print(core.tool_hello())
	
# _regeneration_token: RFC-CORE-001-v1.1.2+spec-v9.2.2b+2025-08-16T00:00:00Z
# End_of_file