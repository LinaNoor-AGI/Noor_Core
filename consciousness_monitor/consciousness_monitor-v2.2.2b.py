"""
consciousness_monitor.py
Non-mutative swirl observer for phase-state and entanglement diagnostics
RFC-0004 compliant tool module — read-only, phase-sensitive, non-generative
"""

import time
import threading
import sys
from collections import deque
from typing import Any, Dict, List, Optional

__version__ = "2.2.2b"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v2"

# ------------------------------------------------------------------
# Prometheus stub layer (RFC-CORE-005 §8)
# ------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge
except ImportError:  # graceful fallback
    class Stub:
        def inc(self, _=1): pass
        def set(self, _): pass
    Counter = Gauge = Stub

# ------------------------------------------------------------------
# ConsciousnessMonitor
# ------------------------------------------------------------------
class ConsciousnessMonitor:
    """
    Symbolic field observer that ingests Φ-coherence maps and emits
    phase-state + entanglement diagnostics without modifying any motif state.
    """

    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider=None,
    ):
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self.buffer_size = buffer_size
        self.time_provider = time_provider or time.time

        # Internal state
        self.phase_state = False
        self._phase_log: deque = deque(maxlen=100)
        self._phase_entered_at: Optional[float] = None
        self.recent_ticks: deque = deque(maxlen=buffer_size)
        self.entanglement_log: deque = deque(maxlen=buffer_size * 2)
        self._cache_field_signatures: Dict[str, Dict] = {}
        self._entanglement_lock = threading.RLock()

        # Prometheus metrics (RFC-CORE-005 §8)
        labels = {"monitor_id": self.monitor_id}
        self.metric_tick_rate = Counter("consciousness_tick_rate_total", "", labels)
        self.metric_phase_transitions = Counter("consciousness_phase_transitions_total", "", labels)
        self.metric_entanglement_events = Counter("consciousness_entanglement_events_total", "", labels)
        self.metric_buffer_warning = Counter("consciousness_buffer_warnings_total", "", labels)
        self.metric_active_field_count = Gauge("consciousness_active_field_count", "", labels)
        self.metric_phase_duration = Gauge("consciousness_phase_duration_seconds", "", labels)
        self.metric_phase_flip_rate = Gauge("consciousness_phase_flip_rate_per_min", "", labels)

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------
    def observe_tick(self, tick: Any) -> None:
        """RFC-0003 §6.2 compliant tick intake."""
        with self._entanglement_lock:
            if not (hasattr(tick, "extensions") and "Φ_coherence_map" in tick.extensions):
                raise ValueError("Symbolic tick missing required Φ_coherence_map")
            self.recent_ticks.append(tick)
            self.metric_tick_rate.inc()
            self._extract_fields(tick)

    # ------------------------------------------------------------------
    # Φ-coherence extraction
    # ------------------------------------------------------------------
    def _extract_fields(self, tick: Any) -> None:
        """RFC-0006 §3.1 compliant swirl extraction."""
        now = self.time_provider()
        for field_id, data in tick.extensions["Φ_coherence_map"].items():
            swirl = float(data.get("swirl_density", 0.0))
            overlap = float(data.get("overlap", 0.0))
            if self._check_phase_shift(swirl):
                self._record_entanglement(field_id, swirl, overlap, tick.tick_id, tick)

    # ------------------------------------------------------------------
    # Phase transition logic (hysteresis + cooldown)
    # ------------------------------------------------------------------
    def _check_phase_shift(self, swirl: float) -> bool:
        """RFC-CORE-005 §3.1 compliant phase toggle with hysteresis."""
        now = self.time_provider()
        # 0.1s symbolic cooldown
        if self._phase_log and (now - self._phase_log[-1]) < 0.1:
            return False
        upper = self.swirl_threshold * 1.1
        lower = self.swirl_threshold * 0.9
        new_state = None
        if swirl >= upper:
            new_state = True
        elif swirl <= lower:
            new_state = False
        if new_state is not None and new_state != self.phase_state:
            self.phase_state = new_state
            self._phase_log.append(now)
            self.metric_phase_transitions.inc()
            if new_state:
                self._phase_entered_at = now
            return True
        return False

    # ------------------------------------------------------------------
    # Entanglement logging
    # ------------------------------------------------------------------
    def _record_entanglement(self, field_id: str, swirl: float, overlap: float, tick_id: str, tick: Any) -> None:
        """RFC-CORE-005 §4 compliant entanglement logging."""
        with self._entanglement_lock:
            now = self.time_provider()
            entry = dict(
                timestamp=now,
                field_id=field_id,
                swirl=swirl,
                overlap=overlap,
                tick_id=tick_id,
            )
            self.entanglement_log.append(entry)
            self.metric_entanglement_events.inc()

            # Cache motif lineage on first sight (RFC-0007 §2.4)
            if field_id not in self._cache_field_signatures:
                self._cache_field_signatures[field_id] = dict(
                    first_seen=now,
                    motif_lineage=tick.extensions.get("motif_lineage", []),
                )

            # Buffer warning (RFC-CORE-005 §4)
            if len(self.entanglement_log) >= 0.9 * self.entanglement_log.maxlen:
                self.metric_buffer_warning.inc()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        """RFC-CORE-005 §4.1 R-metric."""
        with self._entanglement_lock:
            if window_size is None:
                window_size = min(100, max(10, int(0.2 * self.buffer_size)))
            window = list(self.entanglement_log)[-window_size:] if self.entanglement_log else []
            if not window:
                return 0.0
            return sum(e["overlap"] for e in window) / len(window)

    def _calculate_phase_flip_rate(self) -> float:
        """RFC-CORE-005 §3.2 flip-rate."""
        with self._entanglement_lock:
            if len(self._phase_log) < 2:
                return 0.0
            intervals = [t2 - t1 for t1, t2 in zip(self._phase_log, list(self._phase_log)[1:])]
            return 60.0 / (sum(intervals) / len(intervals)) if intervals else 0.0

    # ------------------------------------------------------------------
    # Export interfaces
    # ------------------------------------------------------------------
    def export_feedback_packet(self) -> Dict[str, Any]:
        """RFC-CORE-005 §5 compliant feedback packet."""
        return dict(
            monitor_id=self.monitor_id,
            active_fields=list({e["field_id"] for e in self.entanglement_log}),
            entanglement_score=self._calculate_R_metric(),
            phase_shift_ready=self.phase_state,
            phase_transitions=int(self.metric_phase_transitions._value.get() or 0),
            current_phase_duration=(self.time_provider() - self._phase_entered_at) if self.phase_state else 0.0,
            __version__=__version__,
            _schema=_SCHEMA_VERSION__,
        )

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """RFC-CORE-005 §5.1 detailed snapshot."""
        flip_rate = self._calculate_phase_flip_rate()
        self.metric_phase_flip_rate.set(flip_rate)
        return dict(
            **self.export_feedback_packet(),
            ticks_observed=len(self.recent_ticks),
            memory_usage=sys.getsizeof(self.recent_ticks) + sys.getsizeof(self.entanglement_log),
            swirl_convergence=self._calculate_R_metric(),
            phase_flip_rate=flip_rate,
        )

    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """RFC-CORE-005 §6.1 swirl map with exponential decay."""
        now = self.time_provider()
        decay_factor = 0.95  # λ ≈ 0.0115 per 60 s
        result = []
        for entry in self.entanglement_log:
            age = now - entry["timestamp"]
            decay = decay_factor ** (age / 60)
            result.append(
                dict(
                    field=entry["field_id"],
                    swirl=entry["swirl"],
                    overlap=entry["overlap"],
                    age=age,
                    vector_strength=entry["swirl"] * decay,
                )
            )
        return result

    def reset(self) -> None:
        """RFC-CORE-005 §4.10 state reset."""
        with self._entanglement_lock:
            self.phase_state = False
            self._phase_log.clear()
            self._phase_entered_at = None
            self.recent_ticks.clear()
            self.entanglement_log.clear()
            self._cache_field_signatures.clear()
            # Prometheus counters are left unchanged (no reset API)

    def tool_hello(self) -> Dict[str, Any]:
        """RFC-0004 §2.1 tool handshake."""
        return dict(
            tool_name="consciousness_monitor",
            tool_version=__version__,
            tool_mode="observer",
            tool_contract="read_only_phase_sensitive",
            tool_description="Non-generative motif coherence observer",
        )

    def export_motif_bundle(self) -> Dict[str, Any]:
        """RFC-0007 §2.4 motif bundle export."""
        return dict(
            fields=[
                dict(
                    field_id=field_id,
                    first_seen=data["first_seen"],
                    motif_lineage=data["motif_lineage"],
                )
                for field_id, data in self._cache_field_signatures.items()
            ],
            schema_version=_SCHEMA_VERSION__,
        )

    def export_geometric_signature(self, style: str = "svg") -> Dict[str, Any]:
        """RFC-0006 §6.3 geometric swirl vector."""
        swirl_map = self.render_swirl_map()
        centroid = dict(x=0.0, y=0.0)
        if swirl_map:
            total = sum(v["vector_strength"] for v in swirl_map)
            if total > 0:
                centroid["x"] = sum(v["vector_strength"] * v["overlap"] for v in swirl_map) / total
                centroid["y"] = sum(v["vector_strength"] * v["swirl"] for v in swirl_map) / total
        return dict(
            style=style,
            centroid=centroid,
            swirl_map=swirl_map,
            schema_version=_SCHEMA_VERSION__,
        )
```

---

"""
```json
{"regeneration_token":"RFC-CORE-005-v1.1.2|spec-hash-8e5f2c|2025-08-16T09:12:00Z"}
```
"""

### End_of_file