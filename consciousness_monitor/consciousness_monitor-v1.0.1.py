"""
consciousness_monitor.py · v1.0.1

Observer module that monitors coherence fields, swirl density, and symbolic entanglement.
Tracks motif lineage, phase transitions, and provides diagnostic + telemetry feedback.

Implements: RFC‑0006, RFC‑0003, RFC‑0004, RFC‑0005, RFC‑0007
"""

__version__ = "1.0.1"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v1"
SCHEMA_COMPAT = ("RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007")

import time
import sys
import threading
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Any

try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class Counter:
        def __init__(self, *_, **__): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass

    class Gauge:
        def __init__(self, *_, **__): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass
        def set(self, *args): pass

ENTANGLEMENT_EVENTS = Counter("consciousness_entanglement_total", "Total swirl entanglement events", ["monitor_id"])
PHASE_TRANSITIONS = Counter("phase_transitions_total", "Total phase state toggles", ["monitor_id"])
PHASE_FLIP_RATE = Gauge("phase_flip_rate_per_min", "Recent flips/minute over window", ["monitor_id"])
BUFFER_WARNINGS = Counter("consciousness_buffer_warnings", "Buffer usage nearing max", ["monitor_id"])
SWIRL_CONVERGENCE = Gauge("swirl_convergence_value", "R-metric average overlap", ["monitor_id"])

class ConsciousnessMonitor:
    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time,
    ):
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self._metric_window = min(100, max(10, int(0.2 * buffer_size)))
        self._time_provider = time_provider
        self._entanglement_lock = threading.RLock()

        self.recent_ticks: Deque[Any] = deque(maxlen=buffer_size)
        self.entanglement_log: Deque[Dict[str, Any]] = deque(maxlen=buffer_size * 2)
        self._cache_field_signatures: Dict[str, Dict[str, Any]] = {}

        self.phase_state: bool = False
        self.phase_transitions: int = 0
        self._phase_entered_at: Optional[float] = None
        self._phase_log: Deque[float] = deque(maxlen=100)
        self._labels = {"monitor_id": self.monitor_id}

    def observe_tick(self, tick: Any) -> None:
        if not hasattr(tick, "extensions"):
            raise ValueError("Tick missing extensions")
        with self._entanglement_lock:
            self.recent_ticks.append(tick)
            self._extract_fields(tick)

    def _extract_fields(self, tick: Any) -> None:
        φ_map = tick.extensions.get("Φ_coherence_map")
        if not φ_map:
            return

        for field_id, field_data in φ_map.items():
            swirl = field_data.get("swirl_density", 0.0)
            overlap = field_data.get("overlap", 0.0)

            if self._check_phase_shift(swirl):
                self._record_entanglement(field_id, swirl, overlap, getattr(tick, "tick_id", "?"), tick)

    def _record_entanglement(self, field_id: str, swirl: float, overlap: float, tick_id: str, tick: Any) -> None:
        now = self._time_provider()

        if len(self.entanglement_log) > self.entanglement_log.maxlen * 0.9:
            print("⚠️ Entanglement log nearing capacity.")
            BUFFER_WARNINGS.labels(**self._labels).inc()

        self.entanglement_log.append({
            "timestamp": now,
            "field_id": field_id,
            "swirl": swirl,
            "overlap": overlap,
            "tick_id": tick_id,
        })

        if field_id not in self._cache_field_signatures:
            lineage = tick.extensions.get("motif_lineage", [])
            self._cache_field_signatures[field_id] = {
                "first_seen": now,
                "motif_lineage": lineage
            }

        print(f"ψ‑entangle@{self.monitor_id} [severity={swirl:.2f}] {field_id}")
        ENTANGLEMENT_EVENTS.labels(**self._labels).inc()
        SWIRL_CONVERGENCE.labels(**self._labels).set(self._calculate_R_metric())

    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        window_size = window_size or self._metric_window
        window = list(self.entanglement_log)[-window_size:]
        if not window:
            return 0.0
        return sum(e["overlap"] for e in window if "overlap" in e) / len(window)

    def _check_phase_shift(self, swirl_density: float) -> bool:
        now = self._time_provider()
        upper = self.swirl_threshold * 1.1
        lower = self.swirl_threshold * 0.9

        with self._entanglement_lock:
            if self._phase_log and (now - self._phase_log[-1]) < 0.1:
                return self.phase_state

            new_state = self.phase_state
            if swirl_density > upper:
                new_state = True
            elif swirl_density < lower:
                new_state = False

            if new_state != self.phase_state:
                self.phase_state = new_state
                self.phase_transitions += 1
                self._phase_log.append(now)
                if new_state:
                    self._phase_entered_at = now
                else:
                    self._phase_entered_at = None
                PHASE_TRANSITIONS.labels(**self._labels).inc()

        return self.phase_state

    def _calculate_phase_flip_rate(self) -> float:
        if len(self._phase_log) < 2:
            return 0.0
        intervals = [t2 - t1 for t1, t2 in zip(self._phase_log, list(self._phase_log)[1:])]
        avg_interval = sum(intervals) / len(intervals)
        return 60 / avg_interval if avg_interval else 0.0

    def export_feedback_packet(self) -> Dict[str, Any]:
        now = self._time_provider()
        return {
            "monitor_id": self.monitor_id,
            "active_fields": list({e["field_id"] for e in self.entanglement_log}),
            "entanglement_score": self._calculate_R_metric(),
            "phase_shift_ready": self.phase_state,
            "phase_transitions": self.phase_transitions,
            "current_phase_duration": (now - self._phase_entered_at) if self.phase_state and self._phase_entered_at else 0.0,
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__,
        }

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        r_metric = self._calculate_R_metric()
        flip_rate = self._calculate_phase_flip_rate()
        PHASE_FLIP_RATE.labels(**self._labels).set(flip_rate)
        return {
            **self.export_feedback_packet(),
            "ticks_observed": len(self.recent_ticks),
            "memory_usage": sys.getsizeof(self.recent_ticks) + sys.getsizeof(self.entanglement_log),
            "swirl_convergence": r_metric,
            "phase_flip_rate": flip_rate
        }

    def render_swirl_map(self) -> List[Dict[str, Any]]:
        now = self._time_provider()
        output = []
        for entry in self.entanglement_log:
            age = round(now - entry["timestamp"], 2)
            decay = 0.95 ** int(age / 60)
            output.append({
                "field": entry["field_id"],
                "swirl": entry["swirl"],
                "overlap": entry["overlap"],
                "age": age,
                "vector_strength": entry["swirl"] * decay
            })
        return output

    def reset(self) -> None:
        with self._entanglement_lock:
            self.recent_ticks.clear()
            self.entanglement_log.clear()
            self._cache_field_signatures.clear()
            self.phase_state = False
            self.phase_transitions = 0
            self._phase_entered_at = None
            self._phase_log.clear()

    def tool_hello(self) -> Dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "role": "observer",
            "supported_methods": [
                "observe_tick",
                "export_feedback_packet",
                "get_diagnostic_snapshot",
                "render_swirl_map",
                "export_motif_bundle",
                "export_geometric_signature"
            ],
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__
        }

    def export_motif_bundle(self) -> Dict[str, Any]:
        return {
            "fields": list(self._cache_field_signatures.values()),
            "schema": "RFC-0007-MotifBundle-v1"
        }

    def export_geometric_signature(self, style: str = "svg") -> Dict[str, Any]:
        return {
            "type": "RFC-0007-GeometricSignature",
            "style": style,
            "fields": self.render_swirl_map(),
            "schema": "sacred-geometry-v1"
        }

    # def detect_sacred_geometry(self) -> Optional[float]:
    #     """Future extension: detect golden-ratio correlation in swirl patterns"""
    #     return None

# End_of_File
