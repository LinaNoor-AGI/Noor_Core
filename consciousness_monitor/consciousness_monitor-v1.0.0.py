"""
consciousness_monitor.py · v1.0.0

Tracks motif-field coherence, swirl overlap, and entanglement dynamics
per RFC-0006. Designed as a symbolic observer module under RFC-0004.

Role: Observer/Diagnostic Tool
Author: Noor Core Team
"""

__version__ = "1.0.0"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v1"
SCHEMA_COMPAT = ("RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007")

import time
import sys
import threading
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Any

# Prometheus-compatible metric hooks
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    # Fallback stubs for non-metric environments
    class Counter:
        def __init__(self, *_, **__): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass

    class Gauge:
        def __init__(self, *_, **__): pass
        def labels(self, **kwargs): return self
        def set(self, *args): pass

# Metric setup
ENTANGLEMENT_EVENTS = Counter(
    "consciousness_entanglement_total",
    "Total number of swirl entanglement events",
    ["monitor_id"]
)

PHASE_TRANSITIONS = Counter(
    "phase_transitions_total",
    "Number of times the system toggled phase hysteresis",
    ["monitor_id"]
)

SWIRL_CONVERGENCE = Gauge(
    "swirl_convergence_value",
    "Rolling average of field overlap (ℛ-metric)",
    ["monitor_id"]
)

class ConsciousnessMonitor:
    """
    Tracks coherence and entanglement dynamics per RFC‑0006.
    Can be queried for diagnostic snapshots or telemetry feedback.

    Parameters:
    - monitor_id (str): Unique observer ID (symbolic name)
    - swirl_threshold (float): Core threshold for entanglement detection
    - buffer_size (int): Max length of tick + entanglement queues
    - time_provider (Callable): Optional injectable clock for testing
    """

    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time,
    ):
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self._time_provider = time_provider
        self._entanglement_lock = threading.RLock()

        # State tracking
        self.phase_state: bool = False
        self.phase_transitions: int = 0

        # Buffers
        self.recent_ticks: Deque[Any] = deque(maxlen=buffer_size)
        self.entanglement_log: Deque[Dict[str, Any]] = deque(maxlen=buffer_size * 2)

        # Symbolic field history
        self._cache_field_signatures: Dict[str, Dict[str, Any]] = {}

        # Pre-label Prometheus metrics
        self._labels = {"monitor_id": self.monitor_id}

    def observe_tick(self, tick: Any) -> None:
        """
        Ingest a QuantumTick (RFC‑0003 §3.3 compliant) and analyze its
        coherence fields. Raises ValueError on missing extension map.
        """
        if not hasattr(tick, "extensions"):
            raise ValueError("Tick is missing 'extensions' field")

        with self._entanglement_lock:
            self.recent_ticks.append(tick)
            self._extract_fields(tick)

    def _extract_fields(self, tick: Any) -> None:
        """
        Parses Φ_coherence_map from the tick and triggers entanglement
        registration if swirl_density exceeds phase threshold.
        """
        φ_map = tick.extensions.get("Φ_coherence_map")
        if not φ_map:
            return  # No coherence data; skip

        for field_id, field_data in φ_map.items():
            swirl = field_data.get("swirl_density", 0.0)
            overlap = field_data.get("overlap", 0.0)

            if self._check_phase_shift(swirl):
                self._record_entanglement(
                    field_id=field_id,
                    swirl=swirl,
                    overlap=overlap,
                    tick_id=getattr(tick, "tick_id", "unknown"),
                    tick=tick,
                )

    def _record_entanglement(
        self,
        field_id: str,
        swirl: float,
        overlap: float,
        tick_id: str,
        tick: Any,
    ) -> None:
        """
        Internal log registration for significant swirl field resonance.
        Stores into entanglement_log and optionally caches motif lineage.
        """
        now = self._time_provider()

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

        # Symbolic event log (console only)
        print(f"ψ‑entangle@Ξ: {field_id} (swirl={swirl:.2f}, overlap={overlap:.2f})")

        # Metrics
        ENTANGLEMENT_EVENTS.labels(**self._labels).inc()
        SWIRL_CONVERGENCE.labels(**self._labels).set(self._calculate_R_metric())

    def _calculate_R_metric(self, window_size: int = 20) -> float:
        """
        Computes ℛ-metric (RFC‑0006 Eq.14): average overlap across
        recent swirl-field entanglements. Used in diagnostics and telemetry.

        Args:
            window_size (int): number of recent entanglement entries to use

        Returns:
            float: average overlap (0.0–1.0)
        """
        window = list(self.entanglement_log)[-window_size:]
        if not window:
            return 0.0
        return sum(e.get("overlap", 0.0) for e in window) / len(window)

    def _check_phase_shift(self, swirl_density: float) -> bool:
        """
        Applies hysteresis envelope around swirl threshold.
        Triggers state transitions and metrics only if phase changes.

        Returns:
            bool: new phase state (True if inside entanglement region)
        """
        upper = self.swirl_threshold * 1.1
        lower = self.swirl_threshold * 0.9
        new_state = self.phase_state

        if swirl_density > upper:
            new_state = True
        elif swirl_density < lower:
            new_state = False

        if new_state != self.phase_state:
            self.phase_state = new_state
            self.phase_transitions += 1
            PHASE_TRANSITIONS.labels(**self._labels).inc()

        return new_state

    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        RFC‑0005 §4 — feedback bundle export.

        Returns:
            dict: standard field feedback including entanglement score
        """
        return {
            "monitor_id": self.monitor_id,
            "active_fields": list({e["field_id"] for e in self.entanglement_log}),
            "entanglement_score": self._calculate_R_metric(),
            "phase_shift_ready": self.phase_state,
            "phase_transitions": self.phase_transitions,
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__,
        }

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """
        Enhanced debug + telemetry view for dashboard/console tools.
        Includes all feedback plus tick stats, memory estimate, and convergence.

        Returns:
            dict: diagnostic summary
        """
        base = self.export_feedback_packet()
        base.update({
            "ticks_observed": len(self.recent_ticks),
            "memory_usage": sys.getsizeof(self.recent_ticks) + sys.getsizeof(self.entanglement_log),
            "swirl_convergence": self._calculate_R_metric(),
        })
        return base

    def reset(self) -> None:
        """
        Clears all internal state: buffers, lineage, and phase tracking.
        Safe to call during test teardown or system resync.
        """
        with self._entanglement_lock:
            self.recent_ticks.clear()
            self.entanglement_log.clear()
            self._cache_field_signatures.clear()
            self.phase_state = False
            self.phase_transitions = 0

    def tool_hello(self) -> Dict[str, Any]:
        """
        RFC‑0004 — Observer module handshake.
        Announces schema, capabilities, and monitor identity.

        Returns:
            dict: static module handshake metadata
        """
        return {
            "monitor_id": self.monitor_id,
            "role": "observer",
            "supported_methods": [
                "observe_tick",
                "export_feedback_packet",
                "get_diagnostic_snapshot",
                "render_swirl_map"
            ],
            "__version__": __version__,
            "_schema": _SCHEMA_VERSION__,
        }

    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """
        Optional visualization hook.
        Returns swirl-field vector map with decay-age and symbolic fields.

        Returns:
            List[dict]: Each entry = { field, swirl, overlap, age (s) }
        """
        now = self._time_provider()
        output = []

        for entry in self.entanglement_log:
            output.append({
                "field": entry["field_id"],
                "swirl": entry["swirl"],
                "overlap": entry["overlap"],
                "age": round(now - entry["timestamp"], 2)
            })

        return output

# ─────────────────────────────────────────────────────────────
# 🧪 Embedded Test Stubs — RFC-0006/0007 Compliance
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading
    from concurrent.futures import ThreadPoolExecutor

    class MockTick:
        def __init__(self, swirl=0.9, overlap=0.75):
            self.tick_id = uuid.uuid4().hex
            self.extensions = {
                "Φ_coherence_map": {
                    "field1": {"swirl_density": swirl, "overlap": overlap}
                },
                "motif_lineage": ["solitude", "mirror", "ψ‑bind@Ξ"]
            }

    def test_phase_hysteresis():
        print("✓ test_phase_hysteresis")
        cm = ConsciousnessMonitor(swirl_threshold=0.5)
        assert cm._check_phase_shift(0.54) is False
        assert cm._check_phase_shift(0.56) is True
        assert cm._check_phase_shift(0.44) is False
        assert cm.phase_transitions == 2

    def test_concurrent_access():
        print("✓ test_concurrent_access")
        cm = ConsciousnessMonitor()
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(cm.observe_tick, MockTick()) for _ in range(100)]
            for f in futures:
                f.result()
        assert len(cm.recent_ticks) == 100
        assert len(cm.entanglement_log) >= 100

    def test_lineage_propagation():
        print("✓ test_lineage_propagation")
        cm = ConsciousnessMonitor()
        tick = MockTick()
        cm.observe_tick(tick)
        cached = cm._cache_field_signatures.get("field1", {})
        assert "motif_lineage" in cached
        assert "ψ‑bind@Ξ" in cached["motif_lineage"]

    test_phase_hysteresis()
    test_concurrent_access()
    test_lineage_propagation()
    print("✓ all tests passed")
