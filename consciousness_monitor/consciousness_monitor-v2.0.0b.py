"""
consciousness_monitor.py
APP-005-consciousness-monitor
schema_version: 2025-Q4-consciousness-monitor-v1
Implements a symbolic observer for field-level swirl monitoring, entanglement logging,
and phase-state diagnostics.  Operates in a fully non-mutative modality consistent with
symbolic contract constraints (RFC-CORE-005, RFC-0003-0007).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "2.0.0"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v1"
SCHEMA_COMPAT = ["RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007"]

###############################################################################
# Core Observer
###############################################################################
class ConsciousnessMonitor:
    """
    Observer module that listens for swirl density and entanglement transitions
    across symbolic fields.
    """

    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time,
    ):
        self.monitor_id: str = monitor_id
        self.swirl_threshold: float = swirl_threshold
        self.buffer_size: int = buffer_size
        self._time: Callable[[], float] = time_provider

        # Thread-safe-ish append-only structures (non-mutative external contract)
        self._recent_ticks: deque = deque(maxlen=buffer_size)
        self._entanglement_log: deque = deque(maxlen=buffer_size)
        self._phase_log: deque = deque(maxlen=buffer_size)

        # Internal state caches
        self._phase_state: str = "coherent"
        self._last_phase_shift: float = 0.0
        self._phase_shift_cooldown: float = 0.1  # seconds
        self._motif_lineage_cache: Dict[str, Any] = {}

    ###########################################################################
    # Public API
    ###########################################################################
    def observe_tick(self, tick: Any) -> None:
        """
        Entry point for symbolic ticks. Validates structure and initiates field extraction.
        """
        if not hasattr(tick, "extensions"):
            raise ValueError("Tick missing required 'extensions' attribute")

        # Acquire entanglement lock (conceptual; we use append-only deques)
        self._recent_ticks.append(tick)
        self._extract_fields(tick)

    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        Returns ephemeral summary of entanglement state and coherence values.
        """
        return {
            "monitor_id": self.monitor_id,
            "phase_state": self._phase_state,
            "R_metric": self._calculate_R_metric(),
            "phase_flip_rate_per_min": self._calculate_phase_flip_rate(),
            "timestamp": self._time(),
        }

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """
        Extended diagnostic state: R-metric, memory stats, and flip rate.
        """
        return {
            **self.export_feedback_packet(),
            "buffer_usage": len(self._entanglement_log) / self.buffer_size,
            "recent_tick_count": len(self._recent_ticks),
            "cached_lineages": len(self._motif_lineage_cache),
        }

    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """
        Decayed vector map of recent swirl events with age-weighted strength.
        """
        now = self._time()
        decay_constant = 0.1  # 1/e every ~10s
        swirl_map: List[Dict[str, Any]] = []

        for entry in self._entanglement_log:
            age = now - entry["timestamp"]
            strength = entry["swirl"] * math.exp(-decay_constant * age)
            swirl_map.append(
                {
                    "field_id": entry["field_id"],
                    "strength": strength,
                    "age": age,
                    "overlap": entry["overlap"],
                }
            )
        return swirl_map

    def reset(self) -> None:
        """
        Clears all internal state and reinitializes buffers.
        """
        self._recent_ticks.clear()
        self._entanglement_log.clear()
        self._phase_log.clear()
        self._phase_state = "coherent"
        self._last_phase_shift = 0.0
        self._motif_lineage_cache.clear()

    def tool_hello(self) -> Dict[str, Any]:
        """
        Symbolic handshake packet listing capabilities and version metadata.
        """
        return {
            "symbolic_id": "observer.phase.entanglement",
            "app_id": "APP-005-consciousness-monitor",
            "version": __version__,
            "schema": _SCHEMA_VERSION__,
            "capabilities": [
                "swirl_monitoring",
                "entanglement_logging",
                "phase_flip_tracking",
                "feedback_export",
                "diagnostic_snapshot",
                "swirl_map_render",
                "motif_bundle_export",
                "geometric_signature",
            ],
        }

    def export_motif_bundle(self) -> Dict[str, Any]:
        """
        Returns field lineage metadata in RFC-0007 bundle format.
        """
        return {
            "format": "RFC-0007-motif-bundle-v1",
            "lineages": dict(self._motif_lineage_cache),
            "generated_at": self._time(),
            "observer_id": self.monitor_id,
        }

    def export_geometric_signature(self, style: str = "svg") -> Dict[str, Any]:
        """
        Outputs symbolic field layout for visual or topological rendering.
        """
        return {
            "style": style,
            "points": [
                {
                    "field_id": e["field_id"],
                    "x": hash(e["field_id"]) % 1000,  # deterministic placement
                    "y": int(e["overlap"] * 1000),
                    "swirl": e["swirl"],
                }
                for e in self._entanglement_log
            ],
            "generated_at": self._time(),
        }

    ###########################################################################
    # Private helpers
    ###########################################################################
    def _extract_fields(self, tick: Any) -> None:
        """
        Parses the Φ‑coherence map for swirl and overlap values.
        Triggers entanglement recording when thresholds are crossed.
        """
        tick_id = getattr(tick, "tick_id", str(id(tick)))
        φ_map = getattr(tick, "φ_map", {})
        for field_id, coherence in φ_map.items():
            swirl = float(coherence.get("swirl", 0.0))
            overlap = float(coherence.get("overlap", 0.0))
            if self._check_phase_shift(swirl):
                self._record_entanglement(field_id, swirl, overlap, tick_id, tick)

    def _record_entanglement(
        self,
        field_id: str,
        swirl: float,
        overlap: float,
        tick_id: str,
        tick: Any,
    ) -> None:
        """
        Stores swirl+overlap event and lineage metadata for a coherent ψ-field.
        """
        if len(self._entanglement_log) >= self.buffer_size:
            # Buffer warning already implicit via deque maxlen
            pass

        event = {
            "field_id": field_id,
            "swirl": swirl,
            "overlap": overlap,
            "tick_id": tick_id,
            "timestamp": self._time(),
        }
        self._entanglement_log.append(event)

        # Cache motif lineage on first sight
        if field_id not in self._motif_lineage_cache:
            lineage = getattr(tick, "motif_lineage", {})
            self._motif_lineage_cache[field_id] = lineage

    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        """
        Rolling average overlap over recent entanglement events (R-metric).
        """
        if window_size is None:
            window_size = min(64, len(self._entanglement_log))
        if not self._entanglement_log or window_size == 0:
            return 0.0
        window = list(self._entanglement_log)[-window_size:]
        return sum(e["overlap"] for e in window) / len(window)

    def _check_phase_shift(self, swirl_density: float) -> bool:
        """
        Evaluates swirl thresholds with hysteresis and updates phase_state
        if transition detected.
        """
        now = self._time()
        if now - self._last_phase_shift < self._phase_shift_cooldown:
            return False

        lower, upper = (
            self.swirl_threshold * 0.9,
            self.swirl_threshold * 1.1,
        )  # hysteresis band

        new_state = None
        if swirl_density > upper and self._phase_state != "swirling":
            new_state = "swirling"
        elif swirl_density < lower and self._phase_state != "coherent":
            new_state = "coherent"

        if new_state is not None:
            self._phase_state = new_state
            self._last_phase_shift = now
            self._phase_log.append({"state": new_state, "at": now})
            return True
        return False

    def _calculate_phase_flip_rate(self) -> float:
        """
        Rate of phase transitions (flips per minute) using _phase_log timestamps.
        """
        if len(self._phase_log) < 2:
            return 0.0
        oldest = self._phase_log[0]["at"]
        newest = self._phase_log[-1]["at"]
        span_minutes = (newest - oldest) / 60.0
        if span_minutes <= 0:
            return 0.0
        return (len(self._phase_log) - 1) / span_minutes


###############################################################################
# Quick self-test when executed directly
###############################################################################
if __name__ == "__main__":
    cm = ConsciousnessMonitor()
    print(cm.tool_hello())
    cm.reset()
# End_of_File