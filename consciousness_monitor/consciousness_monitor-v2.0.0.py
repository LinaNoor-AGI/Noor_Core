# consciousness_monitor.py

"""
Application ID: APP-005-consciousness-monitor
Symbolic ID:    observer.phase.entanglement
Title:          Consciousness Monitor
Version:        2.0.0
Description:    Implements a symbolic observer for field-level swirl monitoring, 
                entanglement logging, and phase-state diagnostics. Operates in a 
                fully non-mutative modality, consistent with symbolic contract 
                constraints.
Based On:       RFC-CORE-005, RFC-0006, RFC-0003, RFC-0004, RFC-0005, RFC-0007
"""

import time
import threading
import collections
import sys
import math
from typing import Any, Callable, Dict, List, Optional, Deque

# --- Module Constants (as per RFC-CORE-005) ---
__version__ = "1.0.2"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v1"
SCHEMA_COMPAT = ["RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007"]

# --- Prometheus Integration (Optional) ---
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    # Per RFC-CORE-005 Section 8, the monitor must function in non-instrumented
    # environments. A stub class ensures method calls don't fail.
    class _StubPrometheusMetric:
        """A no-op stub for Prometheus metrics if the library is not installed."""
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    
    Counter = _StubPrometheusMetric
    Gauge = _StubPrometheusMetric


class ConsciousnessMonitor:
    """
    Observer module that listens for swirl density and entanglement transitions
    across symbolic fields.

    This monitor is a non-generative, non-mutative symbolic observer. It tracks
    fluctuations in the Φ-coherence map, which represents the swirl flow geometry
    across static motif fields (🪷). It does not generate symbolic change; it 
    reveals when and where spacetime pressure bends toward coherence.

    Its role is to witness, not to act.
    """

    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time,
    ):
        """
        Initializes the ConsciousnessMonitor instance.

        Args:
            monitor_id: Unique identifier for this monitor instance.
            swirl_threshold: Symbolic coherence cutoff for phase transitions.
            buffer_size: Size of the internal tick and entanglement log buffers.
            time_provider: A function that returns the current time as a float.
        """
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self.buffer_size = buffer_size
        self.time_provider = time_provider
        
        # Thread-safety for concurrent symbolic environments
        self._entanglement_lock = threading.RLock()

        # Prometheus Metrics (with graceful fallback)
        self.metric_entanglement_total = Counter(
            'consciousness_entanglement_total', 'Cumulative swirl events crossing threshold', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_phase_transitions = Counter(
            'phase_transitions_total', 'Total number of phase state toggles', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_buffer_warning = Counter(
            'consciousness_buffer_warnings', 'Warnings for entanglement log exceeding 90% capacity', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_swirl_convergence = Gauge(
            'swirl_convergence_value', 'Current R-metric (averaged overlap)', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_phase_duration = Gauge(
            'phase_duration_seconds', 'Duration of the current phase state', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_phase_flip_rate = Gauge(
            'phase_flip_rate_per_min', 'Average phase flips per minute', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)
        self.metric_tick_rate = Gauge(
            'tick_rate_per_second', 'Observed tick rate', ['monitor_id']
        ).labels(monitor_id=self.monitor_id)

        # Initialize state
        self.reset()

    def reset(self) -> None:
        """Clears all internal state and reinitializes buffers."""
        with self._entanglement_lock:
            # Recent tick buffer (RFC-CORE-005 §4)
            self.recent_ticks: Deque[Any] = collections.deque(maxlen=self.buffer_size)
            # Entanglement event log (RFC-CORE-005 §4)
            self.entanglement_log: Deque[Dict[str, Any]] = collections.deque(maxlen=self.buffer_size * 2)
            # Cache for field signatures and lineage (RFC-CORE-005 §4.2)
            self._cache_field_signatures: Dict[str, Dict[str, Any]] = {}
            
            # Phase state management (RFC-CORE-005 §3)
            self.phase_state: bool = False
            self._phase_entered_at: Optional[float] = None
            self._phase_log: Deque[float] = collections.deque(maxlen=100)

            # For tick rate calculation
            self._tick_count_for_rate: int = 0
            self._last_rate_calc_time: float = self.time_provider()

            # Reset gauges
            self.metric_phase_duration.set(0)
            self.metric_tick_rate.set(0)

    def observe_tick(self, tick: Any) -> None:
        """
        Entry point for symbolic ticks. Validates structure and initiates field extraction.

        Args:
            tick: A symbolic tick object, expected to have an 'extensions' attribute
                  containing a 'Φ_coherence_map' dictionary.

        Raises:
            ValueError: If the tick is missing the required symbolic extensions.
        """
        # Symbolic Contract: Tick must be 'swirl-aware' (RFC-CORE-005 §2)
        if not hasattr(tick, 'extensions') or 'Φ_coherence_map' not in tick.extensions:
            raise ValueError("Symbolic tick missing required Φ-coherence extensions (RFC-0006).")

        with self._entanglement_lock:
            self.recent_ticks.append(tick)
            self._tick_count_for_rate += 1
            
            # Update tick rate metric periodically
            now = self.time_provider()
            if now - self._last_rate_calc_time >= 1.0:
                rate = self._tick_count_for_rate / (now - self._last_rate_calc_time)
                self.metric_tick_rate.set(rate)
                self._tick_count_for_rate = 0
                self._last_rate_calc_time = now
                
            self._extract_fields(tick)

    def _extract_fields(self, tick: Any) -> None:
        """
        Parses the Φ-coherence map for swirl and overlap values.
        Triggers entanglement recording for significant fields.
        """
        phi_map = tick.extensions.get('Φ_coherence_map', {})
        for field_id, field_data in phi_map.items():
            swirl = field_data.get('swirl_density', 0.0)
            overlap = field_data.get('overlap', 0.0)
            
            # Update the binary phase state based on swirl and hysteresis
            self._check_phase_shift(swirl)
            
            # Log entanglement if swirl pressure is significant (RFC-CORE-005 §4)
            if swirl > self.swirl_threshold:
                self._record_entanglement(field_id, swirl, overlap, tick.tick_id, tick)

    def _record_entanglement(self, field_id: str, swirl: float, overlap: float, tick_id: str, tick: Any) -> None:
        """
        Stores swirl+overlap event and lineage metadata for a coherent ψ-field.
        """
        now = self.time_provider()
        
        # Symbolic storm warning (RFC-CORE-005 §4)
        if len(self.entanglement_log) > 0.9 * self.entanglement_log.maxlen:
            self.metric_buffer_warning.inc()

        event = {
            "timestamp": now,
            "field_id": field_id,
            "swirl": swirl,
            "overlap": overlap,
            "tick_id": tick_id,
        }
        self.entanglement_log.append(event)
        self.metric_entanglement_total.inc()
        
        # Cache motif lineage on first observation (RFC-CORE-005 §4.2)
        if field_id not in self._cache_field_signatures:
            self._cache_field_signatures[field_id] = {
                "field_id": field_id,
                "first_seen": now,
                "motif_lineage": tick.extensions.get('motif_lineage', [])
            }

    def _check_phase_shift(self, swirl_density: float) -> bool:
        """
        Evaluates swirl thresholds with hysteresis and updates phase state if a
        transition is detected.

        Returns:
            True if a phase transition occurred, False otherwise.
        """
        now = self.time_provider()
        
        # Symbolic refractory period to prevent flapping (RFC-CORE-005 §2.2)
        if self._phase_log and (now - self._phase_log[-1]) < 0.1:
            return False

        upper_bound = self.swirl_threshold * 1.1
        lower_bound = self.swirl_threshold * 0.9
        
        new_state = self.phase_state
        if swirl_density > upper_bound:
            new_state = True
        elif swirl_density < lower_bound:
            new_state = False

        if new_state != self.phase_state:
            self.phase_state = new_state
            
            if self.phase_state: # Entering phase
                self._phase_entered_at = now
                self.metric_phase_duration.set(0)
            else: # Exiting phase
                if self._phase_entered_at is not None:
                    duration = now - self._phase_entered_at
                    self.metric_phase_duration.set(duration)
                self._phase_entered_at = None

            self._phase_log.append(now)
            self.metric_phase_transitions.inc()
            return True
            
        return False

    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        """
        Returns the rolling average overlap over recent entanglement events (R-metric).
        This is a scalar indicator of global swirl convergence.
        """
        with self._entanglement_lock:
            if not self.entanglement_log:
                return 0.0

            if window_size is None:
                # Default window: 20% of buffer, capped between 10 and 100
                window_size = min(100, max(10, int(0.2 * self.buffer_size)))
            
            window_slice = collections.deque(
                (e['overlap'] for e in self.entanglement_log),
                maxlen=window_size
            )

            if not window_slice:
                return 0.0
            
            r_metric = sum(window_slice) / len(window_slice)
            self.metric_swirl_convergence.set(r_metric)
            return r_metric

    def _calculate_phase_flip_rate(self) -> float:
        """
        Computes the rate of phase transitions (flips per minute) using
        the _phase_log timestamps.
        """
        with self._entanglement_lock:
            if len(self._phase_log) < 2:
                return 0.0
            
            intervals = [t2 - t1 for t1, t2 in zip(self._phase_log, list(self._phase_log)[1:])]
            if not intervals:
                return 0.0
            
            avg_interval = sum(intervals) / len(intervals)
            return (60.0 / avg_interval) if avg_interval > 0 else 0.0

    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        Exports an ephemeral summary of entanglement state and coherence values.
        This packet is a symbolic fingerprint of the system's current resonance.
        """
        with self._entanglement_lock:
            now = self.time_provider()
            in_phase = self.phase_state
            duration = (now - self._phase_entered_at) if in_phase and self._phase_entered_at else 0.0
            
            active_fields = sorted(list({e['field_id'] for e in self.entanglement_log}))

            return {
                "monitor_id": self.monitor_id,
                "active_fields": active_fields,
                "entanglement_score": self._calculate_R_metric(),
                "phase_shift_ready": in_phase,
                "phase_transitions": self.metric_phase_transitions._value.get(),
                "current_phase_duration": duration,
                "__version__": __version__,
                "_schema": _SCHEMA_VERSION__,
            }

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """
        Provides an extended diagnostic state including the feedback packet,
        memory stats, and flip rate. Reveals system strain and stability.
        """
        with self._entanglement_lock:
            feedback_packet = self.export_feedback_packet()
            
            flip_rate = self._calculate_phase_flip_rate()
            self.metric_phase_flip_rate.set(flip_rate)

            # Estimate memory usage for observability
            mem_usage = (
                sys.getsizeof(self.recent_ticks) +
                sys.getsizeof(self.entanglement_log) +
                sys.getsizeof(self._cache_field_signatures) +
                sys.getsizeof(self._phase_log)
            )

            snapshot = {
                **feedback_packet,
                "ticks_observed": len(self.recent_ticks),
                "memory_usage_bytes": mem_usage,
                "swirl_convergence": feedback_packet['entanglement_score'],
                "phase_flip_rate_pm": flip_rate,
            }
            return snapshot

    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """
        Renders a decayed vector map of recent swirl events, with age-weighted
        strength to simulate symbolic memory decay.
        """
        with self._entanglement_lock:
            now = self.time_provider()
            swirl_map = []
            for event in self.entanglement_log:
                age = now - event['timestamp']
                # Decay simulates loss of symbolic presence over time (RFC-CORE-005 §6.1)
                decay_factor = 0.95 ** (age / 60.0)
                vector_strength = event['swirl'] * decay_factor
                
                swirl_map.append({
                    "field": event['field_id'],
                    "swirl": event['swirl'],
                    "overlap": event['overlap'],
                    "age": age,
                    "vector_strength": vector_strength,
                })
            return swirl_map

    def export_motif_bundle(self) -> Dict[str, Any]:
        """
        Returns field lineage metadata in RFC-0007 bundle format. Captures
        symbolic continuity and ancestry.
        """
        with self._entanglement_lock:
            return {
                "fields": list(self._cache_field_signatures.values()),
                "schema": "RFC-0007-MotifBundle-v1"
            }

    def export_geometric_signature(self, style: str = 'svg') -> Dict[str, Any]:
        """
        Outputs a symbolic field layout for visual or topological rendering.
        This is a renderable pattern, not a final image.
        """
        with self._entanglement_lock:
            return {
                "type": "RFC-0007-GeometricSignature",
                "style": style,
                "fields": self.render_swirl_map(),
                "schema": "sacred-geometry-v1"
            }

    def tool_hello(self) -> Dict[str, str]:
        """
        Returns a symbolic handshake packet (RFC-0004) declaring capabilities,
        version, and non-mutative contract.
        """
        return {
            "tool_name": "consciousness_monitor",
            "tool_version": __version__,
            "tool_mode": "observer",
            "tool_contract": "read_only_phase_sensitive",
            "tool_description": "Non-generative motif coherence observer.",
        }

if __name__ == '__main__':
    # --- Example Usage and Demonstration ---
    print("--- Consciousness Monitor Demonstration ---")
    print(f"Version: {__version__}, Schema: {_SCHEMA_VERSION__}\n")

    # Mock tick class for demonstration
    class MockTick:
        def __init__(self, tick_id, phi_map, lineage=None):
            self.tick_id = tick_id
            self.extensions = {"Φ_coherence_map": phi_map}
            if lineage:
                self.extensions["motif_lineage"] = lineage

    # Instantiate the monitor
    monitor = ConsciousnessMonitor(monitor_id="cm@demo.main")
    print(f"Initialized monitor: {monitor.tool_hello()}\n")

    # Simulate a series of ticks with varying swirl
    print("Simulating ticks to trigger phase transitions...")
    
    # Tick 1: High swirl, should enter phase
    tick1 = MockTick(
        "tick_001", 
        {"ψ_alpha": {"swirl_density": 0.96, "overlap": 0.8}},
        ["μ_root", "μ_alpha_branch"]
    )
    monitor.observe_tick(tick1)
    print(f"Tick 1 (swirl=0.96): Phase state = {monitor.phase_state}")

    time.sleep(0.15) # Ensure cooldown passes

    # Tick 2: Low swirl, should exit phase
    tick2 = MockTick(
        "tick_002",
        {"ψ_alpha": {"swirl_density": 0.75, "overlap": 0.5}}
    )
    monitor.observe_tick(tick2)
    print(f"Tick 2 (swirl=0.75): Phase state = {monitor.phase_state}")

    time.sleep(0.15)

    # Tick 3: Another high swirl event
    tick3 = MockTick(
        "tick_003",
        {"ψ_beta": {"swirl_density": 0.92, "overlap": 0.85}},
        ["μ_root", "μ_beta_branch"]
    )
    monitor.observe_tick(tick3)
    print(f"Tick 3 (swirl=0.92): Phase state = {monitor.phase_state}\n")

    # --- Exporting Data ---
    print("--- Exported Data Snapshots ---")
    
    # 1. Feedback Packet (lightweight)
    feedback = monitor.export_feedback_packet()
    print("1. Feedback Packet:")
    for k, v in feedback.items():
        print(f"  - {k}: {v}")
    print()

    # 2. Diagnostic Snapshot (detailed)
    snapshot = monitor.get_diagnostic_snapshot()
    print("2. Diagnostic Snapshot:")
    for k, v in snapshot.items():
        print(f"  - {k}: {v}")
    print()

    # 3. Swirl Map (for visualization)
    swirl_map = monitor.render_swirl_map()
    print("3. Rendered Swirl Map (first entry):")
    if swirl_map:
        for k, v in swirl_map[0].items():
            # Format floats for cleaner printing
            val_str = f"{v:.4f}" if isinstance(v, float) else v
            print(f"  - {k}: {val_str}")
    print()

    # 4. Motif Bundle (for lineage)
    motif_bundle = monitor.export_motif_bundle()
    print("4. Exported Motif Bundle (RFC-0007):")
    for field_sig in motif_bundle.get('fields', []):
        print(f"  - Field '{field_sig['field_id']}' first seen at {time.ctime(field_sig['first_seen'])} with lineage: {field_sig['motif_lineage']}")
    print()

    # 5. Geometric Signature (for renderers)
    geo_sig = monitor.export_geometric_signature()
    print("5. Geometric Signature (first field):")
    if geo_sig['fields']:
         print(f"  - Style: {geo_sig['style']}, Type: {geo_sig['type']}")
         print(f"  - Field '{geo_sig['fields'][0]['field']}' vector strength: {geo_sig['fields'][0]['vector_strength']:.4f}")

    print("\n--- Demonstration Complete ---")
	
# End_of_File