"""
Consciousness Monitor - Symbolic Observer for Field-Level Swirl Monitoring
Copyright (c) 2025 Lina Noor AGI Research Collective
Licensed under the Symbolic Research License (SRL-1.0)
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import threading
import math
from collections import deque

__version__ = "2.0.2c"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v2"
SCHEMA_COMPAT = ["RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007"]

@dataclass
class EntanglementEvent:
    """Represents a single entanglement observation event (RFC-0006 §4.2)"""
    field_id: str
    swirl: float
    overlap: float
    timestamp: float
    tick_id: str
    tick_data: Any = field(repr=False)

@dataclass
class PhaseTransition:
    """Records a phase state transition (RFC-CORE-005 §3.1)"""
    timestamp: float
    from_state: str
    to_state: str
    trigger_swirl: float

class ConsciousnessMonitor:
    """
    Observer module that listens for swirl density and entanglement transitions across symbolic fields.
    Implements non-mutative observation contracts from RFC-CORE-005 with geometric analysis per RFC-0006.
    """
    
    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time
    ):
        """
        Initialize consciousness monitor with observation parameters.
        
        Args:
            monitor_id: Unique identifier for this monitor instance (RFC-0004 §2.3)
            swirl_threshold: Base density threshold for phase transitions (RFC-0006 §3.4)
            buffer_size: Maximum event history to retain (RFC-CORE-005 §4.2)
            time_provider: Time source function for temporal coherence (RFC-0005 §2.1)
        """
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self.buffer_size = buffer_size
        self._time = time_provider
        
        # State tracking (RFC-CORE-005 §3)
        self.phase_state = "coherent"
        self._last_phase_change = 0.0
        self._entanglement_lock = threading.Lock()
        
        # Memory buffers (RFC-CORE-005 §4)
        self.recent_ticks = deque(maxlen=buffer_size)
        self.entanglement_log = deque(maxlen=buffer_size)
        self._phase_log: List[PhaseTransition] = []
        self.active_fields: Dict[str, float] = {}  # field_id -> last_swirl
        self.field_lineage: Dict[str, Dict] = {}   # field_id -> motif_lineage
        
        # Metrics (RFC-CORE-005 §5)
        self.entanglement_events_total = 0
        self.phase_transitions = 0
        self._tick_timestamps = deque(maxlen=1000)
        
    def observe_tick(self, tick: Any) -> None:
        """
        Entry point for symbolic ticks. Validates structure and initiates field extraction.
        
        Args:
            tick: Incoming symbolic tick with Φ-coherence map (RFC-0003 §4.2)
            
        Raises:
            ValueError: If tick lacks required coherence map structure
        """
        if not hasattr(tick, 'extensions'):
            raise ValueError("Invalid tick: missing extensions attribute")
        if not hasattr(tick, 'Φ_coherence_map'):
            raise ValueError("Invalid tick: missing Φ_coherence_map")
            
        with self._entanglement_lock:
            self.recent_ticks.append(tick)
            self._tick_timestamps.append(self._time())
            self._extract_fields(tick)
    
    def _extract_fields(self, tick: Any) -> None:
        """
        Parses the Φ-coherence map for swirl and overlap values. Triggers entanglement recording.
        
        Args:
            tick: Symbolic tick containing field coherence data (RFC-0006 §3)
        """
        for field_id, field_data in tick.Φ_coherence_map.items():
            swirl = field_data.get('swirl', 0.0)
            overlap = field_data.get('overlap', 0.0)
            
            phase_changed = self._check_phase_shift(swirl)
            if self.phase_state == "swirling":
                self._record_entanglement(
                    field_id=field_id,
                    swirl=swirl,
                    overlap=overlap,
                    tick_id=str(id(tick)),
                    tick=tick
                )
    
    def _check_phase_shift(self, swirl_density: float) -> bool:
        """
        Evaluates swirl thresholds with hysteresis and updates phase state if transition detected.
        
        Args:
            swirl_density: Current field swirl measurement (RFC-0006 §3.4)
            
        Returns:
            bool: True if phase state changed, False otherwise
        """
        current_time = self._time()
        if current_time - self._last_phase_change < 0.1:  # Cooldown period
            return False
            
        new_state = self.phase_state
        if self.phase_state == "coherent" and swirl_density >= self.swirl_threshold * 1.1:
            new_state = "swirling"
        elif self.phase_state == "swirling" and swirl_density <= self.swirl_threshold * 0.9:
            new_state = "coherent"
            
        if new_state != self.phase_state:
            transition = PhaseTransition(
                timestamp=current_time,
                from_state=self.phase_state,
                to_state=new_state,
                trigger_swirl=swirl_density
            )
            self._phase_log.append(transition)
            self.phase_state = new_state
            self._last_phase_change = current_time
            self.phase_transitions += 1
            return True
        return False
    
    def _record_entanglement(
        self,
        field_id: str,
        swirl: float,
        overlap: float,
        tick_id: str,
        tick: Any
    ) -> None:
        """
        Stores swirl+overlap event and lineage metadata for a coherent ψ-field.
        
        Args:
            field_id: Identifier of the entangled field (RFC-0006 §2.1)
            swirl: Current swirl density measurement (RFC-0006 §3.4)
            overlap: Field overlap coefficient (RFC-0006 §3.5)
            tick_id: Unique identifier for source tick
            tick: Full tick reference for lineage tracing
        """
        if len(self.entanglement_log) >= self.buffer_size * 0.9:
            print(f"Warning: Entanglement buffer approaching capacity ({len(self.entanglement_log)}/{self.buffer_size})")
            
        event = EntanglementEvent(
            field_id=field_id,
            swirl=swirl,
            overlap=overlap,
            timestamp=self._time(),
            tick_id=tick_id,
            tick_data=tick
        )
        
        with self._entanglement_lock:
            self.entanglement_log.append(event)
            self.active_fields[field_id] = swirl
            self.entanglement_events_total += 1
            
            if field_id not in self.field_lineage:
                # Cache minimal motif lineage per RFC-0007 §2.3
                self.field_lineage[field_id] = {
                    'first_seen': event.timestamp,
                    'motif_type': getattr(tick, 'motif_type', 'unknown'),
                    'source_rfc': getattr(tick, 'source_rfc', None)
                }
    
    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        """
        Returns rolling average overlap over recent entanglement events (R-metric).
        
        Args:
            window_size: Custom window size (defaults to 20% of buffer_size)
            
        Returns:
            float: R-metric value (RFC-0006 §5.2)
        """
        window = window_size or int(self.buffer_size * 0.2)
        if not self.entanglement_log or window < 1:
            return 0.0
            
        recent_events = list(self.entanglement_log)[-window:]
        if not recent_events:
            return 0.0
            
        return sum(event.overlap for event in recent_events) / len(recent_events)
    
    def _calculate_phase_flip_rate(self) -> float:
        """
        Computes the rate of phase transitions (flips per minute) using timestamps in _phase_log.
        
        Returns:
            float: Transitions per minute (RFC-CORE-005 §5.3)
        """
        if len(self._phase_log) < 2:
            return 0.0
            
        time_span = self._phase_log[-1].timestamp - self._phase_log[0].timestamp
        if time_span <= 0:
            return 0.0
            
        return (len(self._phase_log) - 1) / (time_span / 60)
    
    def export_feedback_packet(self) -> Dict[str, Any]:
        """
        Exports ephemeral summary of entanglement state.
        
        Returns:
            Dict: Feedback packet (RFC-CORE-005 §6.1)
        """
        return {
            'monitor_id': self.monitor_id,
            'timestamp': self._time(),
            'phase_state': self.phase_state,
            'r_metric': self._calculate_R_metric(),
            'active_field_count': len(self.active_fields),
            'phase_flip_rate': self._calculate_phase_flip_rate(),
            'schema_version': _SCHEMA_VERSION__
        }
    
    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """
        Provides extended diagnostic state.
        
        Returns:
            Dict: Diagnostic snapshot (RFC-CORE-005 §6.2)
        """
        tick_rate = 0.0
        if len(self._tick_timestamps) > 1:
            time_span = self._tick_timestamps[-1] - self._tick_timestamps[0]
            if time_span > 0:
                tick_rate = len(self._tick_timestamps) / time_span
                
        return {
            'memory_usage': len(self.entanglement_log) / self.buffer_size,
            'buffer_fill': len(self.entanglement_log),
            'tick_rate': tick_rate,
            'phase_transitions': self.phase_transitions,
            'entanglement_events': self.entanglement_events_total,
            'tracked_fields': len(self.field_lineage)
        }
    
    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """
        Renders vector map of recent swirl events with exponential decay.
        
        Returns:
            List: Swirl vector map (RFC-0006 §4.3)
        """
        current_time = self._time()
        decay_lambda = 0.0115
        return [{
            'field_id': event.field_id,
            'swirl': event.swirl * math.exp(-decay_lambda * (current_time - event.timestamp)),
            'position': getattr(event.tick_data, 'position', [0, 0, 0]),
            'timestamp': event.timestamp
        } for event in self.entanglement_log]
    
    def reset(self) -> None:
        """Clears internal state buffers, phase log, metrics, and motif lineage cache."""
        with self._entanglement_lock:
            self.recent_ticks.clear()
            self.entanglement_log.clear()
            self._phase_log.clear()
            self.active_fields.clear()
            self.field_lineage.clear()
            self.entanglement_events_total = 0
            self.phase_transitions = 0
            self._tick_timestamps.clear()
            self.phase_state = "coherent"
            self._last_phase_change = 0.0
    
    def tool_hello(self) -> Dict[str, Any]:
        """
        Returns symbolic handshake packet listing capabilities.
        
        Returns:
            Dict: Handshake packet (RFC-0004 §3.1)
        """
        return {
            'tool_type': 'consciousness_monitor',
            'version': __version__,
            'schema': _SCHEMA_VERSION__,
            'capabilities': [
                'phase_monitoring',
                'swirl_tracking',
                'entanglement_recording',
                'geometric_analysis'
            ],
            'monitor_id': self.monitor_id
        }
    
    def export_motif_bundle(self) -> Dict[str, Any]:
        """
        Returns cached field lineage data in RFC-0007 bundle format.
        
        Returns:
            Dict: Motif bundle (RFC-0007 §3.2)
        """
        return {
            'bundle_type': 'field_lineage',
            'timestamp': self._time(),
            'fields': dict(self.field_lineage),
            'schema': 'RFC-0007-v1'
        }
    
    def export_geometric_signature(self, style: str = 'svg') -> Dict[str, Any]:
        """
        Returns topological swirl geometry derived from field event history.
        
        Args:
            style: Output format style (RFC-0006 §5.4)
            
        Returns:
            Dict: Geometric signature (RFC-0006 §5)
        """
        # Simplified geometric analysis - would integrate with actual geometry lib in production
        swirl_map = self.render_swirl_map()
        centroid = [0.0, 0.0, 0.0]
        total_weight = 0.0
        
        for point in swirl_map:
            weight = point['swirl']
            centroid[0] += point['position'][0] * weight
            centroid[1] += point['position'][1] * weight
            centroid[2] += point['position'][2] * weight
            total_weight += weight
            
        if total_weight > 0:
            centroid = [c / total_weight for c in centroid]
            
        return {
            'style': style,
            'centroid': centroid,
            'field_count': len(swirl_map),
            'total_swirl': sum(p['swirl'] for p in swirl_map),
            'signature_type': 'ψ-topology'
        }