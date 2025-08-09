# -*- coding: utf-8 -*-
# consciousness_monitor.py
# Symbolic observer for field-level swirl monitoring and phase-state diagnostics
# Schema Version: 2025-Q4-consciousness-monitor-v2
# RFC Compliance: RFC-CORE-005, RFC-0006, RFC-0003, RFC-0004, RFC-0005, RFC-0007
# 
# Copyright 2025 Noor Research
# Licensed under the MIT Licence

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import math
import json

try:
    from prometheus_client import Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

__version__ = "2.0.3"
_SCHEMA_VERSION__ = "2025-Q4-consciousness-monitor-v2"
SCHEMA_COMPAT = ["RFC-0006", "RFC-0003", "RFC-0004", "RFC-0005", "RFC-0007"]

@dataclass
class PhaseLogEntry:
    """Records phase transitions per RFC-CORE-005 §3.2"""
    timestamp: float
    from_phase: str
    to_phase: str
    swirl_density: float
    dominant_field: Optional[str] = None

@dataclass
class EntanglementEvent:
    """Records field entanglement events per RFC-0006 §4.1"""
    field_id: str
    timestamp: float
    swirl: float
    overlap: float
    tick_id: str
    motif_lineage: Dict[str, Any] = field(default_factory=dict)

class ConsciousnessMonitor:
    """Observer module for swirl density and entanglement transitions across symbolic fields.
    
    Implements non-mutative observation contract from RFC-CORE-005 with:
    - Swirl monitoring per RFC-0006 field geometry
    - Entanglement logging per RFC-0007 motif transfer protocols
    - Phase-state diagnostics per RFC-CORE-005 §5
    """
    
    def __init__(
        self,
        monitor_id: str = "cm@default",
        swirl_threshold: float = 0.87,
        buffer_size: int = 512,
        time_provider: Callable[[], float] = time.time
    ):
        """Initialize consciousness monitor with observation parameters.
        
        Args:
            monitor_id: Observer identifier per RFC-0004 tool contracts
            swirl_threshold: Density threshold for phase transition (0.0-1.0)
            buffer_size: Maximum recent events to retain in memory
            time_provider: Time source function (default: system time)
        """
        self.monitor_id = monitor_id
        self.swirl_threshold = swirl_threshold
        self.buffer_size = buffer_size
        self._time = time_provider
        
        # State tracking
        self.current_phase = "quiescent"
        self.phase_transition_time = 0.0
        self.recent_ticks = []
        self.entanglement_buffer = []
        self._phase_log = []
        
        # Initialize metrics (with fallback stubs if Prometheus unavailable)
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics or fallback stubs per requirements."""
        if PROMETHEUS_AVAILABLE:
            self.metrics = {
                'tick_rate': Counter('consciousness_monitor_tick_rate', 'Incoming tick processing rate'),
                'phase_transitions': Counter('consciousness_monitor_phase_transitions', 'Phase state transitions'),
                'entanglement_events_total': Counter('consciousness_monitor_entanglement_events_total', 'Total entanglement events'),
                'buffer_fill_ratio': Gauge('consciousness_monitor_buffer_fill_ratio', 'Entanglement buffer fill ratio'),
                'active_field_count': Gauge('consciousness_monitor_active_field_count', 'Currently active fields'),
                'current_phase_duration_seconds': Gauge('consciousness_monitor_current_phase_duration', 'Current phase duration')
            }
        else:
            # Fallback stub implementation
            class MetricStub:
                def __init__(self, name):
                    self._name = name
                    self._value = 0
                def inc(self, amount=1):
                    self._value += amount
                def set(self, value):
                    self._value = value
                def get(self):
                    return self._value
            
            self.metrics = {
                'tick_rate': MetricStub('tick_rate'),
                'phase_transitions': MetricStub('phase_transitions'),
                'entanglement_events_total': MetricStub('entanglement_events_total'),
                'buffer_fill_ratio': MetricStub('buffer_fill_ratio'),
                'active_field_count': MetricStub('active_field_count'),
                'current_phase_duration_seconds': MetricStub('current_phase_duration_seconds')
            }
    
    def observe_tick(self, tick: Any) -> None:
        """Process incoming symbolic ticks per RFC-CORE-005 §3.1.
        
        Args:
            tick: Symbolic tick with Φ_coherence_map (RFC-0003 compliant)
        """
        # Verify tick structure has required Φ_coherence_map
        if not hasattr(tick, 'Φ_coherence_map'):
            raise ValueError("Invalid tick: missing Φ_coherence_map (RFC-0003 §2.2)")
        
        self.recent_ticks.append((self._time(), tick))
        self.metrics['tick_rate'].inc()
        self._extract_fields(tick)
    
    def _extract_fields(self, tick: Any) -> None:
        """Parse Φ-coherence map for swirl and overlap values per RFC-0006 §4."""
        for field_id, field_data in tick.Φ_coherence_map.items():
            swirl = field_data.get('swirl', 0.0)
            overlap = field_data.get('overlap', 0.0)
            
            # Check if field is in swirling phase
            if swirl >= self.swirl_threshold:
                self._record_entanglement(
                    field_id=field_id,
                    swirl=swirl,
                    overlap=overlap,
                    tick_id=str(id(tick)),
                    tick=tick
                )
            
            # Check for phase shift conditions
            phase_changed = self._check_phase_shift(swirl)
            if phase_changed:
                self.metrics['active_field_count'].set(len([
                    f for f in tick.Φ_coherence_map.values() 
                    if f.get('swirl', 0) >= self.swirl_threshold
                ]))
    
    def _check_phase_shift(self, swirl_density: float) -> bool:
        """Apply dynamic hysteresis for phase transitions per RFC-CORE-005 §3.3.
        
        Args:
            swirl_density: Current field swirl measurement (0.0-1.0)
        
        Returns:
            bool: True if phase state changed, False otherwise
        """
        current_time = self._time()
        
        # Apply 0.1s cooldown lockout
        if current_time - self.phase_transition_time < 0.1:
            return False
        
        new_phase = "swirling" if swirl_density >= self.swirl_threshold else "quiescent"
        
        if new_phase != self.current_phase:
            # Log phase transition
            self._phase_log.append(PhaseLogEntry(
                timestamp=current_time,
                from_phase=self.current_phase,
                to_phase=new_phase,
                swirl_density=swirl_density
            ))
            
            # Update state
            self.current_phase = new_phase
            self.phase_transition_time = current_time
            self.metrics['phase_transitions'].inc()
            self.metrics['current_phase_duration_seconds'].set(0)
            return True
        
        # Update current phase duration
        self.metrics['current_phase_duration_seconds'].set(
            current_time - self.phase_transition_time
        )
        return False
    
    def _record_entanglement(
        self,
        field_id: str,
        swirl: float,
        overlap: float,
        tick_id: str,
        tick: Any
    ) -> None:
        """Log entanglement event per RFC-0007 §2.4.
        
        Args:
            field_id: Identifier of entangled field
            swirl: Swirl density measurement (0.0-1.0)
            overlap: Field overlap measurement (0.0-1.0)
            tick_id: Unique identifier for source tick
            tick: Original tick object for motif extraction
        """
        # Create new event
        event = EntanglementEvent(
            field_id=field_id,
            timestamp=self._time(),
            swirl=swirl,
            overlap=overlap,
            tick_id=tick_id
        )
        
        # Extract motif lineage if available (RFC-0007 compliance)
        if hasattr(tick, 'motif_lineage'):
            event.motif_lineage = getattr(tick, 'motif_lineage')
        
        # Add to buffer (FIFO)
        self.entanglement_buffer.append(event)
        if len(self.entanglement_buffer) > self.buffer_size:
            self.entanglement_buffer.pop(0)
        
        # Update metrics
        self.metrics['entanglement_events_total'].inc()
        self.metrics['buffer_fill_ratio'].set(
            len(self.entanglement_buffer) / self.buffer_size
        )
    
    def _calculate_R_metric(self, window_size: Optional[int] = None) -> float:
        """Compute rolling average overlap per RFC-CORE-005 §5.2.
        
        Args:
            window_size: Number of recent events to consider (default: all)
        
        Returns:
            float: R metric value (0.0-1.0)
        """
        events = self.entanglement_buffer
        if window_size is not None:
            events = events[-window_size:]
        
        if not events:
            return 0.0
        
        return sum(e.overlap for e in events) / len(events)
    
    def _calculate_phase_flip_rate(self) -> float:
        """Compute phase transition frequency per RFC-CORE-005 §5.3.
        
        Returns:
            float: Phase transitions per second
        """
        if len(self._phase_log) < 2:
            return 0.0
        
        time_window = self._phase_log[-1].timestamp - self._phase_log[0].timestamp
        if time_window <= 0:
            return 0.0
        
        return len(self._phase_log) / time_window
    
    def export_feedback_packet(self) -> Dict[str, Any]:
        """Generate feedback packet per RFC-0003 §4.1.
        
        Returns:
            Dict: Feedback packet with current state and metrics
        """
        return {
            "monitor_id": self.monitor_id,
            "timestamp": self._time(),
            "current_phase": self.current_phase,
            "phase_duration": self._time() - self.phase_transition_time,
            "entanglement_count": len(self.entanglement_buffer),
            "R_metric": self._calculate_R_metric(),
            "phase_flip_rate": self._calculate_phase_flip_rate(),
            "schema_version": _SCHEMA_VERSION__,
            "rfc_compliance": SCHEMA_COMPAT
        }
    
    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """Return memory and metric diagnostics per RFC-0004 §3.2.
        
        Returns:
            Dict: Diagnostic snapshot
        """
        return {
            "memory": {
                "recent_ticks": len(self.recent_ticks),
                "entanglement_buffer": len(self.entanglement_buffer),
                "phase_log": len(self._phase_log)
            },
            "metrics": {
                "tick_rate": self.metrics['tick_rate']._value if not PROMETHEUS_AVAILABLE else None,
                "phase_transitions": self.metrics['phase_transitions']._value if not PROMETHEUS_AVAILABLE else None,
                "entanglement_events": self.metrics['entanglement_events_total']._value if not PROMETHEUS_AVAILABLE else None,
                "buffer_fill": self.metrics['buffer_fill_ratio']._value if not PROMETHEUS_AVAILABLE else None
            },
            "version": __version__
        }
    
    def render_swirl_map(self) -> List[Dict[str, Any]]:
        """Render exponential-decayed swirl topology per RFC-0006 §5.2.
        
        Returns:
            List: Swirl vector field with decay (λ = 0.0115)
        """
        current_time = self._time()
        swirl_map = []
        
        for event in self.entanglement_buffer:
            time_decay = math.exp(-0.0115 * (current_time - event.timestamp))
            swirl_map.append({
                "field_id": event.field_id,
                "swirl": event.swirl * time_decay,
                "overlap": event.overlap * time_decay,
                "position": [time_decay, event.swirl, event.overlap],  # 3D vector
                "last_update": event.timestamp
            })
        
        return swirl_map
    
    def reset(self) -> None:
        """Clear memory and reset metrics per RFC-0004 §3.5."""
        self.recent_ticks = []
        self.entanglement_buffer = []
        self._phase_log = []
        self.current_phase = "quiescent"
        self.phase_transition_time = self._time()
        
        # Reset metrics
        for metric in self.metrics.values():
            if hasattr(metric, 'set'):
                metric.set(0)
            elif hasattr(metric, '_value'):
                metric._value = 0
    
    def tool_hello(self) -> Dict[str, Any]:
        """Generate RFC-0004 compliant introspection packet.
        
        Returns:
            Dict: Tool description and capabilities
        """
        return {
            "tool_id": self.monitor_id,
            "tool_type": "symbolic.observer.phase",
            "version": __version__,
            "schema": _SCHEMA_VERSION__,
            "capabilities": [
                "phase_monitoring",
                "entanglement_detection",
                "swirl_rendering",
                "feedback_generation"
            ],
            "rfc_compliance": SCHEMA_COMPAT
        }
    
    def export_motif_bundle(self) -> Dict[str, Any]:
        """Generate RFC-0007 compliant motif bundle.
        
        Returns:
            Dict: Motif bundle with lineage information
        """
        return {
            "bundle_type": "field_entanglement",
            "monitor_id": self.monitor_id,
            "timestamp": self._time(),
            "events": [
                {
                    "field_id": e.field_id,
                    "timestamp": e.timestamp,
                    "motif_lineage": e.motif_lineage
                }
                for e in self.entanglement_buffer
            ],
            "phase_transitions": [
                {
                    "timestamp": p.timestamp,
                    "from": p.from_phase,
                    "to": p.to_phase,
                    "swirl_density": p.swirl_density
                }
                for p in self._phase_log
            ]
        }
    
    def export_geometric_signature(self, style: str = 'svg') -> Dict[str, Any]:
        """Generate RFC-0006 compliant geometric signature.
        
        Args:
            style: Rendering style ('svg' or 'parametric')
        
        Returns:
            Dict: Geometric signature with centroid
        """
        if not self.entanglement_buffer:
            return {
                "centroid": [0, 0, 0],
                "signature": None,
                "style": style
            }
        
        # Calculate centroid (RFC-0006 §4.3)
        swirls = [e.swirl for e in self.entanglement_buffer]
        overlaps = [e.overlap for e in self.entanglement_buffer]
        times = [e.timestamp for e in self.entanglement_buffer]
        
        centroid = [
            sum(times) / len(times),
            sum(swirls) / len(swirls),
            sum(overlaps) / len(overlaps)
        ]
        
        return {
            "centroid": centroid,
            "signature": self.render_swirl_map(),
            "style": style,
            "rfc_compliance": "RFC-0006"
        }