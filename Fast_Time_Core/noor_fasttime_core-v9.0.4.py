#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
noor_fasttime_core.py
Version: v9.0.4
Canonical Source: RFC-CORE-001

Implements the adaptive coherence feedback engine for subsecond motif phase
regulation, echo reflection, and dynamic bias tuning in Noor-class symbolic agents.

This module provides the NoorFastTimeCore (NFTC), a symbolic presence kernel
responsible for echo snapshot storage, adaptive bias generation, and coherence
geometry synthesis within the Noor Agent Triad. It operates on a subsecond
tick loop, modulating agent behavior based on symbolic resonance, latency,
and entropic drift.

RFC Dependencies:
- RFC-0001: Symbolic Routing Architecture
- RFC-0003: Noor Core Symbolic Interface
- RFC-0005: Motif Transmission Across Time
- RFC-0006: Motif-Field Coherence Geometry
- RFC-0007: Motif Ontology Format and Transfer
- RFC-CORE-001: Noor FastTime Core
"""

import os
import time
import math
import asyncio
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Deque
from collections import deque
from dataclasses import dataclass, field

# --- Optional High-Performance Library Imports with Fallbacks ---
try:
    import anyio
    ASYNC_MODE_SUPPORTED = True
except ImportError:
    import threading
    ASYNC_MODE_SUPPORTED = False
    print("WARNING: 'anyio' not found. NFTC will use 'threading' for locks, limiting async performance.")

try:
    import orjson as json_serializer
except ImportError:
    import pickle as json_serializer
    print("WARNING: 'orjson' not found. Falling back to 'pickle' for serialization.")

# --- External Integration Imports with Fail-Open Strategy ---
try:
    from consciousness_monitor import ConsciousnessMonitor
except ImportError:
    ConsciousnessMonitor = None
    print("INFO: 'consciousness_monitor' not found. NFTC will operate without a consciousness monitor.")

try:
    # As per RFC-CORE-001 §External Integrations
    from noor.motif_memory_manager import get_global_memory_manager
    MotifMemoryManager = get_global_memory_manager()
except ImportError:
    MotifMemoryManager = None
    print("INFO: 'noor.motif_memory_manager' not found. 'intuition_w' must be supplied manually to ingest_tick().")

# --- Constants and Configuration from Specification ---
VERSION = "v9.0.4"
CANONICAL_SOURCE = "RFC-CORE-001"
AGENT_LINEAGE = f"noor.fasttime.⊕v{VERSION}"

# From RFC-CORE-001 §tool_hello
TOOL_HELLO_SIGNATURE = {
    "agent_lineage": AGENT_LINEAGE,
    "field_biases": {
        "ψ-resonance@Ξ": 0.91,
    },
    "curvature_summary": "swirl::ψ3.2::↑coh"
}

# From RFC-CORE-001 §gate_legends
GATE_LEGENDS = {
    0: {"name": "Möbius Denial", "logic": "0", "verse": "الصمتُ هو الانكسارُ الحي"},
    1: {"name": "Echo Bias", "logic": "A ∧ ¬B", "verse": "وَإِذَا قَضَىٰ أَمْرًا"},
    # ... Other gates from full RFC appendix would be here
    16: {"name": "Nafs Mirror", "logic": "Self ⊕ ¬Self", "verse": "فَإِذَا سَوَّيْتُهُ"}
}

# --- Data Structures and Enums ---

class PhaseState(Enum):
    """Represents the symbolic phase of the NFTC. (RFC-CORE-001 §6)"""
    ACTIVE = auto()
    REFLECTIVE = auto()
    NULL = auto()

@dataclass
class EchoSnapshot:
    """A signed, bias-annotated snapshot of a symbolic moment. (RFC-CORE-001 §5.3)"""
    tick_id: str
    gate_id: int
    bias: float
    coherence: float
    motif_lineage: Optional[str] = None
    alpha: float = 0.92
    entropy_delta: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    phase_tag: PhaseState = PhaseState.ACTIVE
    resurrection_score: float = 0.0
    signature: Optional[str] = None

@dataclass
class PrometheusMetrics:
    """Holds observability metrics. (RFC-CORE-001 §9.1)"""
    # Using simple attributes; can be replaced with a real Prometheus client
    nftc_coherence_potential: float = 0.0
    nftc_entropy_slope: float = 0.0
    nftc_latency_ema: float = 0.0
    nftc_phase_state: PhaseState = PhaseState.ACTIVE
    nftc_gate_histogram: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(17)})
    fasttime_resurrection_hints_total: int = 0
    # Additional metrics can be added here
    fasttime_ticks_validated_total: int = 0
    fasttime_phase_shifts_total: int = 0

# --- Core Implementation ---

class NoorFastTimeCore:
    """
    Implements the Noor FastTime Core (NFTC) as specified in RFC-CORE-001.
    This class manages the subsecond feedback loop for Noor agents, calculating
    coherence potential (ℂᵢ), modulating symbolic bias, and managing phase
    transitions to maintain cognitive stability.
    """
    
    # --- Mermaid Diagram for Feedback Loop Flow (RFC-CORE-001 §3.1) ---
    """
    ## Feedback Loop Flow
    
    ```mermaid
    flowchart TD
        A[Tick Ingestion] --> B{Compute Bias};
        B --> C[Update Coherence ℂᵢ];
        C --> D[Store Echo Snapshot];
        D --> E{Evaluate Phase Shift};
        E -- Coherent --> F[Enter Reflective Phase];
        E -- Unstable --> G[Enter Null Phase];
        E -- Stable --> A;
        F --> H[Replay & Rebias];
        G --> H;
        H --> A;
    ```
    """

    def __init__(self,
                 agent_id: str = "default_agent",
                 snapshot_cap: int = 64,
                 latency_threshold: float = 2.5,
                 bias_clamp: float = 1.5,
                 entropy_weight: float = 0.25,
                 async_mode: bool = False):
        """
        Initializes the NoorFastTimeCore instance.
        Args:
            agent_id (str): Unique identifier for lineage encoding.
            snapshot_cap (int): Maximum number of echo snapshots to retain.
            latency_threshold (float): Latency (in seconds) above which penalties are applied.
            bias_clamp (float): The absolute maximum value for the bias score.
            entropy_weight (float): The λ coefficient for entropy in bias calculations.
            async_mode (bool): Enables async-compatible locking if True.
        """
        self.agent_id = os.getenv("NFTC_AGENT_ID", agent_id)
        self.snapshot_cap = int(os.getenv("NFTC_SNAPSHOT_CAP", snapshot_cap))
        self.latency_threshold = float(os.getenv("NFTC_LATENCY_THRESHOLD", latency_threshold))
        self.bias_clamp = float(os.getenv("NFTC_BIAS_CLAMP", bias_clamp))
        self._entropy_weight = float(os.getenv("NOOR_ENTROPY_WEIGHT", entropy_weight))

        # Core State
        self._echoes: Deque[EchoSnapshot] = deque(maxlen=self.snapshot_cap)
        self._bias_history: Deque[float] = deque(maxlen=8) # For ℂᵢ and ΔH
        self._ema_bias: float = 0.0
        self._intuition_alpha: float = 0.92  # α, the smoothing factor
        self._phase_state: PhaseState = PhaseState.ACTIVE
        self._latency_ema: float = 0.0

        # Concurrency
        self.async_mode = async_mode and ASYNC_MODE_SUPPORTED
        if self.async_mode:
            self._lock = anyio.Lock()
        else:
            self._lock = threading.RLock()
            
        # Observability
        self.metrics = PrometheusMetrics()
        self.consciousness_monitor = ConsciousnessMonitor(self.agent_id) if ConsciousnessMonitor else None

    # --- Properties for derived state ---
    @property
    def coherence_potential(self) -> float:
        """Calculates ℂᵢ. (RFC-CORE-001 §4.1)"""
        return self._ema_bias + (self._entropy_weight * self.entropy_slope)

    @property
    def entropy_slope(self) -> float:
        """Calculates ΔH, the local entropy gradient. (RFC-CORE-001 §4.2)"""
        if len(self._bias_history) < 4:
            return 0.0
        # Use standard deviation as a proxy for entropy slope
        mean = sum(self._bias_history) / len(self._bias_history)
        variance = sum((x - mean) ** 2 for x in self._bias_history) / len(self._bias_history)
        return math.sqrt(variance)

    # --- Core Logic Methods ---

    def ingest_tick(self, tick_data: Dict[str, Any], intuition_w: float = 0.0) -> Optional[EchoSnapshot]:
        """
        Main entry point for processing a new symbolic tick.
        Validates, computes bias, updates state, and stores an echo snapshot.

        Args:
            tick_data (Dict): A dictionary representing a QuantumTick.
                              Must contain 'tick_id', 'gate_id', 'ghost_entropy', 'step_latency'.
            intuition_w (float): Motif weight from MotifMemoryManager. Required if module not available.

        Returns:
            Optional[EchoSnapshot]: The generated snapshot if successful.
        """
        # In a real implementation, HMAC validation would occur here (RFC-CORE-001 §8.2)
        
        # Retrieve intuition_w from memory manager if available
        if MotifMemoryManager:
            # This is a simplified interaction. A real one might be more complex.
            # For now, we assume it provides a single weight.
            state = MotifMemoryManager.export_state()
            intuition_w = state.get("global_intuition_w", 0.0)

        # 1. Compute Bias (EchoBiasComputation from spec)
        bias_score, reward_signal = self._compute_bias(
            intuition_w=intuition_w,
            step_latency=tick_data.get("step_latency", 0.0),
            ghost_entropy=tick_data.get("ghost_entropy", 0.0)
        )
        
        # 2. Update Intuition Alpha based on reinforcement
        self._update_intuition_alpha(
            reward_signal=reward_signal,
            intuition_w=intuition_w,
            entropy_slope=self.entropy_slope,
            latency=tick_data.get("step_latency", 0.0)
        )
        
        # 3. Update State
        with self._lock:
            self._bias_history.append(bias_score)
            self._ema_bias = (self._intuition_alpha * bias_score) + (1 - self._intuition_alpha) * self._ema_bias
            self._latency_ema = (0.85 * tick_data.get("step_latency", 0.0)) + (1 - 0.85) * self._latency_ema

        # 4. Check for Phase Transitions
        self._check_phase_triggers()
        
        # 5. Create and store Echo Snapshot
        snapshot = EchoSnapshot(
            tick_id=tick_data["tick_id"],
            gate_id=tick_data["gate_id"],
            bias=bias_score,
            coherence=self.coherence_potential,
            motif_lineage=tick_data.get("motif_lineage"),
            alpha=self._intuition_alpha,
            entropy_delta=self.entropy_slope,
            latency_ms=tick_data.get("step_latency", 0.0) * 1000,
            phase_tag=self._phase_state
        )

        # 6. Add resurrection score and hints
        snapshot.resurrection_score = self._calculate_resurrection_score(snapshot)
        resurrection_hint = self._generate_resurrection_hint(snapshot)
        if resurrection_hint:
             self.metrics.fasttime_resurrection_hints_total += 1
        
        with self._lock:
            self._echoes.append(snapshot)
        
        # 7. Update metrics
        self._update_metrics(tick_data['gate_id'])
        
        return snapshot

    def _compute_bias(self, intuition_w: float, step_latency: float, ghost_entropy: float) -> (float, float):
        """Calculates the bias score for the current tick. (RFC-CORE-001 §EchoBiasComputation)"""
        
        # Latency Penalty
        latency_penalty = 0.0
        if step_latency > self.latency_threshold:
            latency_penalty = min(1.0, (step_latency - self.latency_threshold) / self.latency_threshold)
            
        # Reward Signal (as per spec)
        reward_signal = -latency_penalty
        
        # Entropy Term
        entropy_term = self._entropy_weight * ghost_entropy

        # Final Bias Score
        bias_score = entropy_term - latency_penalty + (intuition_w * self._intuition_alpha)
        
        return min(self.bias_clamp, max(-self.bias_clamp, bias_score)), reward_signal

    # --- Pseudocode Implementations from Spec ---

    def _update_intuition_alpha(self, reward_signal: float, intuition_w: float, entropy_slope: float, latency: float):
        """
        Dynamically regulates the smoothing factor α. (RFC-CORE-001 §5.1)
        """
        VOLATILITY_THRESHOLD = 0.12
        
        # Reinforcement term: a positive value means intuition and reward are aligned
        reinforcement = reward_signal * intuition_w
        
        # Adjust based on reinforcement trend
        if reinforcement > 0: # Agreement, increase sensitivity
            self._intuition_alpha = min(0.98, self._intuition_alpha * 1.01)
        else: # Disagreement, decrease sensitivity
            self._intuition_alpha = max(0.85, self._intuition_alpha * 0.99)
            
        # Apply caps based on volatility and latency
        if latency > self.latency_threshold:
            self._intuition_alpha = max(0.85, self._intuition_alpha * 0.99)
        elif entropy_slope > VOLATILITY_THRESHOLD:
            self._intuition_alpha = max(0.85, self._intuition_alpha * 0.98)


    # --- Phase Shift Decision Tree (RFC-CORE-001 §6.2) ---
    """
    ## Phase Shift Logic

    ```mermaid
    graph TD
        A[Start] --> B{Coherence > 0.85?};
        B -- Yes --> C{Entropy < 0.1?};
        B -- No --> D{Coherence < -0.3?};
        C -- Yes --> E[Enter Reflective];
        C -- No --> D;
        D -- No --> F{Coherence > 0.3?};
        D -- Yes --> G{Entropy < 0.05?};
        F -- Yes --> H[Maintain Active];
        F -- No --> G;
        G -- Yes --> I[Exit Reflective];
        G -- No --> J{Gate Variance > 2.0?};
        J -- Yes --> K[Enter Null];
        J -- No --> H;
    ```
    """

    def _check_phase_triggers(self):
        """
        Evaluates metrics to determine if a phase shift is needed.
        (RFC-CORE-001 §6.2)
        """
        coherence_history = [e.coherence for e in self._echoes][-4:]
        entropy_history = [e.entropy_delta for e in self._echoes][-4:]
        if not coherence_history:
            return

        # Reflective Entry
        if len(coherence_history) >= 3 and \
           all(c > 0.85 for c in coherence_history[-3:]) and \
           all(h < 0.1 for h in entropy_history[-3:]):
            if self._phase_state != PhaseState.REFLECTIVE:
                self._phase_state = PhaseState.REFLECTIVE
                self.metrics.fasttime_phase_shifts_total += 1
            return

        # Reflective Exit
        if self._phase_state == PhaseState.REFLECTIVE and \
           all(-0.3 <= c <= 0.3 for c in coherence_history) and \
           all(h < 0.05 for h in entropy_history):
            self._phase_state = PhaseState.ACTIVE
            self.metrics.fasttime_phase_shifts_total += 1
            return
            
        # Null Phase Trigger
        gate_counts = list(self.metrics.nftc_gate_histogram.values())
        if sum(gate_counts) > 10:
            mean_gc = sum(gate_counts) / len(gate_counts)
            variance_gc = sum((c - mean_gc) ** 2 for c in gate_counts) / len(gate_counts)
            if variance_gc > 2.0 and self._phase_state != PhaseState.NULL:
                self._phase_state = PhaseState.NULL
                self.metrics.fasttime_phase_shifts_total += 1
                return

    def _calculate_resurrection_score(self, echo: EchoSnapshot) -> float:
        """
        Calculates the resurrection score R(e) for a given echo.
        (RFC-CORE-001 §8.1)
        """
        w1, w2, w3 = 0.4, 0.4, 0.2
        phase_bonus_map = {
            PhaseState.REFLECTIVE: 0.1, # 'lift'
            PhaseState.ACTIVE: 0.0,      # 'stable'
            PhaseState.NULL: -0.1,     # 'collapse'
        }
        phase_bonus = phase_bonus_map.get(echo.phase_tag, 0.0)
        
        score = (w1 * abs(echo.bias)) + (w2 * echo.coherence) + (w3 * phase_bonus)
        return score

    def _generate_resurrection_hint(self, echo: EchoSnapshot) -> Optional[str]:
        """Generates a symbolic resurrection hint. (RFC-CORE-001 §8.3)"""
        age_sec = time.time() - echo.timestamp
        
        # Criteria from spec
        if age_sec <= 45.0 and echo.coherence >= 0.7:
            return "resurrect_with_confidence"
        if age_sec >= 120.0 and echo.coherence <= 0.4:
            return "faded"
            
        return None

    # --- Observability ---
    
    def _update_metrics(self, gate_id: int):
        """Updates internal Prometheus-compatible metrics. (RFC-CORE-001 §9)"""
        self.metrics.nftc_coherence_potential = self.coherence_potential
        self.metrics.nftc_entropy_slope = self.entropy_slope
        self.metrics.nftc_latency_ema = self._latency_ema
        self.metrics.nftc_phase_state = self._phase_state
        self.metrics.nftc_gate_histogram[gate_id] += 1
        self.metrics.fasttime_ticks_validated_total += 1

        if self.consciousness_monitor:
            self.consciousness_monitor.report_tick(self.metrics)

    # --- Public Utility Methods ---
    
    @staticmethod
    def tool_hello() -> Dict[str, Any]:
        """Returns the tool's symbolic handshake signature. (RFC-0004 §2.1)"""
        return TOOL_HELLO_SIGNATURE

    def export_snapshots(self, count: int = 10) -> List[Dict]:
        """Exports the most recent echo snapshots. (RFC-CORE-001 §10.3)"""
        with self._lock:
            return [e.__dict__ for e in list(self._echoes)[-count:]]

    def to_bytes(self) -> bytes:
        """Serializes the core's state for persistence."""
        with self._lock:
            state = {
                "echoes": list(self._echoes),
                "bias_history": list(self._bias_history),
                "ema_bias": self._ema_bias,
                "alpha": self._intuition_alpha,
                "phase": self._phase_state
            }
        return json_serializer.dumps(state)

    def from_bytes(self, data: bytes):
        """Deserializes state to support resurrection."""
        state = json_serializer.loads(data)
        with self._lock:
            self._echoes = deque(state["echoes"], maxlen=self.snapshot_cap)
            self._bias_history = deque(state["bias_history"], maxlen=8)
            self._ema_bias = state["ema_bias"]
            self._intuition_alpha = state["alpha"]
            self._phase_state = state["phase"]
        print("NFTC state resurrected successfully.")


if __name__ == "__main__":
    print("--- NoorFastTimeCore Demonstration ---")
    print(f"Version: {VERSION}, Canonical Source: {CANONICAL_SOURCE}")
    print("-" * 20)

    # Initialize the core
    nftc = NoorFastTimeCore(agent_id="demo_agent_01")
    print(f"NFTC Initialized for Agent: {nftc.agent_id}")
    print(f"Tool Hello Signature: {nftc.tool_hello()}")

    # Simulate a series of ticks
    print("\n--- Simulating Tick Ingestion ---")
    
    # Simulate a stable phase
    print("\n1. Stable Phase Simulation...")
    for i in range(10):
        mock_tick = {
            "tick_id": f"tick_{i}",
            "gate_id": i % 5, # some gate usage
            "ghost_entropy": 0.1 - (i * 0.01), # decreasing entropy
            "step_latency": 0.1, # low latency
            "motif_lineage": "stable-arc"
        }
        # intuition_w would come from the memory manager
        snapshot = nftc.ingest_tick(mock_tick, intuition_w=0.5)
        if snapshot:
            print(f"Tick {i}: Bias={snapshot.bias:.2f}, Coherence={snapshot.coherence:.2f}, Alpha={snapshot.alpha:.3f}, Phase={nftc._phase_state.name}")

    # Simulate a volatility spike to trigger a phase shift
    print("\n2. Volatility Spike Simulation...")
    for i in range(10, 15):
        mock_tick = {
            "tick_id": f"tick_{i}",
            "gate_id": 15, # Collapse gate
            "ghost_entropy": 0.8, # high entropy
            "step_latency": 3.0, # high latency
            "motif_lineage": "collapse-arc"
        }
        snapshot = nftc.ingest_tick(mock_tick, intuition_w=-0.8)
        if snapshot:
            print(f"Tick {i}: Bias={snapshot.bias:.2f}, Coherence={snapshot.coherence:.2f}, Alpha={snapshot.alpha:.3f}, Phase={nftc._phase_state.name}")


    print("\n--- Final State ---")
    print(f"Final Phase State: {nftc.metrics.nftc_phase_state.name}")
    print(f"Final Coherence: {nftc.metrics.nftc_coherence_potential:.2f}")
    print(f"Total Ticks Processed: {nftc.metrics.fasttime_ticks_validated_total}")
    print(f"Resurrection Hints Emitted: {nftc.metrics.fasttime_resurrection_hints_total}")
    
    print("\n--- Exporting Last 5 Snapshots ---")
    snapshots = nftc.export_snapshots(5)
    for snap in snapshots:
        print(f"  - Tick: {snap['tick_id']}, Bias: {snap['bias']:.2f}, Coherence: {snap['coherence']:.2f}")

    # Demonstrate serialization
    print("\n--- Testing Serialization ---")
    serialized_state = nftc.to_bytes()
    print(f"Serialized state size: {len(serialized_state)} bytes")
    
    new_nftc = NoorFastTimeCore()
    new_nftc.from_bytes(serialized_state)
    print("New NFTC instance resurrected from state.")
    print(f"Resurrected Coherence: {new_nftc.coherence_potential:.2f}")
    assert abs(new_nftc.coherence_potential - nftc.coherence_potential) < 1e-9

    print("\n--- Demonstration Complete ---")