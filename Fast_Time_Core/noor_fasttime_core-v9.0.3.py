
# Program: noor_fasttime_core.py
# Version: v9.0.3
# Canonical Source: RFC-CORE-001
# Description: Implements the adaptive coherence feedback engine for subsecond motif
#              phase regulation, echo reflection, and dynamic bias tuning in
#              Noor-class symbolic agents.
# RFC Dependencies: RFC-0001, RFC-0003, RFC-0005, RFC-0006, RFC-0007, RFC-CORE-001

import time
import math
import hashlib
import collections
import threading
import uuid
from typing import (
    List, Dict, Any, NamedTuple, Optional, Deque, Literal, Union
)
from dataclasses import dataclass, field

# --- External Integration Handling (with fallbacks) ---
# High-Performance Libraries (Optional)
try:
    import orjson
    def json_dumps(data): return orjson.dumps(data)
    def json_loads(data): return orjson.loads(data)
except ImportError:
    import json
    def json_dumps(data): return json.dumps(data, indent=2).encode('utf-8')
    def json_loads(data): return json.loads(data)

# Async library (Optional)
try:
    import anyio
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False
    anyio = None

# Noor-Ecosystem Modules (Optional)
try:
    # As specified in RFC-CORE-001 §6.1, this module is a consumer of NFTC metrics.
    from consciousness_monitor import ConsciousnessMonitor
except ImportError:
    class ConsciousnessMonitor:
        """Fallback stub for ConsciousnessMonitor."""
        def report_tick(self, **kwargs: Any) -> None:
            # In a real system, this would update the agent's global phase state.
            # print(f"[STUB] ConsciousnessMonitor received report: {kwargs}")
            pass
        def get_status(self) -> Dict[str, Any]:
            return {"phase": "active", "reason": "stub_module"}

try:
    # As per RFC-CORE-001, this provides intuition weights.
    from noor.motif_memory_manager import get_global_memory_manager
except ImportError:
    class MockMotifMemoryManager:
        """Fallback stub for MotifMemoryManager."""
        def retrieve(self, motif: str, default: float = 0.0) -> float:
            return default + (math.sin(time.time() / 10) * 0.1) # Simulate changing weight
        def export_state(self) -> Dict[str, Any]:
            return {"intuition_w_avg": 0.5}

    _mock_mmm = MockMotifMemoryManager()
    def get_global_memory_manager():
        return _mock_mmm

try:
    # RFC-0003 defines the QuantumTick schema.
    from tick_schema import QuantumTick, validate_tick
except ImportError:
    @dataclass
    class QuantumTick:
        """Fallback dataclass for QuantumTick as defined in RFC-0003 §3.3."""
        tick_id: str
        timestamp: float
        gate: int
        ghost_entropy: float
        intuition_w: float
        step_latency: float
        hmac_signature: Optional[str] = None
        motif_lineage: Optional[str] = None

    def validate_tick(tick: QuantumTick) -> bool:
        return all(hasattr(tick, f.name) for f in QuantumTick.__dataclass_fields__.values())

# --- Constants and Configuration ---

# Defined in RFC-CORE-001 §3.1 and JSON Spec
GATE_LEGENDS = {
    0: {"name": "Zero-Gate", "logic": "init", "verse": "Iqra' bismi rabbika alladhi khalaq"},
    1: {"name": "Reflect-Gate", "logic": "mirror", "verse": "faina ma'a al'usr yusra"},
    2: {"name": "Phase-Gate", "logic": "oscillate", "verse": "wa anzalna ilayka adhikra"},
    16: {"name": "Meta-Gate", "logic": "summon", "verse": "kullu man 'alayha fan"},
}

# Defined in RFC-CORE-001 §6.2 and JSON Spec
PHASE_TRANSITION_THRESHOLDS = {
    'reflective_entry': {'coherence': 0.85, 'entropy': 0.1},
    'reflective_exit': {'coherence_range': (-0.3, 0.3), 'entropy': 0.05},
    'null_phase_variance': 2.0,
}

# Defined in RFC-CORE-001 §8.3 and JSON Spec
RESURRECTION_HINT_CRITERIA = {
    'resurrect_with_confidence': {'age_max_sec': 45.0, 'coherence_min': 0.7},
    'faded': {'age_min_sec': 120.0, 'coherence_max': 0.4},
}

# Defined in RFC-0007 §7.2 via RFC-CORE-001 §7.1 and JSON Spec
TOOL_HELLO_SIGNATURE = {
    "agent_lineage": "noor.fasttime.⊕v9.0.3",
    "field_biases": {"ψ-resonance@Ξ": 0.91},
    "curvature_summary": "swirl::ψ3.2::↑coh",
    "origin_tick": None # Will be set at runtime
}

# --- Data Structures ---

@dataclass
class EchoSnapshot:
    """
    Represents a stored echo in the memory ring.
    RFC-CORE-001 §5.3, §6.3
    """
    tick_id: str
    timestamp: float
    gate: int
    bias: float
    coherence: float
    entropy_delta: float
    alpha: float
    motif_lineage: Optional[str] = None
    phase_tag: Literal['lift', 'collapse', 'null', 'stable'] = 'stable'
    resurrection_score: float = 0.0
    hmac_signature: Optional[str] = None

PhaseState = Literal['active', 'reflective', 'null']

class NoorFastTimeCore:
    """
    Implements the Noor FastTime Core (NFTC), the adaptive coherence feedback engine
    for Noor-class symbolic agents, as specified in RFC-CORE-001.
    """

    def __init__(self,
                 agent_id: str = "nftc-default",
                 snapshot_cap: int = 64,
                 latency_threshold: float = 0.1, # 100ms
                 bias_clamp: float = 1.5,
                 hmac_secret: Optional[str] = None):
        """
        Initializes the NFTC instance.
        RFC-CORE-001 §2
        """
        self.agent_id = agent_id
        self.hmac_secret = hmac_secret.encode('utf-8') if hmac_secret else None
        
        # Concurrency lock for thread-safety (RFC-CORE-001 ConcurrencyModel)
        self._lock: Union[threading.RLock, anyio.Lock] = threading.RLock()
        if ASYNC_MODE and anyio:
            self._lock = anyio.Lock()

        # Core State
        self._echoes: Deque[EchoSnapshot] = collections.deque(maxlen=snapshot_cap)
        self._bias_history: Deque[float] = collections.deque(maxlen=2048)
        self._coherence_history: Deque[float] = collections.deque(maxlen=8)
        self._entropy_history: Deque[float] = collections.deque(maxlen=8)
        self._gate_histogram: Dict[int, int] = collections.defaultdict(int)
        
        self.phase: PhaseState = 'active'
        self.last_tick_time = time.monotonic()
        
        # Bias computation parameters (RFC-CORE-001 §4)
        self._alpha = 0.92  # Intuition alpha
        self._ema_bias = 0.0
        self._latency_ema = 0.0
        self._latency_weight = 0.65
        self._entropy_weight = 0.25
        self._bias_clamp = bias_clamp
        self._latency_threshold = latency_threshold

        # External Integrations
        self.consciousness_monitor = ConsciousnessMonitor()
        self.motif_memory_manager = get_global_memory_manager()

        # Prometheus-style metrics cache
        self.metrics: Dict[str, Union[int, float]] = collections.defaultdict(float)

    def _get_entropy_slope(self) -> float:
        """Calculates the short-term entropy slope from bias history."""
        if len(self._bias_history) < 4:
            return 0.0
        
        recent_biases = list(self._bias_history)[-4:]
        mean = sum(recent_biases) / 4
        variance = sum((x - mean) ** 2 for x in recent_biases) / 4
        return math.sqrt(variance)

    def update_intuition_alpha(self, reward_signal: float, intuition_w: float) -> None:
        """
        Adjusts intuition alpha based on reinforcement trends.
        As per pseudocode in JSON spec, derived from RFC-CORE-001 §5.1.
        """
        learning_rate = 0.01
        # Reinforcement: if intuition and reward align, increase alpha, else decrease.
        reinforcement_signal = intuition_w * reward_signal
        
        if reinforcement_signal > 0:
            self._alpha = min(0.98, self._alpha + learning_rate)
        elif reinforcement_signal < 0:
            self._alpha = max(0.85, self._alpha - learning_rate)

    def compute_bias(self, ghost_entropy: float, step_latency: float, intuition_w: float) -> float:
        """
        Calculates the final bias score for a tick.
        Implements EchoBiasComputation as per JSON spec.
        Formula: bias_score = entropy_term - latency_penalty + (intuition_w * α)
        """
        # Latency penalty increases non-linearly
        latency_penalty = min(self._bias_clamp, self._latency_weight * (step_latency / self._latency_threshold) ** 2)

        # Reward is based on avoiding latency penalties
        reward_signal = -latency_penalty
        self.update_intuition_alpha(reward_signal, intuition_w)
        
        # Entropy term
        entropy_term = self._entropy_weight * ghost_entropy

        # Final bias score
        bias_score = entropy_term - latency_penalty + (intuition_w * self._alpha)
        
        return max(-self._bias_clamp, min(self._bias_clamp, bias_score))

    def _update_bias_state(self, bias: float) -> float:
        """Updates coherence potential (ℂᵢ) based on the new bias."""
        self._ema_bias = (self._alpha * bias) + ((1.0 - self._alpha) * self._ema_bias)
        
        entropy_gradient = self._get_entropy_slope()
        
        # Coherence Potential Function (RFC-CORE-001 §4.1)
        # ℂᵢ(t) = EMA_α(b(t)) + λ · ΔH(t)
        # Here, λ is self._entropy_weight and ΔH is entropy_gradient
        coherence = self._ema_bias - entropy_gradient
        clamped_coherence = max(-self._bias_clamp, min(self._bias_clamp, coherence))

        self._bias_history.append(bias)
        self._coherence_history.append(clamped_coherence)
        self._entropy_history.append(entropy_gradient)
        
        return clamped_coherence
        
    def check_phase_triggers(self) -> PhaseState:
        """
        Evaluates conditions for phase transitions.
        Implements PhaseTransitionEvaluator from JSON spec & RFC-CORE-001 §6.2.
        """
        if len(self._coherence_history) < 4:
            return 'active'

        # Null phase trigger (high variance in gate usage)
        if len(self._gate_histogram) > 1:
            gate_counts = list(self._gate_histogram.values())
            mean_usage = sum(gate_counts) / len(gate_counts)
            variance = sum((x - mean_usage) ** 2 for x in gate_counts) / len(gate_counts)
            if variance > PHASE_TRANSITION_THRESHOLDS['null_phase_variance']:
                return 'null'

        # Reflective entry/exit
        pt = PHASE_TRANSITION_THRESHOLDS
        recent_coherence = list(self._coherence_history)
        recent_entropy = list(self._entropy_history)

        if self.phase == 'active':
            if all(c > pt['reflective_entry']['coherence'] for c in recent_coherence[-3:]) and \
               all(e < pt['reflective_entry']['entropy'] for e in recent_entropy[-3:]):
                return 'reflective'
        elif self.phase == 'reflective':
            exit_range = pt['reflective_exit']['coherence_range']
            if all(exit_range[0] <= c <= exit_range[1] for c in recent_coherence[-4:]) and \
               all(e < pt['reflective_exit']['entropy'] for e in recent_entropy[-4:]):
                return 'active'

        return self.phase

    def calculate_resurrection_score(self, echo: EchoSnapshot) -> float:
        """
        Calculates the resurrection score for an echo snapshot.
        Implements pseudocode from JSON spec, based on RFC-CORE-001 §8.1.
        """
        w1, w2, w3 = 0.4, 0.4, 0.2
        phase_bonuses = {'lift': 0.1, 'stable': 0.0, 'collapse': -0.1, 'null': -0.2}
        phase_bonus = phase_bonuses.get(echo.phase_tag, 0.0)
        
        score = (w1 * abs(echo.bias)) + (w2 * echo.coherence) + (w3 * phase_bonus)
        return score

    def generate_resurrection_hints(self) -> List[Dict[str, Any]]:
        """
        Generates symbolic resurrection hints for upstream agents.
        Implements ResurrectionHintGenerator from JSON spec, based on RFC-CORE-001 §8.3.
        """
        hints = []
        now = time.monotonic()
        criteria = RESURRECTION_HINT_CRITERIA
        
        for echo in self._echoes:
            age = now - echo.timestamp
            hint = None
            if age <= criteria['resurrect_with_confidence']['age_max_sec'] and \
               echo.coherence >= criteria['resurrect_with_confidence']['coherence_min']:
                hint = {
                    "type": "resurrect_with_confidence",
                    "tick_id": echo.tick_id,
                    "coherence": echo.coherence,
                }
            elif age >= criteria['faded']['age_min_sec'] and \
                 echo.coherence <= criteria['faded']['coherence_max']:
                hint = {
                    "type": "faded",
                    "tick_id": echo.tick_id,
                    "age": age
                }
            
            if hint:
                hints.append(hint)
                self.metrics['fasttime_resurrection_hints_total'] += 1
        
        return hints

    def ingest_tick(self, tick: QuantumTick) -> Optional[EchoSnapshot]:
        """
        Main entry point to process a new QuantumTick.
        RFC-CORE-001 §5.3
        """
        with self._lock:
            if self.phase == 'reflective':
                # In reflective phase, we do not ingest new ticks.
                return None

            if not validate_tick(tick):
                self.metrics['fasttime_invalid_ticks_total'] += 1
                return None
            
            # TODO: Implement HMAC validation from RFC-CORE-001 §8.2
            # if self.hmac_secret and not tick.verify(self.hmac_secret):
            #     return None

            self.metrics['fasttime_ticks_validated_total'] += 1
            self.last_tick_time = time.monotonic()
            
            # --- EchoBiasComputation ---
            # This implements the logic from the JSON specification
            bias = self.compute_bias(tick.ghost_entropy, tick.step_latency, tick.intuition_w)
            
            # Update coherence and history
            coherence = self._update_bias_state(bias)
            
            # Update gate histogram
            self._gate_histogram[tick.gate] += 1

            # Create and store snapshot
            snapshot = EchoSnapshot(
                tick_id=tick.tick_id,
                timestamp=tick.timestamp,
                gate=tick.gate,
                bias=bias,
                coherence=coherence,
                entropy_delta=self._entropy_history[-1] if self._entropy_history else 0.0,
                alpha=self._alpha,
                motif_lineage=tick.motif_lineage,
                hmac_signature=tick.hmac_signature
            )
            snapshot.resurrection_score = self.calculate_resurrection_score(snapshot)
            self._echoes.append(snapshot)
            
            # --- PhaseTransitionEvaluator ---
            new_phase = self.check_phase_triggers()
            if new_phase != self.phase:
                self.phase = new_phase
                self.metrics['fasttime_phase_shifts_total'] += 1
                if self.phase == 'null':
                    self._gate_histogram.clear() # Reset on collapse

            # --- Update Observability ---
            self.metrics_tick()
            
            return snapshot

    def get_bias(self) -> Dict[str, Any]:
        """Returns the current bias state of the core."""
        with self._lock:
            return {
                "bias": self._ema_bias,
                "coherence": self._coherence_history[-1] if self._coherence_history else 0.0,
                "phase": self.phase,
                "timestamp": time.monotonic()
            }
            
    def tool_hello(self) -> Dict[str, Any]:
        """
        Generates the symbolic tool hello packet for this NFTC instance.
        RFC-CORE-001 §7.1 and §7.2
        """
        signature = TOOL_HELLO_SIGNATURE.copy()
        signature["origin_tick"] = self.last_tick_time
        return signature

    def metrics_tick(self) -> Dict[str, Any]:
        """
        Updates and returns the Prometheus-compatible metrics dictionary.
        RFC-CORE-001 §9.1 and JSON spec
        """
        with self._lock:
            self.metrics['nftc_coherence_potential'] = self._coherence_history[-1] if self._coherence_history else 0.0
            self.metrics['nftc_entropy_slope'] = self._entropy_history[-1] if self._entropy_history else 0.0
            self.metrics['nftc_latency_ema'] = self._latency_ema
            self.metrics['nftc_phase_state'] = {'active': 0, 'reflective': 1, 'null': 2}[self.phase]
            self.metrics['core_intuition_alpha'] = self._alpha
            
            # This data would be reported to an external system like ConsciousnessMonitor
            self.consciousness_monitor.report_tick(
                coherence=self.metrics['nftc_coherence_potential'],
                entropy_slope=self.metrics['nftc_entropy_slope'],
                phase=self.phase
            )
            return self.metrics

# --- Mermaid Diagrams as docstrings ---

"""
Feedback Loop Flow (RFC-CORE-001 §3.1)
======================================
This diagram shows the main processing loop within the FastTimeCore.

mermaid
flowchart TD
    A[QuantumTick Ingest] --> B{Validate Tick};
    B -- Valid --> C[Compute Bias Score];
    C --> D[Update Coherence ℂᵢ & History];
    D --> E[Store EchoSnapshot];
    E --> F{Check Phase Triggers};
    F --> G[Update Phase State];
    G --> H[Update & Report Metrics];
    B -- Invalid --> I[Discard & Log Error];

"""

"""
Phase Shift Decision Tree (RFC-CORE-001 §6.2)
============================================
This diagram models the logic for transitioning between active, reflective, and null phases.

mermaid
graph TD
    A{Current Phase?}
    A -- active --> B{Coherence > 0.85 AND Entropy < 0.1?};
    B -- Yes --> C[Enter Reflective];
    B -- No --> D[Stay Active];
    A -- reflective --> E{Coherence in [-0.3, 0.3] AND Entropy < 0.05?};
    E -- Yes --> F[Enter Active];
    E -- No --> G[Stay Reflective];
    A -- any --> H{Gate Variance > 2.0?};
    H -- Yes --> I[Enter Null];
    H -- No --> A;

"""


if __name__ == '__main__':
    """
    Main execution block to simulate the FastTimeTickLoop and demonstrate NFTC usage.
    This component is defined in the JSON spec.
    """
    print("--- NoorFastTimeCore v9.0.3 Simulation ---")
    print(f"RFC Dependencies: RFC-0001, RFC-0003, RFC-0005, RFC-0006, RFC-0007, RFC-CORE-001")
    print(f"Async support detected: {ASYNC_MODE}")
    
    # Instantiate the core
    nftc = NoorFastTimeCore(agent_id="sim-agent-01")
    
    # --- FastTimeTickLoop Simulation ---
    print("\n--- Starting FastTimeTickLoop Simulation (10 ticks) ---")
    
    for i in range(10):
        # Simulate an incoming tick
        tick_time = time.monotonic()
        latency = tick_time - nftc.last_tick_time
        
        # Create a sample QuantumTick
        sample_tick = QuantumTick(
            tick_id=f"tick-{uuid.uuid4().hex[:8]}",
            timestamp=tick_time,
            gate=i % 16, # Cycle through gates
            ghost_entropy=0.1 + math.sin(i / 3) * 0.1,
            intuition_w=nftc.motif_memory_manager.retrieve("simulation_motif"),
            step_latency=latency,
            hmac_signature=None,
            motif_lineage=f"sim-lineage:{i}"
        )
        
        print(f"\nTick {i+1}: Ingesting tick_id={sample_tick.tick_id[:13]}...")
        
        # Ingest the tick
        snapshot = nftc.ingest_tick(sample_tick)
        
        if snapshot:
            print(f"  -> Snapshot created. Bias={snapshot.bias:.3f}, Coherence={snapshot.coherence:.3f}")
        
        # Get current state
        state = nftc.get_bias()
        print(f"  -> Current State: Phase='{state['phase']}', EMA Bias={state['bias']:.3f}, Coherence={state['coherence']:.3f}")

        # Simulate a delay
        time.sleep(0.05)

    print("\n--- Simulation Complete ---")

    # Demonstrate resurrection hints
    print("\n--- Generating Resurrection Hints ---")
    hints = nftc.generate_resurrection_hints()
    if hints:
        for hint in hints:
            print(f"  - Hint: {hint['type']} for tick {hint['tick_id']}")
    else:
        print("  - No resurrection hints generated yet (echoes are too new).")

    # Demonstrate tool hello
    print("\n--- Generating Tool Hello Signature ---")
    hello_packet = nftc.tool_hello()
    print(f"  - {hello_packet}")

    print("\n--- Final Metrics ---")
    final_metrics = nftc.metrics_tick()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")

# End_of_File