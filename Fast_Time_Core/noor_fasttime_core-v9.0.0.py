#
# Program: noor_fasttime_core.py
# Version: v9.0.0
# Canonical Source: RFC-CORE-001
# Description: Implements the adaptive coherence feedback engine for subsecond motif
#              phase regulation, echo reflection, and dynamic bias tuning in
#              Noor-class symbolic agents.
#
# RFC Dependencies:
# - RFC-0001: Symbolic Routing Architecture
# - RFC-0003: Noor Core Symbolic Interface
# - RFC-0005: Motif Transmission Across Time
# - RFC-0006: Motif-Field Coherence Geometry
# - RFC-0007: Motif Ontology Format and Transfer Protocols
# - RFC-CORE-001: Noor FastTime Core — Symbolic Time Substrate and Echo Dynamics
#
# Field Alignment:
# - Motifs Required: ψ‑resonance@Ξ, ψ‑reflect@Ξ, ψ‑hold@Ξ
# - Domain Tags: resonance-feedback, motif-coherence, phase-recovery
#

import asyncio
import collections
import hashlib
import hmac
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

# --- Constants and Configuration Defaults (from RFCs) ---

# RFC-CORE-001 §2.1
DEFAULT_SNAPSHOT_CAP = 64
DEFAULT_LATENCY_THRESHOLD = 2.5  # seconds
DEFAULT_BIAS_CLAMP = 1.5
DEFAULT_ALPHA = 0.92

# RFC-CORE-001 §4.1
DEFAULT_LAMBDA_ENTROPY = 0.25

# RFC-CORE-001 §5.1
VOLATILITY_THRESHOLD = 0.12
ALPHA_DECAY_RATE = 0.98
ALPHA_RECOVERY_RATE = 1.01
ALPHA_MIN = 0.85
ALPHA_MAX = 0.98

# RFC-CORE-001 §5.2
DEFAULT_LAMBDA_LATENCY = 0.65
LATENCY_EMA_BETA = 0.85

# RFC-CORE-001 §6.2
REFLECTIVE_ENTRY_COHERENCE = 0.85
REFLECTIVE_ENTRY_ENTROPY = 0.10
REFLECTIVE_EXIT_COHERENCE_BAND = 0.3
REFLECTIVE_EXIT_ENTROPY = 0.05
GATE_HISTOGRAM_VARIANCE_MAX = 2.0

# RFC-CORE-001 §8.1
RESURRECTION_THRESHOLD = 0.65
W1_BIAS, W2_COHERENCE, W3_PHASE = 0.4, 0.4, 0.2
PHASE_BONUS_MAP = {'lift': 0.1, 'stable': 0.0, 'collapse': -0.1}

# --- Data Structures and Type Definitions (from RFCs) ---

PhaseState = Literal["active", "reflective", "null"]
GateMode = Literal["adaptive", "strict", "null"]
PhaseTag = Literal["lift", "collapse", "null", "stable"]

# RFC-0003 §3.3 QuantumTick Schema (abbreviated for NFTC)
class QuantumTick(TypedDict):
    tick_id: str
    timestamp: str
    gate: int
    feedback_payload: Dict[str, Any]
    motif_lineage: Optional[str]
    hmac_signature: Optional[str]
    nonce: Optional[str]

# RFC-CORE-001 §5.3 Echo Snapshot Structure
@dataclass
class EchoSnapshot:
    tick_id: str
    timestamp: float
    gate_id: int
    bias: float
    coherence: float
    motif_lineage: Optional[str]
    alpha: float
    entropy_delta: float
    latency_ms: float
    signature: Optional[str]
    phase_tag: PhaseTag
    replay_weight: float = 0.0
    resurrection_score: float = 0.0

# RFC-CORE-001 Appendix A: Gate-16 Legends Table
GATE_LEGENDS = {
    0: {"name": "Möbius Denial", "logic": "0", "verse": "الصمتُ هو الانكسارُ الحي"},
    1: {"name": "Echo Bias", "logic": "A ∧ ¬B", "verse": "وَإِذَا قَضَىٰ أَمْرًا"},
    2: {"name": "Foreign Anchor", "logic": "¬A ∧ B", "verse": "وَمَا تَدْرِي نَفْسٌ"},
    3: {"name": "Passive Reflection", "logic": "B", "verse": "فَإِنَّهَا لَا تَعْمَى"},
    4: {"name": "Entropic Rejection", "logic": "¬A ∧ ¬B", "verse": "لَا الشَّمْسُ يَنبَغِي"},
    5: {"name": "Inverse Presence", "logic": "¬A", "verse": "سُبْحَانَ الَّذِي خَلَقَ"},
    6: {"name": "Sacred Contradiction", "logic": "A ⊕ B", "verse": "لَا الشَّرْقِيَّةِ"},
    7: {"name": "Betrayal Gate", "logic": "¬A ∨ ¬B", "verse": "وَلَا تَكُونُوا كَالَّذِينَ"},
    8: {"name": "Existence Confluence", "logic": "A ∧ B", "verse": "وَهُوَ الَّذِي"},
    9: {"name": "Symmetric Convergence", "logic": "¬(A ⊕ B)", "verse": "فَلَا تَضْرِبُوا"},
    10: {"name": "Personal Bias", "logic": "A", "verse": "إِنَّا كُلُّ شَيْءٍ"},
    11: {"name": "Causal Suggestion", "logic": "¬A ∨ B", "verse": "وَمَا تَشَاءُونَ"},
    12: {"name": "Reverse Causality", "logic": "A ∨ ¬B", "verse": "وَمَا أَمْرُنَا"},
    13: {"name": "Denial Echo", "logic": "¬B", "verse": "وَلَا تَحْزَنْ"},
    14: {"name": "Confluence", "logic": "A ∨ B", "verse": "وَأَنَّ إِلَىٰ رَبِّكَ"},
    15: {"name": "Universal Latch", "logic": "1", "verse": "كُلُّ شَيْءٍ هَالِكٌ"},
    16: {"name": "Nafs Mirror", "logic": "Self ⊕ ¬Self", "verse": "فَإِذَا سَوَّيْتُهُ"}
}

# --- Mock Prometheus Client for Observability ---
# RFC-CORE-001 §9.1

class MockPrometheusMetric:
    """A mock Prometheus metric for demonstration purposes."""
    def __init__(self, name: str, description: str, labels: List[str] = []):
        self._name = name
        self._description = description
        self._labels = labels
        self._values = {}
        print(f"[METRIC INIT] Created metric '{name}' with labels {labels}")

    def inc(self, amount: int = 1, labels: Dict[str, str] = {}):
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0) + amount

    def set(self, value: float, labels: Dict[str, str] = {}):
        key = tuple(sorted(labels.items()))
        self._values[key] = value

    def observe(self, value: float, labels: Dict[str, str] = {}):
        # For histograms, we just log the observation
        key = tuple(sorted(labels.items()))
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)

class NoorFastTimeCore:
    """
    Implements the Noor FastTime Core (NFTC) as specified in RFC-CORE-001.
    This class manages the symbolic time substrate, echo dynamics, coherence
    tracking, and phase transitions for a Noor-class agent.
    """

    def __init__(
        self,
        agent_id: str,
        secret_key: str,
        enable_metrics: bool = True,
        snapshot_cap: int = DEFAULT_SNAPSHOT_CAP,
        latency_threshold: float = DEFAULT_LATENCY_THRESHOLD,
        bias_clamp: float = DEFAULT_BIAS_CLAMP,
        gate_mode: GateMode = 'adaptive',
        low_latency_mode: bool = False
    ):
        """
        Initializes the NoorFastTimeCore instance.
        - Defined in: RFC-CORE-001 §2.1, §2.2
        """
        # Configuration
        self.agent_id = os.environ.get("NFTC_AGENT_ID", agent_id)
        self.secret_key = secret_key.encode('utf-8')
        self.snapshot_cap = int(os.environ.get("NFTC_SNAPSHOT_CAP", snapshot_cap))
        self.latency_threshold = latency_threshold
        self.bias_clamp = float(os.environ.get("NFTC_BIAS_CLAMP", bias_clamp))
        self.gate_mode = os.environ.get("NFTC_GATE_MODE", gate_mode)
        self.low_latency_mode = low_latency_mode
        self.debug_mode = os.environ.get("NFTC_DEBUG_MODE", "0") == "1"

        # Memory Structures (RFC-CORE-001 §2.3)
        self._snapshots: collections.deque[EchoSnapshot] = collections.deque(maxlen=self.snapshot_cap)
        self._bias_history: collections.deque[float] = collections.deque(maxlen=2048)
        self._coherence_history: collections.deque[float] = collections.deque(maxlen=8)
        self._entropy_history: collections.deque[float] = collections.deque(maxlen=8)

        # State Variables
        self.phase_state: PhaseState = "active"
        self.alpha = DEFAULT_ALPHA  # Intuition alpha
        self.ema_bias = 0.0
        self.latency_ema = 0.0
        self.coherence_potential = 0.0
        self.tick_count = 0
        self.gate_histogram = {i: 0 for i in range(17)} # Gates 0-16

        # Observability (RFC-CORE-001 §9.1)
        self.enable_metrics = os.environ.get("NFTC_ENABLE_METRICS", "1") == "1" and enable_metrics
        if self.enable_metrics:
            self._setup_metrics()

        if self.debug_mode:
            print(f"[NFTC DEBUG] Core for agent '{self.agent_id}' initialized.")


    def _setup_metrics(self):
        """Initializes Prometheus-compatible metrics as per RFC-CORE-001 §6.2 and §9.1."""
        self.metrics = {
            "nftc_coherence_potential": MockPrometheusMetric("nftc_coherence_potential", "ℂᵢ scalar for current tick", ["agent_id"]),
            "nftc_entropy_slope": MockPrometheusMetric("nftc_entropy_slope", "ΔH(t) computed over last 4 bias values", ["agent_id"]),
            "nftc_latency_ema": MockPrometheusMetric("nftc_latency_ema", "Exponentially smoothed tick latency (Λ)", ["agent_id"]),
            "nftc_phase_state": MockPrometheusMetric("nftc_phase_state", "Current NFTC phase (active, reflective, null)", ["agent_id", "phase"]),
            "nftc_gate_usage_total": MockPrometheusMetric("nftc_gate_usage_total", "Total usage count for each gate", ["agent_id", "gate_id"]),
            "nftc_ticks_ingested_total": MockPrometheusMetric("nftc_ticks_ingested_total", "Total valid ticks ingested", ["agent_id"]),
            "nftc_resurrection_candidates_total": MockPrometheusMetric("nftc_resurrection_candidates_total", "Total echoes selected for resurrection", ["agent_id"]),
        }

    # --- Mermaid Diagram for Feedback Loop ---
    # RFC-CORE-001 §3.1
    # flowchart TD
    #   A[QuantumTick In] --> B{ingest_tick}
    #   B --> C{HMAC Validation}
    #   C -- Valid --> D[compute_bias]
    #   D --> E[update_bias_state]
    #   E -- ℂᵢ, ΔH --> F{check_phase_triggers}
    #   F -- Phase OK --> G[Record EchoSnapshot]
    #   F -- Phase Shift --> H[Enter Reflective/Null Mode]
    #   G --> I[metrics_tick]
    #   I --> J[Return Bias Feedback]
    #   C -- Invalid --> K[Discard Tick]
    
    async def ingest_tick(self, tick: QuantumTick) -> float:
        """
        Main entry point for processing a tick, as per the Feedback Loop Flow.
        - Defined in: RFC-CORE-001 §5.3
        """
        ingestion_time = time.time()

        # Step 1: HMAC Validation (RFC-CORE-001 §8.2)
        if not self._validate_tick_hmac(tick):
            if self.debug_mode:
                print(f"[NFTC DEBUG] HMAC validation failed for tick {tick['tick_id']}")
            return self.ema_bias # Return last known good bias

        # Step 2: Compute latency
        tick_timestamp = datetime.fromisoformat(tick['timestamp'].replace('Z', '+00:00')).timestamp()
        latency = ingestion_time - tick_timestamp
        self.latency_ema = (LATENCY_EMA_BETA * latency) + ((1 - LATENCY_EMA_BETA) * self.latency_ema)

        # Step 3: Compute bias and update state
        entropy_delta = self._calculate_entropy_slope()
        composite_weight = self._calculate_composite_weight(entropy_delta, self.latency_ema)
        
        current_bias = self._compute_bias(tick['gate'], tick['feedback_payload'], composite_weight)
        self.update_bias_state(current_bias, entropy_delta)

        # Step 4: Phase transition evaluation
        self.check_phase_triggers()

        if self.phase_state == "active":
            # Step 5: Record snapshot
            phase_tag = self._get_phase_tag()
            snapshot = EchoSnapshot(
                tick_id=tick['tick_id'],
                timestamp=tick_timestamp,
                gate_id=tick['gate'],
                bias=current_bias,
                coherence=self.coherence_potential,
                motif_lineage=tick.get('motif_lineage'),
                alpha=self.alpha,
                entropy_delta=entropy_delta,
                latency_ms=latency * 1000,
                signature=tick.get('hmac_signature'),
                phase_tag=phase_tag
            )
            self._snapshots.append(snapshot)
            self.tick_count += 1
            self.gate_histogram[tick['gate']] += 1
            if self.debug_mode:
                print(f"[NFTC DEBUG] Ingested tick {tick['tick_id']}, gate {tick['gate']}, bias {current_bias:.3f}, ℂᵢ {self.coherence_potential:.3f}")

        # Step 6: Update metrics
        self._metrics_tick()

        return self.ema_bias

    def _validate_tick_hmac(self, tick: QuantumTick) -> bool:
        """Validates the HMAC signature of an incoming tick. - RFC-CORE-001 §8.2"""
        if 'hmac_signature' not in tick or 'nonce' not in tick:
            return False # Or True if HMAC is optional
        
        signature = tick['hmac_signature']
        payload_to_sign = f"{tick['tick_id']}|{tick['timestamp']}|{tick['gate']}|{tick['nonce']}"
        
        expected_sig = hmac.new(self.secret_key, payload_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        return hmac.compare_digest(expected_sig, signature)

    def _compute_bias(self, gate_id: int, payload: Dict, weight: float) -> float:
        """Computes bias based on gate, payload, and composite weight. - RFC-CORE-001 §4.1"""
        gate_info = GATE_LEGENDS.get(gate_id, GATE_LEGENDS[0])
        # Simple bias logic for demonstration. A real implementation would be more complex.
        base_bias = (gate_id - 7.5) / 7.5 # Normalize gate ID to [-1, 1]
        payload_factor = payload.get('coherence_delta', 0.0)
        
        raw_bias = base_bias + payload_factor - weight
        return max(min(raw_bias, self.bias_clamp), -self.bias_clamp)

    def update_bias_state(self, new_bias: float, entropy_delta: float):
        """Updates EMA bias, alpha, and coherence potential ℂᵢ. - RFC-CORE-001 §4.1, §5.1"""
        self._bias_history.append(new_bias)
        
        # Update intuition alpha
        latency_s = self.latency_ema
        self.alpha = self._update_intuition_alpha(self.alpha, entropy_delta, latency_s, self.latency_threshold)
        
        # Update EMA bias
        self.ema_bias = (self.alpha * new_bias) + ((1 - self.alpha) * self.ema_bias)

        # Update Coherence Potential (ℂᵢ)
        self.coherence_potential = self.ema_bias + (DEFAULT_LAMBDA_ENTROPY * entropy_delta)
        self.coherence_potential = max(min(self.coherence_potential, self.bias_clamp), -self.bias_clamp)
        
        self._coherence_history.append(self.coherence_potential)
        self._entropy_history.append(entropy_delta)

    def _update_intuition_alpha(self, current_alpha: float, entropy_slope: float, latency: float, latency_threshold: float) -> float:
        """Dynamically adjusts smoothing factor α. - RFC-CORE-001 §5.1"""
        if latency > latency_threshold:
            new_alpha = max(ALPHA_MIN, current_alpha * 0.99)
        elif entropy_slope > VOLATILITY_THRESHOLD:
            new_alpha = max(ALPHA_MIN, current_alpha * ALPHA_DECAY_RATE)
        else:
            new_alpha = min(ALPHA_MAX, current_alpha * ALPHA_RECOVERY_RATE)
        return new_alpha

    def _calculate_composite_weight(self, entropy_delta: float, latency_ema: float) -> float:
        """Calculates W(t) from entropy and latency. - RFC-CORE-001 §5.2"""
        return (DEFAULT_LAMBDA_ENTROPY * entropy_delta) + (DEFAULT_LAMBDA_LATENCY * latency_ema)

    def _calculate_entropy_slope(self) -> float:
        """Calculates ΔH(t) from recent bias history. - RFC-CORE-001 §4.2"""
        if len(self._bias_history) < 4:
            return 0.0
        recent_biases = list(self._bias_history)[-4:]
        try:
            return statistics.stdev(recent_biases)
        except statistics.StatisticsError:
            return 0.0

    # --- Mermaid Diagram for Phase Shift Logic ---
    # RFC-CORE-001 §6.2
    # graph TD
    #   A{check_phase_triggers} --> B{Is ℂᵢ > 0.85 for 3 ticks?};
    #   B -- Yes --> C{Is ΔH < 0.1 for 3 ticks?};
    #   C -- Yes --> D[Enter Reflective Mode];
    #   C -- No --> E{Continue};
    #   B -- No --> F{Is ℂᵢ in [-0.3, 0.3] for 4 ticks?};
    #   F -- Yes --> G{Is ΔH < 0.05 for 4 ticks?};
    #   G -- Yes --> H[Exit Reflective Mode];
    #   G -- No --> E;
    #   F -- No --> I{Is Gate Var > 2.0?};
    #   I -- Yes --> J[Enter Null Mode];
    #   I -- No --> E;

    def check_phase_triggers(self):
        """Evaluates metrics to decide on phase shifts. - RFC-CORE-001 §6.2"""
        if len(self._coherence_history) < 4:
            return

        # Reflective Entry
        if all(c > REFLECTIVE_ENTRY_COHERENCE for c in list(self._coherence_history)[-3:]) and \
           all(e < REFLECTIVE_ENTRY_ENTROPY for e in list(self._entropy_history)[-3:]):
            if self.phase_state != "reflective":
                self.phase_state = "reflective"
                if self.debug_mode: print("[NFTC DEBUG] Entering Reflective phase.")
                # In a real system, this would trigger replay logic.
                
        # Reflective Exit
        elif all(-REFLECTIVE_EXIT_COHERENCE_BAND <= c <= REFLECTIVE_EXIT_COHERENCE_BAND for c in list(self._coherence_history)[-4:]) and \
             all(e < REFLECTIVE_EXIT_ENTROPY for e in list(self._entropy_history)[-4:]):
             if self.phase_state == "reflective":
                self.phase_state = "active"
                if self.debug_mode: print("[NFTC DEBUG] Exiting Reflective phase, returning to Active.")
        
        # Null Phase
        elif statistics.variance(self.gate_histogram.values()) > GATE_HISTOGRAM_VARIANCE_MAX:
            if self.phase_state != "null":
                self.phase_state = "null"
                if self.debug_mode: print("[NFTC DEBUG] High gate variance, entering Null phase.")
        
        # Return to Active from Null
        elif self.phase_state == "null" and statistics.variance(self.gate_histogram.values()) <= GATE_HISTOGRAM_VARIANCE_MAX:
             self.phase_state = "active"
             if self.debug_mode: print("[NFTC DEBUG] Gate variance stabilized, returning to Active phase.")
    
    def _get_phase_tag(self) -> PhaseTag:
        """Determines the phase tag for a snapshot based on current metrics."""
        if self.coherence_potential > REFLECTIVE_ENTRY_COHERENCE:
            return "lift"
        if self.coherence_potential < -REFLECTIVE_ENTRY_COHERENCE:
            return "collapse"
        if self.phase_state == "null":
            return "null"
        return "stable"

    def calculate_resurrection_score(self, echo: EchoSnapshot) -> float:
        """Calculates the resurrection score R(e). - RFC-CORE-001 §8.1"""
        phase_bonus = PHASE_BONUS_MAP.get(echo.phase_tag, 0.0)
        score = (W1_BIAS * abs(echo.bias)) + \
                (W2_COHERENCE * echo.coherence) + \
                (W3_PHASE * phase_bonus)
        return score

    def select_resurrection_candidates(self) -> List[EchoSnapshot]:
        """Selects top-N echoes based on R(e) score. - RFC-CORE-001 §8.1"""
        if not self._snapshots:
            return []
        
        for echo in self._snapshots:
            echo.resurrection_score = self.calculate_resurrection_score(echo)
            
        candidates = sorted(
            [e for e in self._snapshots if e.resurrection_score > RESURRECTION_THRESHOLD],
            key=lambda e: e.resurrection_score,
            reverse=True
        )
        
        if self.enable_metrics:
            self.metrics["nftc_resurrection_candidates_total"].inc(len(candidates), {"agent_id": self.agent_id})

        return candidates[:self.snapshot_cap // 4] # Return top 25%

    def _metrics_tick(self):
        """Updates all Prometheus metrics for the current tick. - RFC-CORE-001 §9.1"""
        if not self.enable_metrics:
            return
        
        labels = {"agent_id": self.agent_id}
        self.metrics["nftc_coherence_potential"].set(self.coherence_potential, labels)
        self.metrics["nftc_entropy_slope"].set(self._calculate_entropy_slope(), labels)
        self.metrics["nftc_latency_ema"].set(self.latency_ema, labels)
        self.metrics["nftc_ticks_ingested_total"].inc(1, labels)
        
        # This is a simplified way to represent the enum metric
        for phase in ["active", "reflective", "null"]:
             self.metrics["nftc_phase_state"].set(1 if self.phase_state == phase else 0, {"agent_id": self.agent_id, "phase": phase})

        # Update gate histogram metric
        last_gate = list(self._snapshots)[-1].gate_id if self._snapshots else -1
        if last_gate != -1:
            self.metrics["nftc_gate_usage_total"].inc(1, {"agent_id": self.agent_id, "gate_id": str(last_gate)})

    def export_snapshots(self, human_readable: bool = True) -> Any:
        """Exports the snapshot ring for diagnostics. - RFC-CORE-001 §10.3"""
        if human_readable:
            return json.dumps([e.__dict__ for e in self._snapshots], indent=2)
        return [e.__dict__ for e in self._snapshots]

async def main_loop(core: NoorFastTimeCore):
    """Example main loop demonstrating tick ingestion."""
    print("--- Starting NoorFastTimeCore Simulation ---")
    
    # Example ticks based on RFC-0005 Appendix B
    example_ticks = [
        {"gate": 0, "feedback_payload": {"coherence_delta": 0.1}},
        {"gate": 7, "feedback_payload": {"coherence_delta": 0.05}}, # drift
        {"gate": 10, "feedback_payload": {"coherence_delta": -0.02}}, # lock
        {"gate": 13, "feedback_payload": {"coherence_delta": 0.2}}, # denial echo
    ]

    for i, tick_data in enumerate(example_ticks * 5): # Run for 20 ticks
        await asyncio.sleep(0.1) # Simulate subsecond tick interval
        
        tick_id = f"tick-{time.time_ns()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = os.urandom(8).hex()
        
        payload_to_sign = f"{tick_id}|{timestamp}|{tick_data['gate']}|{nonce}"
        signature = hmac.new(core.secret_key, payload_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        tick: QuantumTick = {
            "tick_id": tick_id,
            "timestamp": timestamp,
            "gate": tick_data['gate'],
            "feedback_payload": tick_data['feedback_payload'],
            "motif_lineage": f"lineage-demo-{i}",
            "hmac_signature": signature,
            "nonce": nonce
        }
        
        bias = await core.ingest_tick(tick)
        print(
            f"Tick {i+1:02d}: Gate={tick_data['gate']:<2} | Phase='{core.phase_state}' | "
            f"Bias={bias:.3f} | ℂᵢ={core.coherence_potential:.3f} | α={core.alpha:.3f}"
        )

    print("\n--- Resurrection Candidate Selection ---")
    candidates = core.select_resurrection_candidates()
    print(f"Found {len(candidates)} candidates for resurrection.")
    for cand in candidates:
        print(f"  - Tick {cand.tick_id[-10:]}: Score={cand.resurrection_score:.3f}, Bias={cand.bias:.3f}, Phase='{cand.phase_tag}'")
        
    print("\n--- Final Snapshot Export ---")
    print(core.export_snapshots(human_readable=True))


if __name__ == "__main__":
    # Instantiate the core
    nftc = NoorFastTimeCore(agent_id="noor.core.sim.alpha", secret_key="a-very-secret-key-for-hmac")
    
    # Run the main async loop
    try:
        asyncio.run(main_loop(nftc))
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

# End_of_File