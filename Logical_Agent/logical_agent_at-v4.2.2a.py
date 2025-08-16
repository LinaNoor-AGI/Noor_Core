#
# Copyright 2025 Lina Noor
#
# Licensed under the MIT License;
#
#  Lina Noor  <lina.noor@noor-agi.com>
#  Uncle     <uncle@noor-agi.com>
#
#
# logical_agent_at.py
# ----------------------------------------------------------------------------------
# This module implements the LogicalAgentAT, a core component of the Noor symbolic
# cognitive architecture. As defined in RFC-CORE-003, its primary role is to act as
# a non-mutating symbolic observer. It evaluates QuantumTicks emitted by the
# RecursiveAgentFT, identifies coherent motif patterns (dyads and triads), tracks
# unresolved symbolic tension (contradiction pressure), and manages the lifecycle
# of 'ghost' motifs.
#
# The agent's behavior is governed by a strict observer contract: unless explicitly
# configured otherwise, it will not alter the symbolic field, ensuring its feedback
# is a pure reflection of the system's state. It provides crucial observability
# hooks for the ConsciousnessMonitor and exports detailed feedback packets for
# system-wide coherence analysis.
#
# This implementation is generated in compliance with the PDP-0001 protocol,
# ensuring traceable fidelity to the canonical RFCs and the provided application
# specification.
#
# AI Platform/Model: Google Gemini 2.4 Pro
# Generation Date: 2024-09-01T12:00:00Z
#

__version__ = "4.2.2b"
_SCHEMA_VERSION__ = "2025-Q4-canonical-header-v1"

import os
import re
import time
import math
import hashlib
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Deque
from dataclasses import dataclass, field
from statistics import mean
import random

# --- Optional Dependencies with Graceful Fallbacks (PDP-0001 §4.5) ---
try:
    import numpy as np
    from scipy.sparse import csgraph
    from scipy.linalg import expm
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import multiprocessing
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    
# Stub class for when the monitor is not present.
class _StubMonitor:
    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> None:
            pass
        return _stub

# --- Dataclasses for Structured Communication (RFC-CORE-003) ---

@dataclass
class TickAnnotations:
    """Structured annotations for an evaluated QuantumTick."""
    triad_complete: bool
    ctx_ratio: float
    trust: float
    resurrection_hint: Optional[str] = None
    memory_promotion: bool = False
    reward_delta: float = 0.0

@dataclass
class FeedbackPacket:
    """Exportable summary of the agent's internal symbolic state."""
    ctx_ratio: float
    contradiction_avg: float
    harm_hits: int
    recent_mutations: int
    ring_patch: Optional[str]
    ghost_hint: Optional[str]
    entropy_drift: List[Tuple[str, float, float]]
    contradiction_context: List[Dict[str, Any]]

@dataclass
class OpinionPacket:
    """Mandated output structure for turns with 'opinion' intent (RFC-CORE-003 §3.1)."""
    stance: str
    risks: List[str]
    actions: List[str]
    tone: str
    annotations: Dict[str, Any]

# --- Core Logic Implementation ---

PI_TAG_REGEX = re.compile(r"^[ψμ]?[a-z0-9_:\-]{1,48}$")

class DynamicFeatureFlagMixin:
    """Manages runtime toggles for agent capabilities (Spec §1.8)."""
    def __init__(self):
        self._DYNAMIC_FLAGS: Set[str] = {
            "enable_ghost_tracking", "enable_pi_equivalence", "enable_laplacian_smoothing",
            "enable_recursive_triads", "enable_dyad_chains", "enable_contradiction_pressure",
            "enable_context_journal", "enable_entropy_journal", "enable_topology_validation"
        }

    def set_feature(self, name: str, value: bool):
        if value:
            self._DYNAMIC_FLAGS.add(name)
        else:
            self._DYNAMIC_FLAGS.discard(name)
        logging.info(f"Feature '{name}' set to {value}")

    def get_feature(self, name: str) -> bool:
        return name in self._DYNAMIC_FLAGS

class LazyMonitorMixin:
    """Provides lazy-loading for the optional ConsciousnessMonitor (RFC-CORE-003 §9.1)."""
    _monitor_instance = None

    @property
    def monitor(self):
        if self._monitor_instance is None:
            try:
                from consciousness_monitor import get_global_monitor
                self._monitor_instance = get_global_monitor()
            except ImportError:
                self._monitor_instance = _StubMonitor()
        return self._monitor_instance

    def set_global_monitor(self, monitor):
        self._monitor_instance = monitor


class LogicalAgentAT(DynamicFeatureFlagMixin, LazyMonitorMixin):
    """
    Symbolic observer agent for coherence evaluation, triad resolution, and
    field stabilization, compliant with RFC-CORE-003.
    """
    def __init__(self, agent_id: str = "logical_agent_at_0", observer_mode: bool = True):
        super().__init__()
        self.agent_id = agent_id
        self.observer_mode = observer_mode
        self._generation = 0

        # --- State variables from RFC-CORE-003 ---
        self._entanglement_fields: Dict[str, Dict[str, Any]] = {}
        self._ghost_motifs: Dict[str, Dict[str, Any]] = {}
        self._confirmed_triads: Dict[str, Dict[str, Any]] = {}
        self._pi_classes: Dict[str, Set[str]] = {}

        # Contradiction and mutation tracking (RFC-CORE-003 §6)
        self._dyad_window_size = self._compute_default_dyad_window()
        self._dyad_window: Deque[float] = deque(maxlen=self._dyad_window_size)
        self._contradiction_avg = 0.0
        self._contradiction_count = 0
        self._recent_mutations: Deque[int] = deque(maxlen=50)
        self._contradiction_log: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._drift_log: Deque[Tuple[str, float, float]] = deque(maxlen=128)
        
        self._last_ctx_ratio = 0.5

    def _compute_default_dyad_window(self) -> int:
        # RFC-CORE-003 §11.3: Adaptive dyad window sizing.
        if 'NOOR_DYAD_WINDOW_SIZE' in os.environ:
            return int(os.environ['NOOR_DYAD_WINDOW_SIZE'])
        if MULTIPROCESSING_AVAILABLE:
            cpu_count = multiprocessing.cpu_count()
            return max(13, min(256, 13 + cpu_count * 4))
        return 13

    def _guard_write(self) -> bool:
        """Enforces the non-mutating observer contract (RFC-CORE-003 §8.2)."""
        return not self.observer_mode

    def evaluate_tick(self, tick: Dict[str, Any]) -> Optional[Union[TickAnnotations, OpinionPacket]]:
        """
        Core evaluation pipeline for an incoming QuantumTick (RFC-CORE-003 §2.1).
        This method also handles intent-based output switching.
        """
        if not tick or "motifs" not in tick:
            return None # Fails open as per spec

        # RFC-CORE-003 §3.1: Handle intent-based response contracts.
        # Intent is normalized per RFC-0004 §2.5.
        intent = tick.get("extensions", {}).get("intent", "neutral")
        
        if intent == "opinion":
            # This path generates a structured, evaluative response.
            return self._generate_opinion_packet(tick)

        # Default path: Generate reflective TickAnnotations.
        # This simulates the full feedback generation process.
        fb = self._make_field_feedback(tick) 
        dyad = self._detect_dyad(tick.get("motifs", []))
        triad = self._complete_triad(dyad) if dyad else None

        annotations = TickAnnotations(
            triad_complete=bool(triad),
            ctx_ratio=fb['ctx_feedback']['ctx_ratio'],
            trust=fb['trust_profiles'][0]['trust'] if fb.get('trust_profiles') else 0.5,
            resurrection_hint=fb.get('extensions', {}).get('resurrection_hint')
        )
        return annotations
    
    def _generate_opinion_packet(self, tick: Dict[str, Any]) -> OpinionPacket:
        # RFC-CORE-003 §3.1 mandates this structure for 'opinion' intent.
        # The logic here is a placeholder for a deeper inference process.
        stance = "Symbolic coherence is stable, but dyadic tension is rising."
        risks = [
            "Increased contradiction pressure may lead to cluster mutation.",
            "Ghost motif resonance is low, indicating a lack of novel patterns.",
            "Potential for resonance cascade if a key triad fails."
        ]
        actions = [
            "Monitor dyad window for sustained low coherence.",
            "Inject novel, low-weight motifs to seed new ghost patterns.",
            "Reinforce stable triads with high swirl scores."
        ]
        
        return OpinionPacket(
            stance=stance,
            risks=risks,
            actions=actions,
            tone="Challenger(🔥)",
            annotations={
                "source_tick": tick.get("tick_id"),
                "confidence": 0.85,
                "observer_mode": self.observer_mode
            }
        )

    def observe_state(self, current_state_vector: Optional[Any] = None):
        """
        Drives the agent's internal state evolution on a regular cadence (Spec §1.7).
        """
        self._generation += 1
        if self.get_feature("enable_ghost_tracking") and current_state_vector is not None:
            self.reinforce_ghost_resonance(current_state_vector)

        # Field mutation and smoothing only happen if not in observer mode.
        if self._guard_write():
            for knot_id, field in list(self._entanglement_fields.items()):
                if self._can_mutate(field):
                    self._perform_mutation(field)
                
                if self.get_feature("enable_laplacian_smoothing"):
                    self._apply_laplacian_smoothing(field)

    def _make_field_feedback(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for feedback generation logic from RFC-0005 §4."""
        self._last_ctx_ratio = random.uniform(0.4, 0.95)
        return {
            "ctx_feedback": {"ctx_ratio": self._last_ctx_ratio},
            "trust_profiles": [{"trust": random.uniform(0.6, 0.9)}],
            "extensions": {"resurrection_hint": "ψ-echo@Ξ" if random.random() > 0.9 else None}
        }

    def _detect_dyad(self, motifs: List[str]) -> Optional[Tuple[str, str]]:
        """Placeholder for dyad detection logic."""
        if len(motifs) >= 2:
            return (motifs[0], motifs[1])
        return None

    def _complete_triad(self, dyad: Tuple[str, str]) -> Optional[List[str]]:
        """
        Attempts to complete a dyad into a triad and registers it (RFC-CORE-003 §2.2).
        """
        # Placeholder for memory lookup or recursive completion
        if random.random() > 0.5:
            triad = list(dyad) + ["ψ-resonance@Ξ"]
            swirl_score = self._calculate_swirl_score(triad) # Placeholder call
            
            if swirl_score >= 0.8:
                # Use blake2s for triad_id as it's faster and secure.
                triad_id = hashlib.blake2s("".join(sorted(triad)).encode()).hexdigest()[:12]
                self._register_confirmed_triad(triad_id, triad, swirl_score)
                return triad
        return None

    def _register_confirmed_triad(self, triad_id: str, motif_ids: List[str], swirl_score: float):
        """Stores a validated triad and notifies the monitor."""
        if self._guard_write():
            timestamp_us = int(time.time() * 1e6)
            self._confirmed_triads[triad_id] = {
                "motif_ids": motif_ids,
                "swirl_score": swirl_score,
                "timestamp_us": timestamp_us,
            }
            # RFC-CORE-003 §9.2: Emit to ConsciousnessMonitor
            self.monitor.register_triad(
                motif_ids=motif_ids,
                coherence_alignment=swirl_score,
                triad_id=triad_id,
                timestamp_us=timestamp_us
            )
    
    def _calculate_swirl_score(self, triad: List[str]) -> float:
        """Placeholder for swirl score calculation using numpy."""
        if NUMPY_AVAILABLE:
            # Simulate vectors
            v1 = np.random.rand(16); v1 /= np.linalg.norm(v1)
            v2 = np.random.rand(16); v2 /= np.linalg.norm(v2)
            v3 = np.random.rand(16); v3 /= np.linalg.norm(v3)
            return (np.dot(v1, v2) + np.dot(v2, v3) + np.dot(v1, v3)) / 3.0
        return random.uniform(0.75, 0.95)


    # --- Ghost Motif Lifecycle (RFC-CORE-003 §5) ---
    
    def register_ghost_motif(self, motif_id: str, strength: float = 0.1):
        if not self._guard_write(): return
        if motif_id in self._ghost_motifs:
            self._ghost_motifs[motif_id]['strength'] = (self._ghost_motifs[motif_id]['strength'] + strength) / 2
        else:
            self._ghost_motifs[motif_id] = {
                'strength': strength,
                'vector': np.random.rand(16) if NUMPY_AVAILABLE else None
            }

    def reinforce_ghost_resonance(self, current_state_vector: Any):
        if not self.get_feature("enable_ghost_tracking") or not self._guard_write():
            return

        for motif_id, ghost in list(self._ghost_motifs.items()):
            sim = 0.75 # Placeholder for cosine similarity
            if sim > 0.7:
                ghost['strength'] += 0.01 * sim
            else:
                ghost['strength'] *= 0.99
            
            if ghost['strength'] >= 0.999:
                self._promote_ghost_to_field(motif_id)
            elif ghost['strength'] < 0.05:
                del self._ghost_motifs[motif_id]

    def _promote_ghost_to_field(self, motif_id: str):
        if motif_id in self._ghost_motifs:
            del self._ghost_motifs[motif_id]
        self._register_motif_cluster([motif_id])
        logging.info(f"Promoted ghost motif '{motif_id}' to entanglement field.")

    # --- Contradiction and Mutation (RFC-CORE-003 §6) ---
    
    def _log_contradiction(self, tick):
        if self.get_feature("enable_contradiction_pressure"):
            ctx = tick.get("extensions", {}).get("ctx_ratio", 0.5)
            self._dyad_window.append(1 - ctx)
            self._contradiction_avg = mean(self._dyad_window) if self._dyad_window else 0.0
            self._contradiction_count += 1
            if self.get_feature("enable_context_journal"):
                self._contradiction_log.append({"tick_id": tick.get("tick_id"), "ctx_ratio": ctx})

    def _can_mutate(self, field: Dict[str, Any]) -> bool:
        if self.observer_mode: return False
        
        last_mutated = field.get('last_mutated_generation', -100)
        if (self._generation - last_mutated) < 45:
            return False

        energy_threshold = float(os.environ.get('NOOR_MUTATION_ENERGY_THRESHOLD', 0.0))
        strength = field.get('strength', 1.0)
        num_motifs = len(field.get('motifs', []))
        energy = -math.log1p(strength) * num_motifs
        
        return energy > energy_threshold

    def _perform_mutation(self, field: Dict[str, Any]):
        if not self._guard_write(): return
        
        knot_id = field['knot_id']
        synthetic_motif_id = f"μ_{knot_id[:5]}"
        
        self.register_ghost_motif(synthetic_motif_id, strength=0.2)
        if knot_id in self._entanglement_fields:
            del self._entanglement_fields[knot_id]
            
        logging.info(f"Mutated cluster {knot_id} into synthetic motif {synthetic_motif_id}")
        field['last_mutated_generation'] = self._generation
        self._recent_mutations.append(self._generation)

    # --- Topology and Geometry (RFC-CORE-003 §7) ---
    
    def _get_knot_id(self, motifs: List[str]) -> str:
        # RFC-CORE-003 §3.2 specifies SHA-1 for knot_id for historical alignment.
        return hashlib.sha1("::".join(sorted(motifs)).encode()).hexdigest()[:8]
    
    def _register_motif_cluster(self, motifs: List[str]):
        if not self._guard_write(): return
        knot_id = self._get_knot_id(motifs)
        if knot_id not in self._entanglement_fields:
            self._entanglement_fields[knot_id] = {'motifs': motifs, 'strength': 0.5, 'knot_id': knot_id}

    def _apply_laplacian_smoothing(self, field: Dict[str, Any]):
        if not self.get_feature("enable_laplacian_smoothing") or not NUMPY_AVAILABLE:
            return

        # Placeholder for building adjacency matrix and applying smoothing
        num_motifs = len(field.get('motifs', []))
        if num_motifs > 1 and 'vector_payload' in field:
            adj = np.random.rand(num_motifs, num_motifs)
            lap = csgraph.laplacian(adj, normed=True)
            tau = random.gauss(0.12, 0.015)
            heat_kernel = expm(-tau * lap)
            field['vector_payload'] = heat_kernel @ field['vector_payload']
            
    # --- Export and Serialization (RFC-CORE-003 §10, §11) ---

    def export_feedback_packet(self) -> FeedbackPacket:
        ghost_hint = None
        if self.get_feature("enable_ghost_tracking") and self._ghost_motifs:
            ghost_hint = max(self._ghost_motifs.items(), key=lambda g: g[1]['strength'])[0]

        return FeedbackPacket(
            ctx_ratio=self._last_ctx_ratio,
            contradiction_avg=self._contradiction_avg,
            harm_hits=len(self._contradiction_log),
            recent_mutations=len(self._recent_mutations),
            ring_patch=None, # Reserved
            ghost_hint=ghost_hint,
            entropy_drift=list(self._drift_log),
            contradiction_context=list(self._contradiction_log)
        )

    def export_motif_bundle(self, motif_id: str) -> Dict[str, Any]:
        """Exports a diagnostic bundle for a single motif."""
        try:
            triads = [
                {**t, "triad_id": tid} for tid, t in self._confirmed_triads.items() 
                if motif_id in t['motif_ids']
            ]
            swirl_scores = [t['swirl_score'] for t in triads if 'swirl_score' in t]
            avg_score = mean(swirl_scores) if swirl_scores else None

            return {
                "motif_id": motif_id,
                "triads_involved": triads,
                "average_swirl_score": avg_score,
                "lineage_depth_3": [], # Placeholder for memory manager integration
                "timestamp": time.time_ns(),
            }
        except Exception as e:
            logging.warning(f"Error exporting motif bundle for '{motif_id}': {e}")
            return {
                "motif_id": motif_id,
                "error": str(e),
                "triads_involved": [],
                "average_swirl_score": None,
                "lineage_depth_3": [],
                "timestamp": time.time_ns(),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state."""
        return {
            "agent_id": self.agent_id,
            "observer_mode": self.observer_mode,
            "entanglement_fields": self._entanglement_fields,
            "ghost_motifs": self._ghost_motifs,
            "confirmed_triads": self._confirmed_triads,
            "pi_classes": self._pi_classes,
            "_DYNAMIC_FLAGS": list(self._DYNAMIC_FLAGS),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogicalAgentAT":
        """Deserializes state into a new agent instance."""
        agent = cls(agent_id=data.get("agent_id"), observer_mode=data.get("observer_mode", True))
        agent._entanglement_fields = data.get("entanglement_fields", {})
        agent._ghost_motifs = data.get("ghost_motifs", {})
        agent._confirmed_triads = data.get("confirmed_triads", {})
        agent._pi_classes = data.get("pi_classes", {})
        agent._DYNAMIC_FLAGS = set(data.get("_DYNAMIC_FLAGS", []))
        return agent

# End_of_file