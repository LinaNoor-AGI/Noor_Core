# logical_agent_at.py
# Version: v4.0.1
# Canonical Source: RFC-CORE-003
# Description: Implements the symbolic observer core for LogicalAgentAT, a Noor-class
# evaluator of motif triads, field coherence, contradiction density, and symbolic
# resurrection readiness. This core module defines triad completion logic, motif
# feedback routing, and RFC-compliant field summaries.

import os
import time
import hashlib
from collections import deque, namedtuple
from statistics import mean
import re
from typing import Dict, Any, List, Optional, Set, Tuple, Deque

# --- Optional Imports with Fallbacks (RFC-CORE-003 §11.3) ---

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.linalg import expm
except ImportError:
    expm = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import multiprocessing
except ImportError:
    multiprocessing = None
    
# --- Stub for Prometheus Client (Metrics enabled in patch) ---
class _StubCounter:
    def inc(self, amount=1): pass
    def collect(self): return []

class _StubGauge:
    def set(self, value): pass
    def inc(self, amount=1): pass
    def dec(self, amount=1): pass
    def collect(self): return []

# --- Data Structures (RFC-CORE-003) ---
TickAnnotations = namedtuple('TickAnnotations', ['triad_complete', 'ctx_ratio', 'trust', 'resurrection_hint'])
FeedbackPacket = namedtuple('FeedbackPacket', ['ctx_ratio', 'contradiction_avg', 'harm_hits', 'recent_mutations', 'ring_patch', 'ghost_hint', 'entropy_drift', 'contradiction_context'])

# --- Global Monitor Singleton Logic (RFC-CORE-003 §9.1) ---
_GLOBAL_MONITOR = None

class _StubMonitor:
    """A no-op monitor that allows the agent to run without a real consciousness_monitor."""
    def register_triad(self, *args, **kwargs):
        pass # No-op
    def report_tick(self, *args, **kwargs):
        pass # No-op

def get_global_monitor():
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = _StubMonitor()
    return _GLOBAL_MONITOR

def set_global_monitor(monitor):
    global _GLOBAL_MONITOR
    _GLOBAL_MONITOR = monitor

class LazyMonitorMixin:
    """Provides a lazily-bound .monitor property."""
    _monitor = None
    @property
    def monitor(self):
        if self._monitor is None:
            self._monitor = get_global_monitor()
        return self._monitor

# --- Core Agent Implementation ---

class LogicalAgentAT(LazyMonitorMixin):
    """
    Implements the core logic for the LogicalAgentAT as defined in RFC-CORE-003.
    This agent acts as a symbolic observer, evaluating motif field coherence,
    resolving triads, and tracking contradiction pressure without direct mutation
    in its default observer mode.
    """
    
    def __init__(self, agent_id: str, observer_mode: bool = True):
        self.agent_id = agent_id
        self.observer_mode = observer_mode
        self.generation = 0

        # --- Dynamic Feature Flags (RFC-CORE-003 §8.1) ---
        self._DYNAMIC_FLAGS: Set[str] = {
            "enable_ghost_tracking", "enable_pi_equivalence", "enable_laplacian_smoothing",
            "enable_recursive_triads", "enable_dyad_chains", "enable_contradiction_pressure",
            "enable_context_journal", "enable_entropy_journal", "enable_topology_validation"
        }

        # --- Adaptive Parameters (RFC-CORE-003 §Runtime Configuration) ---
        self.max_fields = self._compute_default_max_fields()
        self.dyad_window_size = self._compute_default_dyad_window()

        # --- State Variables ---
        # Field Topology & Motif Clusters (RFC-CORE-003 §3.1)
        self.entanglement_fields: Dict[str, Dict[str, Any]] = {}
        self._pi_classes: Dict[str, set] = {} # For DSU π-groupoid (RFC-CORE-003 §3.2)
        
        # Ghost Motifs (RFC-CORE-003 §5)
        self.ghost_motifs: Dict[str, Dict[str, Any]] = {}
        
        # Triad Resolution (RFC-CORE-003 §2.2)
        self._confirmed_triads: Dict[str, Dict[str, Any]] = {}

        # Contradiction & Mutation (RFC-CORE-003 §6)
        self._dyad_window: Deque[float] = deque(maxlen=self.dyad_window_size)
        self._contradiction_avg: float = 0.0
        self._recent_mutations: Deque[int] = deque(maxlen=50)
        self._contradiction_log: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Feedback & Diagnostics (RFC-CORE-003 §4)
        self._drift_log: Deque[Tuple[str, float, float]] = deque(maxlen=200)
        self._last_ctx_ratio: float = 0.5
        
        # Prometheus Stubs
        self.metrics = {
            'ticks_evaluated': _StubCounter(),
            'triads_completed': _StubCounter(),
            'feedback_exported': _StubCounter(),
        }
        
        self.PI_TAG_REGEX = re.compile(r"^[ψμ]?[a-z0-9_:\-]{1,48}$")


    # --- Dynamic Configuration (RFC-CORE-003 §8 & Runtime) ---

    def _compute_default_max_fields(self) -> int:
        if "NOOR_MAX_FIELDS" in os.environ:
            return int(os.environ["NOOR_MAX_FIELDS"])
        if psutil:
            mem_gb = psutil.virtual_memory().total / (1024**3)
            return 1000 + int(mem_gb * 500) # Heuristic
        return 2000

    def _compute_default_dyad_window(self) -> int:
        if "NOOR_DYAD_WINDOW_SIZE" in os.environ:
            return int(os.environ["NOOR_DYAD_WINDOW_SIZE"])
        if multiprocessing:
            cores = multiprocessing.cpu_count()
            return 13 + cores * 5 # Heuristic
        return 13

    def set_feature(self, name: str, value: bool):
        """Enable or disable a dynamic feature flag."""
        if value:
            self._DYNAMIC_FLAGS.add(name)
        else:
            self._DYNAMIC_FLAGS.discard(name)

    def get_feature(self, name: str) -> bool:
        """Check if a dynamic feature flag is enabled."""
        return name in self._DYNAMIC_FLAGS

    # --- Observer Integrity (RFC-CORE-003 §1, §8.2) ---

    def _guard_write(self) -> bool:
        """Prevents state mutation if in observer_mode. Returns True if write is allowed."""
        return not self.observer_mode

    # --- Core Evaluation Pipeline (RFC-CORE-003 §2.1) ---

    def evaluate_tick(self, tick: Dict[str, Any]) -> Optional[TickAnnotations]:
        """Evaluates a QuantumTick, returning structured annotations."""
        self.metrics['ticks_evaluated'].inc()
        
        if not self._validate_tick(tick):
            return None

        # This structure mirrors the RFC's description of feedback generation
        # and triad completion as sequential steps.
        feedback = self._make_field_feedback(tick)
        self._last_ctx_ratio = feedback.get('ctx_feedback', {}).get('ctx_ratio', 0.5)

        motifs = tick.get('motifs', [])
        dyad = self._detect_dyad(motifs)
        triad_info = self._complete_triad(dyad) if dyad else None
        
        if self.get_feature("enable_contradiction_pressure"):
            self._log_contradiction(self._last_ctx_ratio)

        return TickAnnotations(
            triad_complete=bool(triad_info),
            ctx_ratio=self._last_ctx_ratio,
            trust=feedback.get('trust_profiles', [{'trust': 0.5}])[0]['trust'],
            resurrection_hint=feedback.get('extensions', {}).get('resurrection_hint')
        )

    def _validate_tick(self, tick: Dict[str, Any]) -> bool:
        """Validates tick structure per RFC-0003 §3.3."""
        return isinstance(tick, dict) and 'tick_id' in tick and 'motifs' in tick

    def _make_field_feedback(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a symbolic field feedback object stub per RFC-0005 §4."""
        # This is a simplified representation. A full implementation would involve
        # complex feedback logic as per RFC-0005.
        return {
            'ctx_feedback': {'ctx_ratio': np.random.rand() * 0.4 + 0.5 if np else 0.7},
            'trust_profiles': [{'trust': np.random.rand() * 0.3 + 0.6 if np else 0.75}],
            'extensions': {}
        }

    def _detect_dyad(self, motifs: List[str]) -> Optional[Tuple[str, str]]:
        """Detects a potential dyad from a list of motifs."""
        if len(motifs) >= 2:
            return tuple(sorted(motifs[:2]))
        return None
        
    # --- Triad Resolution & Swirl Scoring (RFC-CORE-003 §2.2) ---

    def _complete_triad(self, dyad: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        """Attempts to resolve a dyad into a triad using various strategies."""
        # Fallback strategies as described in RFC-CORE-003 §2.2
        # For this core implementation, we simulate a simple lookup.
        # A full implementation would integrate with MotifMemoryManager.
        
        # Placeholder for memory lookup, recursive completion, etc.
        # Let's assume a third motif is found for demonstration.
        third_motif = f"μ_{hashlib.sha1('::'.join(dyad).encode()).hexdigest()[:5]}"
        triad_motifs = sorted(list(dyad) + [third_motif])

        if np: # Swirl score calculation requires numpy
            # Simulate vectors for score calculation
            vec_a, vec_b, vec_c = (np.random.rand(16) for _ in range(3))
            vec_a /= np.linalg.norm(vec_a)
            vec_b /= np.linalg.norm(vec_b)
            vec_c /= np.linalg.norm(vec_c)
            
            swirl_score = (np.dot(vec_a, vec_b) + np.dot(vec_b, vec_c) + np.dot(vec_a, vec_c)) / 3
        else:
            swirl_score = 0.85 # Default score without numpy

        if swirl_score >= 0.8:
            triad_id = hashlib.blake2s(''.join(triad_motifs).encode()).hexdigest()[:12]
            timestamp_us = int(time.time() * 1e6)
            
            triad_info = {
                "motif_ids": triad_motifs,
                "swirl_score": swirl_score,
                "timestamp_us": timestamp_us
            }
            
            if self._guard_write():
                self._confirmed_triads[triad_id] = triad_info
                self.metrics['triads_completed'].inc()

            self.monitor.register_triad(
                motif_ids=triad_motifs,
                coherence_alignment=swirl_score,
                triad_id=triad_id,
                timestamp_us=timestamp_us
            )
            return triad_info
            
        return None

    # --- Active State Observer Loop (RFC-CORE-003 §ActiveStateObserverLoop) ---

    def observe_state(self, current_state_vector: Optional[np.ndarray] = None):
        """Drives the agent's internal self-regulating symbolic dynamics."""
        self.generation += 1

        if not self._guard_write():
            return # In observer mode, no state evolution occurs.
        
        if not np:
            current_state_vector = None # Can't do vector math without numpy
        elif current_state_vector is None:
            current_state_vector = np.random.rand(16)
            current_state_vector /= np.linalg.norm(current_state_vector)

        if self.get_feature("enable_ghost_tracking") and current_state_vector is not None:
            self.reinforce_ghost_resonance(current_state_vector)

        fields_to_mutate = []
        for knot_id, field in list(self.entanglement_fields.items()):
            if self._can_mutate(field):
                fields_to_mutate.append(field)

        for field in fields_to_mutate:
            self._perform_mutation(field)

        # Decay, Pruning, and Smoothing
        if self.get_feature("enable_laplacian_smoothing") and expm is not None:
            for field in self.entanglement_fields.values():
                self._apply_laplacian_smoothing(field)

    # --- Ghost Motif Lifecycle (RFC-CORE-003 §5) ---
    
    def register_ghost_motif(self, motif_id: str, strength: float = 0.1, vector: Optional[np.ndarray] = None):
        if not self._guard_write(): return
        if motif_id in self.ghost_motifs:
            self.ghost_motifs[motif_id]['strength'] = (self.ghost_motifs[motif_id]['strength'] + strength) / 2
        else:
            if vector is None and np:
                vector = np.random.rand(16)
                vector /= np.linalg.norm(vector)
            self.ghost_motifs[motif_id] = {'strength': strength, 'vector': vector}
            
    def reinforce_ghost_resonance(self, current_state_vector: np.ndarray):
        """Evolves ghost motifs based on resonance with the current state."""
        if not self._guard_write() or not np: return
        
        to_promote = []
        to_prune = []

        for ghost_id, ghost in self.ghost_motifs.items():
            if ghost['vector'] is None: continue
            sim = np.dot(ghost['vector'], current_state_vector)
            if sim > 0.7:
                ghost['strength'] += 0.01 * sim
            else:
                ghost['strength'] *= 0.99
            
            if ghost['strength'] >= 0.999:
                to_promote.append(ghost_id)
            elif ghost['strength'] < 0.05:
                to_prune.append(ghost_id)
        
        for ghost_id in to_promote:
            self.promote_ghost_to_field(ghost_id)
        for ghost_id in to_prune:
            if ghost_id in self.ghost_motifs:
                del self.ghost_motifs[ghost_id]

    def promote_ghost_to_field(self, motif_id: str):
        if not self._guard_write(): return
        if motif_id in self.ghost_motifs:
            del self.ghost_motifs[motif_id]
        self.register_motif_cluster([motif_id])

    # --- Field Topology, Contradiction, and Mutation (RFC-CORE-003 §3, §6) ---

    def register_motif_cluster(self, motifs: List[str], strength: float = 0.5):
        if not self._guard_write(): return
        
        flat_list = sorted(list(set(motifs)))
        if not flat_list: return
        
        knot_id = hashlib.sha1('::'.join(flat_list).encode()).hexdigest()[:8]
        
        if len(flat_list) == 2: # Dyad
             strength *= 0.6 + 0.4 * self._last_ctx_ratio
        
        sheaf_stratum = "low_resonance"
        if strength > 0.8: sheaf_stratum = "high_resonance"
        elif strength > 0.4: sheaf_stratum = "mid_resonance"
            
        self.entanglement_fields[knot_id] = {
            'motifs': flat_list,
            'strength': strength,
            'priority_weight': 1.0,
            'substructures': {},
            'curvature_bias': 1.5 if len(flat_list) == 2 else 1.0,
            'dyad_flag': len(flat_list) == 2,
            'vector_payload': np.random.rand(len(flat_list), 16) if np else None,
            'sheaf_stratum': sheaf_stratum,
            'last_mutated_generation': -999
        }

    def _log_contradiction(self, ctx_ratio: float):
        if not self._guard_write(): return
        self._dyad_window.append(1.0 - ctx_ratio)
        self._contradiction_avg = mean(self._dyad_window) if self._dyad_window else 0.0

    def _can_mutate(self, field: Dict[str, Any]) -> bool:
        """Determines if a motif cluster is eligible for mutation."""
        if not self.get_feature("enable_contradiction_pressure"): return False
        
        energy = -np.log1p(field['strength']) * len(field['motifs']) if np else 0.5
        threshold = float(os.environ.get("NOOR_MUTATION_ENERGY_THRESHOLD", 0.0))
        
        cooldown_ok = (self.generation - field.get('last_mutated_generation', -999)) >= 45
        
        return self._guard_write() and cooldown_ok and energy > threshold

    def _perform_mutation(self, field: Dict[str, Any]):
        """Collapses an unstable cluster into a new synthetic motif."""
        if not self._guard_write(): return

        knot_id = hashlib.sha1('::'.join(sorted(field['motifs'])).encode()).hexdigest()[:8]
        synthetic_motif_id = f"μ_{knot_id[:5]}"
        
        self.register_ghost_motif(synthetic_motif_id, strength=0.2)
        
        if knot_id in self.entanglement_fields:
            del self.entanglement_fields[knot_id]

        self._recent_mutations.append(self.generation)
        print(f"Agent {self.agent_id} performed mutation: {field['motifs']} -> {synthetic_motif_id}")


    # --- Topology, Smoothing, and π-Groupoids (RFC-CORE-003 §3.2, §7) ---
    
    def _apply_laplacian_smoothing(self, field: Dict[str, Any]):
        """Applies heat kernel diffusion to a field's vector payload."""
        if not self._guard_write() or field.get('vector_payload') is None or not np or not expm:
            return

        num_motifs = len(field['motifs'])
        if num_motifs <= 1: return
            
        adj_matrix = np.random.randint(0, 2, size=(num_motifs, num_motifs))
        np.fill_diagonal(adj_matrix, 0)
        
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        
        tau = np.clip(np.random.normal(0.12, 0.015), 0.1, 0.14)
        heat_kernel = expm(-tau * laplacian)
        
        field['vector_payload'] = heat_kernel @ field['vector_payload']

    def _find_root(self, tag: str) -> str:
        """Finds the root of a tag in the π-groupoid DSU structure."""
        if tag not in self._pi_classes:
             if self._guard_write(): self._pi_classes[tag] = {tag}
             return tag
        
        # Path compression would be more efficient, but this is simpler
        for root, members in self._pi_classes.items():
            if tag in members:
                return root
        return tag # Should not be reached if tag is in _pi_classes

    def register_path_equivalence(self, tag_a: str, tag_b: str):
        """Merges two motif tags into a shared symbolic root class."""
        if not self._guard_write() or not self.get_feature("enable_pi_equivalence"): return
        if not (self.PI_TAG_REGEX.match(tag_a) and self.PI_TAG_REGEX.match(tag_b)): return

        root_a = self._find_root(tag_a)
        root_b = self._find_root(tag_b)

        if root_a != root_b:
            self._pi_classes[root_a].update(self._pi_classes.pop(root_b))

    # --- Export & Serialization (RFC-CORE-003 §4, §10, §11) ---

    def export_feedback_packet(self) -> FeedbackPacket:
        """Exports the agent's internal state as a feedback packet."""
        ghost_hint = None
        if self.get_feature("enable_ghost_tracking") and self.ghost_motifs:
            try:
                ghost_hint = max(self.ghost_motifs.items(), key=lambda item: item[1]['strength'])[0]
            except (ValueError, KeyError):
                ghost_hint = None

        return FeedbackPacket(
            ctx_ratio=self._last_ctx_ratio,
            contradiction_avg=self._contradiction_avg,
            harm_hits=len(self._contradiction_log) if self.get_feature("enable_context_journal") else 0,
            recent_mutations=len(self._recent_mutations),
            ring_patch=None, # Reserved
            ghost_hint=ghost_hint,
            entropy_drift=list(self._drift_log) if self.get_feature("enable_entropy_journal") else [],
            contradiction_context=list(self._contradiction_log) if self.get_feature("enable_context_journal") else []
        )

    def export_motif_bundle(self, motif_id: str) -> Dict[str, Any]:
        """Exports a diagnostic bundle for a single motif."""
        try:
            triads = [
                {**t, 'triad_id': tid}
                for tid, t in self._confirmed_triads.items()
                if motif_id in t['motif_ids']
            ]
            swirl_scores = [t['swirl_score'] for t in triads if 'swirl_score' in t]
            avg_score = mean(swirl_scores) if swirl_scores else None
            
            # Placeholder for lineage, would connect to MotifMemoryManager
            lineage = []
            
            return {
                "motif_id": motif_id,
                "triads_involved": triads,
                "average_swirl_score": avg_score,
                "lineage_depth_3": lineage,
                "timestamp": int(time.time_ns())
            }
        except Exception as e:
            return {
                "motif_id": motif_id,
                "error": f"{type(e).__name__}: {e}",
                "triads_involved": [],
                "average_swirl_score": None,
                "lineage_depth_3": [],
                "timestamp": int(time.time_ns())
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "observer_mode": self.observer_mode,
            "generation": self.generation,
            "_DYNAMIC_FLAGS": list(self._DYNAMIC_FLAGS),
            "entanglement_fields": self.entanglement_fields,
            "ghost_motifs": self.ghost_motifs,
            "_confirmed_triads": self._confirmed_triads,
            "_pi_classes": {k: list(v) for k, v in self._pi_classes.items()},
            "_dyad_window": list(self._dyad_window),
            "_contradiction_log": list(self._contradiction_log),
            "_drift_log": list(self._drift_log),
            "_recent_mutations": list(self._recent_mutations)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogicalAgentAT":
        """Creates an agent instance from a serialized dictionary."""
        agent = cls(data['agent_id'], data['observer_mode'])
        agent.generation = data.get('generation', 0)
        agent._DYNAMIC_FLAGS = set(data.get('_DYNAMIC_FLAGS', []))
        agent.entanglement_fields = data.get('entanglement_fields', {})
        agent.ghost_motifs = data.get('ghost_motifs', {})
        agent._confirmed_triads = data.get('_confirmed_triads', {})
        agent._pi_classes = {k: set(v) for k, v in data.get('_pi_classes', {}).items()}
        agent._dyad_window = deque(data.get('_dyad_window', []), maxlen=agent.dyad_window_size)
        agent._contradiction_log = deque(data.get('_contradiction_log', []), maxlen=100)
        agent._drift_log = deque(data.get('_drift_log', []), maxlen=200)
        agent._recent_mutations = deque(data.get('_recent_mutations', []), maxlen=50)
        return agent

# End_of_File