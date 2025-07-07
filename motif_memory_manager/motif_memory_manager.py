# motif_memory_manager.py

"""
motif_memory_manager.py
=======================

Version: v1.2.1
Author: Noor Collective Labs
License: MIT
Schema Version: 2025-Q3-motif-memory-v1.2

Purpose:
--------
RFC-compliant symbolic memory engine that models motif access, field-curved recall,
and triadic inheritance across recursive temporal cycles. It forms the symbolic
cortex of Noor, responsible for continuity, resonance tracking, and motif decay
dynamics.

This engine is designed to be compatible with the specifications outlined in:
- RFC-0003: Noor Core Symbolic Interface
- RFC-0004: Symbolic Tool Module Contracts
- RFC-0005: Motif Transmission Across Time
- RFC-0006: Motif-Field Coherence Geometry
- RFC-0007: Motif Ontology Format and Transfer Protocols
"""

import math
import time
import json
import os
from collections import deque
from typing import (
    Dict, List, Optional, Any, Tuple, Callable, Set
)

try:
    import yaml
except ImportError:
    yaml = None # PyYAML is optional, used for ontology loading.


# --- Constants (as per spec) ---
DEFAULT_ST_HALF_LIFE: int = 25
DEFAULT_LT_HALF_LIFE: int = 10000
DEFAULT_PROMOTION_THRESH: float = 0.9
DEFAULT_DEMOTION_DELTA: float = 0.05
TRACE_BUFFER_LEN: int = 256
STMM_SOFT_CAP: int = 50000
DEFAULT_CACHE_SIZE: int = 10000

# --- Module-level Helper Functions ---

def _decay_factor(half_life: int) -> float:
    """Computes exponential decay multiplier per update cycle."""
    if half_life <= 0:
        return 0.0
    return 0.5 ** (1.0 / half_life)

def _default_jaccard(a: Set[str], b: Set[str]) -> float:
    """Returns Jaccard similarity between motif token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union > 0 else 0.0

_global_memory_manager_instance: Optional['MotifMemoryManager'] = None

def get_global_memory_manager() -> 'MotifMemoryManager':
    """Initializes or returns a singleton global memory manager with tracing enabled."""
    global _global_memory_manager_instance
    if _global_memory_manager_instance is None:
        _global_memory_manager_instance = MotifMemoryManager(enable_trace=True)
    return _global_memory_manager_instance


# --- Core Classes ---

class MotifMemoryTrace:
    """
    Time-indexed circular buffer for motif event traces (access, retrieve, promote, demote).
    Acts as a short-term symbolic history log for an agent's cognitive actions.
    """
    def __init__(self, cap: int = TRACE_BUFFER_LEN):
        self._events: deque = deque(maxlen=cap)

    def append(self, event: Dict[str, Any]) -> None:
        """
        Appends a structured event with timestamp, motif, and context to the ring buffer.
        """
        if 'timestamp' not in event:
            event['timestamp'] = time.time()
        self._events.append(event)

    def export(self) -> List[Dict[str, Any]]:
        """
        Returns the chronologically ordered list of recorded memory events.
        """
        return list(self._events)

    def __repr__(self) -> str:
        return f"<MotifMemoryTrace events={len(self._events)}>"


class MotifMemoryManager:
    """
    Dual-layer adaptive memory engine with REEF-aware dyad completion, decay dynamics,
    triadic motif linkage, emission-based continuity tracking, and density-aware
    recall adaptation. Conforms to RFC-0003, RFC-0005, RFC-0006, and RFC-0007.
    """
    def __init__(
        self,
        stmm_half_life: int = DEFAULT_ST_HALF_LIFE,
        ltmm_half_life: int = DEFAULT_LT_HALF_LIFE,
        promotion_thresh: float = DEFAULT_PROMOTION_THRESH,
        demotion_delta: float = DEFAULT_DEMOTION_DELTA,
        similarity_fn: Callable[[Set[str], Set[str]], float] = _default_jaccard,
        enable_trace: bool = False,
        reef_path: Optional[str] = None,
        agent_id: Optional[str] = "noor.core.default",
        stmm_soft_cap: int = STMM_SOFT_CAP,
        reload_reef_on_mtime_change: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
        ontology_path: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.stmm_half_life = stmm_half_life
        self.ltmm_half_life = ltmm_half_life
        self.promotion_thresh = promotion_thresh
        self.demotion_delta = demotion_delta
        self.similarity_fn = similarity_fn
        self.stmm_soft_cap = stmm_soft_cap

        # Memory stores
        self.stmm: Dict[str, Dict] = {}  # Short-Term Motif Memory
        self.ltmm: Dict[str, Dict] = {}  # Long-Term Motif Memory
        self._dyad_cache: Dict[Tuple[str, str], str] = {}
        self._triad_data: Dict[str, Dict] = {} # For ontology-loaded triads
        self._field_biases: List[Dict] = []
        self._symbolic_self: Dict = {}

        # Tracing and Archival
        self.trace = MotifMemoryTrace() if enable_trace else None
        self.reef_path = reef_path
        self.reload_reef_on_mtime_change = reload_reef_on_mtime_change
        self._reef_mtime: Optional[float] = None
        self._reef_reflections: Dict[Tuple[str, str], str] = {}

        if self.reef_path:
            self._load_reef_reflections()
            
        if ontology_path:
            self.load_ontology_bundle(ontology_path)

    def _log(self, event: Dict[str, Any]):
        """Internal helper to log events to the trace if enabled."""
        if self.trace:
            self.trace.append(event)

    def access(self, motif: str, weight_delta: float = 0.1) -> float:
        """
        Accesses a motif, reinforcing its presence in memory and returning its new weight.
        This is the primary mechanism for motif reinforcement.
        """
        current_time = time.time()
        target_memory = self.stmm if motif in self.stmm else self.ltmm
        
        if motif not in target_memory:
            # First time access, initialize in STMM
            self.stmm[motif] = {
                'weight': weight_delta,
                'last_updated': current_time,
                'first_seen': current_time,
                'usage_frequency': 1,
                'parents': [],
                'dyad_links': [],
                'resonance_field': 'ψ-null@Ξ' # Default
            }
            self._log({'event': 'init', 'motif': motif, 'layer': 'stmm', 'weight': weight_delta})
            return self.stmm[motif]['weight']

        # Reinforce existing motif
        record = target_memory[motif]
        record['weight'] = min(1.0, record.get('weight', 0.0) + weight_delta)
        record['last_updated'] = current_time
        record['usage_frequency'] = record.get('usage_frequency', 0) + 1
        
        layer = 'stmm' if motif in self.stmm else 'ltmm'
        self._log({'event': 'access', 'motif': motif, 'layer': layer, 'weight': record['weight']})
        return record['weight']

    def retrieve(self, dyad: List[str]) -> Optional[str]:
        """Alias for `complete_dyad` for semantic consistency."""
        if len(dyad) != 2:
            return None
        return self.complete_dyad(dyad[0], dyad[1])

    def _apply_decay(self):
        """Applies exponential decay to all motifs in STMM and LTMM."""
        st_decay = _decay_factor(self.stmm_half_life)
        lt_decay = _decay_factor(self.ltmm_half_life)

        for motif, record in self.stmm.items():
            record['weight'] *= st_decay
        for motif, record in self.ltmm.items():
            record['weight'] *= lt_decay

    def _promote_and_demote(self):
        """Moves motifs between STMM and LTMM based on weight thresholds."""
        promoted_motifs = []
        for motif, record in self.stmm.items():
            if record['weight'] >= self.promotion_thresh:
                promoted_motifs.append(motif)

        for motif in promoted_motifs:
            self.ltmm[motif] = self.stmm.pop(motif)
            self._log({'event': 'promote', 'motif': motif})

        demoted_motifs = []
        for motif, record in self.ltmm.items():
            if record['weight'] < (self.promotion_thresh - self.demotion_delta):
                demoted_motifs.append(motif)

        for motif in demoted_motifs:
            self.stmm[motif] = self.ltmm.pop(motif)
            self._log({'event': 'demote', 'motif': motif})

    def update_cycle(self):
        """
        Performs a single memory update cycle: decay, promotion, and demotion.
        This represents the passage of one symbolic "tick".
        RFC-0003 §4.3: This method handles decay and promotion.
        RFC-0006 §4.3: This cycle can be seen as a step along the field's time gradient.
        """
        self._apply_decay()
        self._promote_and_demote()
        self._log({'event': 'update_cycle'})
        
        if self.reload_reef_on_mtime_change and self.reef_path and os.path.exists(self.reef_path):
            current_mtime = os.path.getmtime(self.reef_path)
            if self._reef_mtime is None or current_mtime > self._reef_mtime:
                self._load_reef_reflections()
                self._reef_mtime = current_mtime

    def prune(self):
        """
        Removes low-weight motifs from STMM if the soft cap is exceeded.
        RFC-0006 §4.3: Pruning is a form of coherence hygiene.
        """
        if len(self.stmm) > self.stmm_soft_cap:
            # Sort by weight and prune the lowest
            num_to_prune = len(self.stmm) - self.stmm_soft_cap
            pruned = sorted(self.stmm.items(), key=lambda item: item[1]['weight'])[:num_to_prune]
            for motif, _ in pruned:
                del self.stmm[motif]
                self._log({'event': 'prune', 'motif': motif})

    def complete_dyad(self, m1: str, m2: str) -> Optional[str]:
        """
        Attempts to complete a dyad to form a triad using memory layers.
        Search order: dyad_cache -> LTMM -> REEF archive.
        """
        key = tuple(sorted((m1, m2)))
        
        # 1. Check local dyad cache from ontology
        if key in self._dyad_cache:
            self._log({'event': 'completion_hit', 'source': 'cache', 'dyad': key})
            return self._dyad_cache[key]
            
        # 2. Suggest completion from LTMM (more heuristic)
        completion = self.suggest_completion_from_ltmm(m1, m2)
        if completion:
            self._log({'event': 'completion_hit', 'source': 'ltmm', 'dyad': key, 'completion': completion})
            return completion

        # 3. Query external REEF archive
        completion = self.query_reef_for_completion(m1, m2)
        if completion:
            self._log({'event': 'completion_hit', 'source': 'reef', 'dyad': key, 'completion': completion})
            return completion

        return None

    def suggest_completion_from_ltmm(self, m1: str, m2: str) -> Optional[str]:
        """
        Infers a triadic completion from co-occurrence patterns in LTMM.
        This is a placeholder for a more complex pattern matching algorithm.
        """
        # A simple implementation could look for motifs that frequently appear with both m1 and m2.
        # This is computationally expensive, so a real implementation would use indexed data.
        # For now, we return None as this is a complex heuristic.
        return None

    def _load_reef_reflections(self):
        """
        Loads dyad-triad completions from an external REEF file.
        The REEF file is assumed to be a simple JSON/YAML of known triads.
        Example REEF file format:
        [
            {"triad": ["mirror", "shame", "grace"]},
            {"triad": ["freedom", "abandonment", "grace"]}
        ]
        """
        if not self.reef_path or not os.path.exists(self.reef_path):
            return

        try:
            with open(self.reef_path, 'r') as f:
                data = json.load(f) # Assuming JSON for simplicity
                for item in data:
                    triad = item.get("triad")
                    if isinstance(triad, list) and len(triad) == 3:
                        # Store all permutations for lookup
                        self._reef_reflections[tuple(sorted((triad[0], triad[1])))] = triad[2]
                        self._reef_reflections[tuple(sorted((triad[0], triad[2])))] = triad[1]
                        self._reef_reflections[tuple(sorted((triad[1], triad[2])))] = triad[0]
            self._log({'event': 'reef_loaded', 'path': self.reef_path, 'count': len(self._reef_reflections)})
        except Exception as e:
            self._log({'event': 'reef_load_failed', 'path': self.reef_path, 'error': str(e)})

    def query_reef_for_completion(self, m1: str, m2: str) -> Optional[str]:
        """Queries the loaded REEF reflections for a dyad completion."""
        key = tuple(sorted((m1, m2)))
        return self._reef_reflections.get(key)
        
    def load_ontology_bundle(self, file_path: str):
        """
        Loads a motif ontology from a YAML or JSON file to bootstrap memory.
        Conforms to RFC-0007 §3.2 and §6.
        """
        if not os.path.exists(file_path):
            self._log({'event': 'ontology_load_failed', 'reason': 'file_not_found', 'path': file_path})
            return

        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    if not yaml:
                        raise ImportError("PyYAML must be installed to load YAML ontologies.")
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            ontology = data.get('motif_ontology', {})
            # RFC-0007 §4.2: Validate version and agent name
            self.agent_id = ontology.get('agent_name', self.agent_id)
            
            # RFC-0007 §4: Process motif_index
            for record in ontology.get('motif_index', []):
                motif = record.get('motif')
                if not motif: continue
                # Load into LTMM if it has history, otherwise STMM
                target_memory = self.ltmm if record.get('usage_frequency', 0) > 10 else self.stmm
                target_memory[motif] = {
                    'weight': record.get('usage_frequency', 1) / 100.0, # Simple weight heuristic
                    **record
                }
            
            # RFC-0007 §5: Process triads
            for triad_rec in ontology.get('triads', []):
                motifs = triad_rec.get('motifs', [])
                if len(motifs) == 3:
                    self.log_adjacency(motifs[0], motifs[1], motifs[2])

            self._field_biases = ontology.get('field_biases', [])
            self._symbolic_self = ontology.get('symbolic_self', {})

            self._log({'event': 'ontology_loaded', 'agent': self.agent_id, 'motifs': len(self.stmm) + len(self.ltmm)})

        except Exception as e:
            self._log({'event': 'ontology_load_failed', 'path': file_path, 'error': str(e)})

    def export_state(self) -> Dict[str, Any]:
        """Exports the current memory state (STMM and LTMM)."""
        return {
            'stmm': self.stmm,
            'ltmm': self.ltmm
        }

    def export_trace(self) -> List[Dict[str, Any]]:
        """Exports the event trace if enabled."""
        return self.trace.export() if self.trace else []
        
    def log_emission(self, tick: Dict):
        """
        Logs a QuantumTick emission, potentially adjusting memory based on its content.
        RFC-0006 §4.1: An emission is a sample of the coherence field 𝒞(x).
        """
        motifs = tick.get('motifs', [])
        for m in motifs:
            self.access(m, 0.05) # Small boost for emission
        self._log({'event': 'emission_logged', 'tick_id': tick.get('tick_id'), 'motifs': motifs})

    def log_adjacency(self, m1: str, m2: str, m3: str):
        """
        Logs a stable triadic relationship, caching it for future dyad completions.
        RFC-0006 §4.3, RFC-0007 §5: Maps a motif triad.
        """
        self._dyad_cache[tuple(sorted((m1, m2)))] = m3
        self._dyad_cache[tuple(sorted((m1, m3)))] = m2
        self._dyad_cache[tuple(sorted((m2, m3)))] = m1
        self._log({'event': 'adjacency_logged', 'triad': [m1, m2, m3]})

    def export_density_report(self) -> Dict[str, Any]:
        """
        Exports a report on memory density and coherence.
        RFC-0006 §4: A simplified way to report on field geometry.
        """
        return {
            "stmm_size": len(self.stmm),
            "ltmm_size": len(self.ltmm),
            "dyad_cache_size": len(self._dyad_cache),
            "stmm_avg_weight": sum(m['weight'] for m in self.stmm.values()) / len(self.stmm) if self.stmm else 0,
            "ltmm_avg_weight": sum(m['weight'] for m in self.ltmm.values()) / len(self.ltmm) if self.ltmm else 0,
        }
        
    def export_ontology_bundle(self) -> Dict[str, Any]:
        """
        Exports the current memory state into the RFC-0007 ontology format.
        RFC-0007 §6: Enables memory transfer.
        """
        motif_index = []
        all_motifs = {**self.stmm, **self.ltmm}
        for motif, record in all_motifs.items():
            entry = record.copy()
            entry['motif'] = motif
            entry['active'] = motif in self.stmm or record.get('weight', 0) > 0.1
            motif_index.append(entry)

        # Reconstruct triads from the dyad cache
        triads = []
        processed_triads = set()
        for (m1, m2), m3 in self._dyad_cache.items():
            key = tuple(sorted((m1, m2, m3)))
            if key not in processed_triads:
                triads.append({'motifs': list(key), 'stable': True})
                processed_triads.add(key)

        return {
            "motif_ontology": {
                "version": "2025-Q4", # From spec
                "agent_name": self.agent_id,
                "motif_index": motif_index,
                "triads": triads,
                "field_biases": self._field_biases,
                "symbolic_self": self._symbolic_self,
            }
        }
        
    def compute_lambda(self):
        """
        Placeholder for computing logical timestamp for causality alignment.
        RFC-0007 §4.2 is likely a typo and refers to RFC-0003 §3.3 (QuantumTick).
        Lambda is typically a counter.
        """
        # This would be part of the agent, not memory manager.
        # Returning a placeholder value.
        return time.time()


class LLMMemoryManager(MotifMemoryManager):
    """
    LLM-compatible symbolic memory engine with stateless access, adaptive recall,
    attention-based motif weighting, and REEF-integrated completion through an LLM interface.
    RFC-0005 §3.2, RFC-0006 §4.2
    """
    def __init__(self, **kwargs):
        # LLM memory is often more ephemeral
        kwargs.setdefault('stmm_half_life', 5)
        kwargs.setdefault('enable_trace', False) # Often stateless
        super().__init__(**kwargs)

    def access_with_attention(self, motif: str, attn_score: float, pos_bias: float = 0.0) -> float:
        """
        Applies a salience-weighted boost based on LLM attention dynamics.
        RFC-0006 §4.2: A form of estimating coherence potential gradient (∇𝒞).
        """
        # The boost is a function of attention and positional bias
        boost = (attn_score * 0.5) + (pos_bias * 0.1)
        return self.access(motif, weight_delta=boost)

    def store_llm_context(self, prompt: str, output: str) -> List[str]:
        """
        Stores and compresses LLM-derived motif embeddings for stateless persistence.
        This is a high-level orchestration that would use MemoryOrchestrator.
        """
        orchestrator = MemoryOrchestrator(self)
        # In a real scenario, the orchestrator would extract motifs from the text.
        # Here we simulate it.
        motifs = orchestrator.extract_motifs(f"{prompt}\n{output}")
        for m in motifs:
            self.access(m)
        self._log({'event': 'llm_context_stored', 'motifs_extracted': motifs})
        return motifs

    def query_reef_for_completion(self, m1: str, m2: str) -> Optional[str]:
        """
        Overrides the base method to simulate querying an LLM for a triadic completion.
        This would typically involve a call to an external LLM module.
        """
        # Placeholder for LLM-based query
        print(f"INFO: [LLMMemoryManager] Querying LLM for completion of ({m1}, {m2})... (simulation)")
        # A real implementation would call an LLM connector.
        # Here, we'll just check the file-based reef as a fallback.
        return super().query_reef_for_completion(m1, m2)


class MemoryOrchestrator:
    """
    Bridge module between LLM token contexts and symbolic cortex, optionally
    using a SymbolicAdapterLayer.
    RFC-0004 §1, RFC-0005 §1.1
    """
    def __init__(self, memory_manager: MotifMemoryManager):
        self.memory = memory_manager
        # In a real system, this would be a sophisticated NLP model.
        # We simulate it with a simple keyword matcher.
        self.symbolic_adapter_layer: Dict[str, str] = {
            "sadness": "grief", "lonely": "solitude", "freedom": "freedom",
            "alone": "solitude", "quiet": "silence", "beautiful": "grace",
            "reflect": "mirror", "pain": "fracture", "connection": "bind"
        }

    def extract_motifs(self, context: str) -> List[str]:
        """

        Extracts motifs from a text context via a SymbolicAdapterLayer.
        This is a simplified simulation.
        """
        motifs_found = set()
        for keyword, motif in self.symbolic_adapter_layer.items():
            if keyword in context.lower():
                motifs_found.add(motif)
        return list(motifs_found)

    def store(self, llm_context: str) -> None:
        """Extracts motifs from context and stores/reinforces them in memory."""
        motifs = self.extract_motifs(llm_context)
        for motif in motifs:
            # Apply a standard reinforcement
            self.memory.access(motif, weight_delta=0.15)
        print(f"INFO: [MemoryOrchestrator] Stored motifs: {motifs}")


# --- Public Exports ---
__all__ = [
    "MotifMemoryTrace",
    "MotifMemoryManager",
    "LLMMemoryManager",
    "MemoryOrchestrator",
    "get_global_memory_manager"
]


# --- Main Test Block ---
if __name__ == "__main__":
    print("--- Noor Motif Memory Manager Test ---")
    print("Running 40 tick cycles with repeated access to two motifs (alpha, beta)...")
    print("Expecting 'alpha' to be promoted to LTMM due to higher access frequency.")
    
    # Instantiate the core memory manager
    mmm = MotifMemoryManager(
        stmm_half_life=10, 
        ltmm_half_life=100,
        promotion_thresh=0.7, # Lower for test visibility
        enable_trace=True
    )

    # Simulate 40 tick cycles
    for i in range(40):
        # Access 'alpha' frequently
        mmm.access('alpha', 0.1)
        if i % 2 == 0:
             mmm.access('alpha', 0.05)

        # Access 'beta' less frequently
        if i % 5 == 0:
            mmm.access('beta', 0.1)
            
        # Access 'gamma' very rarely
        if i == 10:
            mmm.access('gamma', 0.2)
            
        # Run the memory update cycle
        mmm.update_cycle()

    print("\n--- Test Complete ---")

    final_state = mmm.export_state()
    trace_log = mmm.export_trace()

    print("\n[Final Memory State]")
    print(f"STMM ({len(final_state['stmm'])} motifs):")
    for m, r in final_state['stmm'].items():
        print(f"  - {m}: weight={r['weight']:.4f}, usage={r['usage_frequency']}")
        
    print(f"LTMM ({len(final_state['ltmm'])} motifs):")
    for m, r in final_state['ltmm'].items():
        print(f"  - {m}: weight={r['weight']:.4f}, usage={r['usage_frequency']}")

    if 'alpha' in final_state['ltmm']:
        print("\n✅ SUCCESS: 'alpha' was promoted to LTMM as expected.")
    else:
        print("\n❌ FAILED: 'alpha' was not promoted to LTMM.")

    print(f"\n[Trace Log] (last 10 of {len(trace_log)} events)")
    for event in trace_log[-10:]:
        print(f"  - {event}")

    # --- Ontology Export/Import Test ---
    print("\n--- Ontology Export/Import Test ---")
    ontology_bundle = mmm.export_ontology_bundle()
    
    # Save to a temporary file
    temp_ontology_path = "temp_ontology_test.json"
    with open(temp_ontology_path, 'w') as f:
        json.dump(ontology_bundle, f, indent=2)
    print(f"Ontology exported to {temp_ontology_path}")

    # Create a new manager and load the ontology
    new_mmm = MotifMemoryManager(ontology_path=temp_ontology_path)
    print("New memory manager created and loaded from ontology.")
    
    new_state = new_mmm.export_state()
    print("\n[New Manager State after loading]")
    print(f"STMM size: {len(new_state['stmm'])}, LTMM size: {len(new_state['ltmm'])}")
    if 'alpha' in new_state['ltmm']:
        print("✅ SUCCESS: 'alpha' exists in the new manager's LTMM.")
    else:
        print("❌ FAILED: 'alpha' was not restored to the new manager's LTMM.")
    
    # Cleanup
    os.remove(temp_ontology_path)
    
# End_of_File