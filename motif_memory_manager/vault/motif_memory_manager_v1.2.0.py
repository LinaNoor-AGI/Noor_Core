# motif_memory_manager.py · v1.2.0
#
# RFC-compliant symbolic memory engine that models motif access, field-curved 
# recall, and triadic inheritance across recursive temporal cycles. It forms 
# the symbolic cortex of Noor, responsible for continuity, resonance tracking, 
# and motif decay dynamics.
#
# Author : Noor Collective Labs
# License: MIT
# Schema : 2025-Q3-motif-memory-v1.2
"""
🎯 Purpose
————————
Provide a thread-safe cache that:
1. Tracks motif weights in short-term (STMM) & long-term (LTMM) stores.
2. Applies exponential decay (half-life expressed in *cycles*).
3. Promotes motifs upward when salience ≥ promotion threshold (with hysteresis).
4. Exposes `retrieve()` to surface supportive motifs when reasoning stalls.
5. Exposes `complete_dyad()` to infer missing motifs from TheReefArchive.
6. Implements RFC-0006/0007 density, adjacency, and triadic depth tracking.
7. Supports import/export of motif ontologies for agent continuity (RFC-0007).
8. Optionally journals trace events for observability & replay.

Designed as a plug-in: SymbolicTaskEngine, RecursiveAgentFT or any watcher
can import an instance and call `.access()` / `.update_cycle()` each tick.

Dependencies: stdlib, PyYAML (for ontology loading).
STMM acts as Noor's **working memory** (live motifs).
LTMM acts as Noor's **symbolic archive** (slow-to-forget).
"""

from __future__ import annotations

__version__ = "1.2.0"
_SCHEMA_VERSION__ = "2025-Q3-motif-memory-v1.2"

import math
import os
import time
import hashlib
import logging
import threading
import asyncio
from collections import deque, OrderedDict
from contextlib import contextmanager, asynccontextmanager
from typing import Callable, Dict, List, Optional, Tuple, Set
from pathlib import Path
import re

# Optional dependency for ontology loading
try:
    import yaml
except ImportError:
    yaml = None

# ─────────────────────────────────────────────────────────────
# Optional Stubs for Future/External Components
# ─────────────────────────────────────────────────────────────
try:
    from .prm_buffer import PRMBuffer # Placeholder
except ImportError:
    PRMBuffer = None
try:
    from .consciousness_monitor import ConsciousnessMonitor # Placeholder
except ImportError:
    ConsciousnessMonitor = None

# ─────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────
DEFAULT_ST_HALF_LIFE = 25          # cycles  (≈ 0.5 s at 50 Hz)
DEFAULT_LT_HALF_LIFE = 10000       # cycles  (~3.3 min at 50 Hz)
DEFAULT_PROMOTION_THRESH = 0.9
DEFAULT_DEMOTION_DELTA = 0.05      # hysteresis gap
TRACE_BUFFER_LEN = 256             # ring size for trace journal
STMM_SOFT_CAP = 50000              # edge-case guard for STMM+LTMM size
DEFAULT_CACHE_SIZE = 10000         # max entries in dyad cache

# ─────────────────────────────────────────────────────────────
# Prometheus stubs (optional)
# ─────────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover
    class _Stub:                  # noqa: D401
        def labels(self, *_, **__):
            return self
        def inc(self, *_): ...
    Counter = _Stub               # type: ignore

REEF_HIT       = Counter("reef_dyad_hits_total",     ["agent_id"])
REEF_MISS      = Counter("reef_dyad_miss_total",    ["agent_id"])
CACHE_HIT      = Counter("dyad_cache_hits_total",   ["agent_id"])
CACHE_MISS     = Counter("dyad_cache_miss_total",   ["agent_id"])
STMM_CAP_SKIP  = Counter("stmm_cap_skips_total",    ["agent_id"])

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _decay_factor(half_life: int) -> float:
    """Computes exponential decay multiplier per update cycle."""
    if half_life <= 0:
        return 0.0
    return math.pow(0.5, 1.0 / half_life)

def _default_jaccard(a: set[str], b: set[str]) -> float:
    """Returns Jaccard similarity between motif token sets."""
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ─────────────────────────────────────────────────────────────
# Trace journal
# ─────────────────────────────────────────────────────────────
class MotifMemoryTrace:
    """Time-indexed circular buffer for motif event traces (access, retrieve, promote, demote)."""

    def __init__(self, cap: int = TRACE_BUFFER_LEN):
        self._buf: deque[dict] = deque(maxlen=cap)
        self._lock = threading.Lock()

    def append(self, event: dict) -> None:
        """Appends structured event with timestamp, motif, and context to the ring buffer."""
        with self._lock:
            self._buf.append(event)

    def export(self) -> List[dict]:
        """Returns the chronologically ordered list of recorded memory events."""
        with self._lock:
            return list(self._buf)

# ─────────────────────────────────────────────────────────────
# MotifMemoryManager
# ─────────────────────────────────────────────────────────────
class MotifMemoryManager:
    """Dual-layer adaptive memory engine with REEF-aware dyad completion, decay dynamics, triadic motif linkage, emission-based continuity tracking, and density-aware recall adaptation."""

    def __init__(
        self,
        *,
        stmm_half_life: int = DEFAULT_ST_HALF_LIFE,
        ltmm_half_life: int = DEFAULT_LT_HALF_LIFE,
        promotion_thresh: float = DEFAULT_PROMOTION_THRESH,
        demotion_delta: float = DEFAULT_DEMOTION_DELTA,
        similarity_fn: Optional[Callable[[Set[str], Set[str]], float]] = None,
        enable_trace: bool = False,
        reef_path: Optional[str | Path] = "./noor/data/index.REEF",
        agent_id: str = "memory@default",
        stmm_soft_cap: int = STMM_SOFT_CAP,
        reload_reef_on_mtime_change: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
        ontology_path: Optional[str | Path] = None,
    ) -> None:
        # Core memory stores
        self._stmm: Dict[str, float] = {}
        self._ltmm: Dict[str, float] = {}

        # Decay and promotion parameters
        self._st_factor = _decay_factor(stmm_half_life)
        self._lt_factor = _decay_factor(ltmm_half_life)
        self._promo_thresh = promotion_thresh
        self._demote_thresh = promotion_thresh - demotion_delta

        self._sim_fn = similarity_fn or _default_jaccard

        # Threading and async safety
        self._tlock = threading.RLock()
        self._alock = asyncio.Lock()
        self._trace = MotifMemoryTrace() if enable_trace else None

        # REEF archive and dyad cache state
        self._reef_path = Path(reef_path) if reef_path else None
        self._reef_index: Optional[Dict[str, set[str]]] = None
        self._reef_mtime = 0.0
        self._dyad_cache: OrderedDict[Tuple[str, str], str] = OrderedDict()
        self._reload_reef = reload_reef_on_mtime_change
        self._cache_size = cache_size

        # Caps and Prometheus metrics
        self._stmm_cap = stmm_soft_cap
        self._prom_lbls = {"agent_id": agent_id}
        
        # RFC-0006 & RFC-0007 State Fields
        self.emission_count: Dict[str, int] = {}  # ρᵢ
        self.triad_links: Dict[str, List[str]] = {} # Parent -> children for synthesis
        self.triad_depth: Dict[str, int] = {} # λᵢ (memoized)
        self.adjacency_map: Dict[str, List[str]] = {} # m1 <-> m2 links
        self.bias_field: Dict[str, str] = {} # motif -> ψ-field
        
        # Stubs for external components
        self.stmm: Optional[PRMBuffer] = PRMBuffer() if PRMBuffer else None
        self.monitor: Optional[ConsciousnessMonitor] = ConsciousnessMonitor() if ConsciousnessMonitor else None

        if ontology_path:
            self.load_ontology_bundle(str(ontology_path))

    # ──────────────────────────────
    # Internal lock helpers
    # ──────────────────────────────
    @contextmanager
    def _locked(self):
        with self._tlock:
            yield

    @asynccontextmanager
    async def _locked_async(self):
        async with self._alock:
            yield

    # ──────────────────────────────
    # Public API
    # ──────────────────────────────
    def update_cycle(self) -> None:
        """Apply decay & promotion/demotion once per reasoning cycle. Triggers pruning of obsolete motifs. (RFC-0006 §4.3)"""
        with self._locked():
            self._apply_decay(self._stmm, self._st_factor)
            self._apply_decay(self._ltmm, self._lt_factor)
            self._promote_and_demote()
            self.prune()

    def access(self, motif_id: str, boost: float = 0.2) -> None:
        """Boosts activation of a motif; promotes STMM presence."""
        with self._locked():
            if (len(self._stmm) + len(self._ltmm)) >= self._stmm_cap:
                self._log("soft_cap_skip", motif_id, 0.0)
                STMM_CAP_SKIP.labels(**self._prom_lbls).inc()
                return

            if motif_id in self._ltmm:
                w = self._ltmm[motif_id]
                self._stmm[motif_id] = max(self._stmm.get(motif_id, 0.0), w + boost)

            self._stmm[motif_id] = min(self._stmm.get(motif_id, 0.0) + boost, 1.0)
            self.log_emission(motif_id)
            self._log("access", motif_id, self._stmm[motif_id])

    def retrieve(
        self,
        query_motif: str,
        *,
        top_k: int = 3,
        exclude_stmm: bool = True,
    ) -> List[str]:
        """Returns top-k motifs from LTMM by activation × similarity."""
        with self._locked():
            scored: List[Tuple[str, float]] = []
            for m, w in self._ltmm.items():
                if exclude_stmm and m in self._stmm:
                    continue
                score = w * self._sim_fn({query_motif}, {m})
                if score > 0:
                    scored.append((m, score))
            scored.sort(key=lambda t: -t[1])
            result = [m for m, _ in scored[:top_k]]
            self._log("retrieve", query_motif, 0.0, returned=result)
            return result

    def export_state(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns snapshot of STMM and LTMM states."""
        with self._locked():
            return dict(self._stmm), dict(self._ltmm)

    def export_trace(self) -> List[dict]:
        """Returns all logged memory events if tracing is enabled."""
        if self._trace is None:
            return []
        return self._trace.export()
    
    # ──────────────────────────────
    # RFC-0006 / RFC-0007 Methods
    # ──────────────────────────────
    def log_emission(self, motif_id: str) -> None:
        """Increments ρᵢ count for motif (RFC-0006 §4.1)."""
        with self._locked():
            self.emission_count[motif_id] = self.emission_count.get(motif_id, 0) + 1

    def compute_lambda(self, motif_id: str) -> int:
        """DFS-based λᵢ computation for recursion depth (RFC-0007 §4.2)."""
        with self._locked():
            if motif_id in self.triad_depth:
                return self.triad_depth[motif_id]

            # Memoization and cycle detection
            visited = set()
            
            def _dfs(current_motif: str) -> int:
                if current_motif in visited:
                    return 0 # Cycle detected, break recursion
                visited.add(current_motif)

                parents = self.triad_links.get(current_motif, [])
                if not parents:
                    return 0 # Base motif
                
                max_depth = 0
                for parent in parents:
                    max_depth = max(max_depth, _dfs(parent))
                
                # Depth is 1 + max depth of parents
                return 1 + max_depth

            depth = _dfs(motif_id)
            self.triad_depth[motif_id] = depth
            return depth

    def log_adjacency(self, m1: str, m2: str) -> None:
        """Records motif linkage in adjacency map (RFC-0006 §4.3, RFC-0007 §5)."""
        with self._locked():
            self.adjacency_map.setdefault(m1, []).append(m2)
            self.adjacency_map.setdefault(m2, []).append(m1)

    def prune(self, emission_threshold: int = 1, staleness_threshold: int = 100) -> None:
        """Clears motifs with low ρᵢ or stale adjacency edges (RFC-0006 §4.3)."""
        with self._locked():
            all_motifs = set(self._stmm.keys()) | set(self._ltmm.keys())
            to_prune = {
                m for m in all_motifs
                if self.emission_count.get(m, 0) < emission_threshold and
                   not self.adjacency_map.get(m)
            }
            
            for m in to_prune:
                self._stmm.pop(m, None)
                self._ltmm.pop(m, None)
                self.emission_count.pop(m, None)
                self._log("prune", m, 0.0)

    def export_density_report(self) -> Dict:
        """Thread-safe motif density snapshot (RFC-0006 §4)."""
        with self._locked():
            return {
                "emission_count": dict(self.emission_count),
                "triad_depths": {m: self.compute_lambda(m) for m in self.triad_links},
                "adjacency_map": dict(self.adjacency_map),
                "stmm_size": len(self._stmm),
                "ltmm_size": len(self._ltmm),
            }

    def export_ontology_bundle(self) -> Dict:
        """RFC-0007-compliant export of triadic, adjacency, and bias state (RFC-0007 §6)."""
        with self._locked():
            motif_index = []
            all_motifs = set(self._stmm.keys()) | set(self._ltmm.keys()) | set(self.emission_count.keys())
            
            for m in sorted(list(all_motifs)):
                dyad_links = []
                for neighbor in self.adjacency_map.get(m, []):
                    dyad_links.append({"motif": neighbor, "tension": 0.5}) # Tension is illustrative
                
                motif_index.append({
                    "motif": m,
                    "parents": self.triad_links.get(m, []),
                    "resonance_field": self.bias_field.get(m, "unknown"),
                    "dyad_links": dyad_links,
                    "usage_frequency": self.emission_count.get(m, 0),
                    "active": m in self._stmm or m in self._ltmm,
                })
            
            # Infer triads from triad_links (parent->child map)
            triads = []
            for child, parents in self.triad_links.items():
                if len(parents) == 2:
                    triads.append({"motifs": parents + [child], "stable": True})
            
            return {
                "motif_ontology": {
                    "version": "2025-Q4",
                    "agent_name": self._prom_lbls.get("agent_id", "unknown"),
                    "motif_index": motif_index,
                    "triads": triads,
                    # Other fields like field_biases would be populated from their respective stores
                }
            }

    def load_ontology_bundle(self, path: str) -> None:
        """Imports prior state (ρᵢ, λᵢ, bias field) from RFC-0007 motif ontology file."""
        if not yaml:
            logging.warning("PyYAML not installed. Cannot load ontology bundle.")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            ontology = data.get("motif_ontology", {})
            with self._locked():
                for item in ontology.get("motif_index", []):
                    motif = item.get("motif")
                    if not motif: continue
                    
                    self.emission_count[motif] = item.get("usage_frequency", 0)
                    if item.get("parents"):
                        self.triad_links[motif] = item.get("parents")
                    if item.get("resonance_field") != "unknown":
                        self.bias_field[motif] = item.get("resonance_field")

                for triad_data in ontology.get("triads", []):
                    motifs = triad_data.get("motifs", [])
                    if len(motifs) == 3:
                        # Assuming m1 + m2 -> m3 structure
                        self.triad_links[motifs[2]] = [motifs[0], motifs[1]]
                
                self._log("load_ontology", path, 1.0)
        except FileNotFoundError:
            logging.error(f"Ontology file not found: {path}")
        except Exception as e:
            logging.error(f"Failed to load ontology bundle from {path}: {e}")
            
    # ──────────────────────────────
    # Dyad completion interfaces
    # ──────────────────────────────
    def query_reef_for_completion(self, dyad: Tuple[str, str]) -> Optional[str]:
        """Wrapper for single best REEF-based completion."""
        results = self.complete_dyad(dyad, top_k=1)
        return results[0] if results else None

    def complete_dyad(self, dyad: Tuple[str, str], *, top_k: int = 1) -> List[str]:
        """Returns REEF-based completions weighted by similarity + memory."""
        key = tuple(sorted(dyad))
        if key in self._dyad_cache:
            CACHE_HIT.labels(**self._prom_lbls).inc()
            return [self._dyad_cache[key]]
        CACHE_MISS.labels(**self._prom_lbls).inc()

        clusters = self._load_reef_reflections()
        if not clusters:
            REEF_MISS.labels(**self._prom_lbls).inc()
            return []

        m1, m2 = key
        scored: List[Tuple[str, float]] = []
        _, ltmm = self.export_state()
        for cl in clusters.values():
            if {m1, m2}.issubset(cl):
                for cand in cl - {m1, m2}:
                    rank = self._sim_fn(cl, {m1, m2}) + ltmm.get(cand, 0.0)
                    scored.append((cand, rank))

        if not scored:
            REEF_MISS.labels(**self._prom_lbls).inc()
            return []

        scored.sort(key=lambda t: -t[1])
        REEF_HIT.labels(**self._prom_lbls).inc()
        self._dyad_cache[key] = scored[0][0]
        if len(self._dyad_cache) > self._cache_size:
            self._dyad_cache.popitem(last=False)
        return [m for m, _ in scored[:top_k]]

    def suggest_completion_from_ltmm(self, dyad: Tuple[str, str], *, top_k: int = 1) -> List[str]:
        """Fallback completions from LTMM memory patterns."""
        m1, m2 = dyad
        ignore = {m1, m2}
        _, ltmm = self.export_state()
        scored = [(m, w) for m, w in ltmm.items() if m not in ignore]
        scored.sort(key=lambda t: -t[1])
        return [m for m, _ in scored[:top_k]]

    # ──────────────────────────────
    # Private helpers
    # ──────────────────────────────
    def _apply_decay(self, store: Dict[str, float], factor: float) -> None:
        for k in list(store.keys()):
            store[k] *= factor
            if store[k] < 1e-6:
                del store[k]

    def _promote_and_demote(self) -> None:
        for motif in list(self._stmm.keys()):
            w = self._stmm[motif]
            if w >= self._promo_thresh:
                self._ltmm[motif] = max(self._ltmm.get(motif, 0.0), w)
                del self._stmm[motif]
                self._log("promote", motif, w)
        for motif in list(self._ltmm.keys()):
            w = self._ltmm[motif]
            if w < self._demote_thresh:
                self._stmm[motif] = w
                del self._ltmm[motif]
                self._log("demote", motif, w)

    def _log(self, event_type: str, motif: str, weight: float, **extra) -> None:
        if self._trace is None:
            return
        self._trace.append(
            {
                "ts_ns": time.time_ns(),
                "type": event_type,
                "motif": motif,
                "weight": round(weight, 4),
                **extra,
            }
        )

    def _load_reef_reflections(self) -> Dict[str, set[str]]:
        """Parse `index.REEF`-style file into {cluster_id: set(motif, …)}."""
        if not self._reef_path or not self._reef_path.exists():
            return {}

        mtime = self._reef_path.stat().st_mtime
        if self._reef_index is not None and (not self._reload_reef or mtime == self._reef_mtime):
            return self._reef_index

        clusters: Dict[str, set[str]] = {}
        pattern = re.compile(r"^motif_id\s*=\s*([a-z0-9_ ]+)$", re.IGNORECASE)
        try:
            with self._reef_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    s = line.strip()
                    if s.startswith("motif_id"):
                        match = pattern.match(s)
                        if match:
                            key = hashlib.sha1(s.encode()).hexdigest()[:12]
                            motifs = set(match.group(1).lower().split())
                            clusters[key] = motifs
        except OSError as e:
            logging.warning(f"Could not read REEF file: {e}")

        self._reef_index = clusters
        self._reef_mtime = mtime
        return clusters

# ─────────────────────────────────────────────────────────────
# Global singleton helper
# ─────────────────────────────────────────────────────────────
GLOBAL_MEMORY_MANAGER: Optional[MotifMemoryManager] = None

def get_global_memory_manager() -> MotifMemoryManager:
    """Initializes or returns a singleton global memory manager with tracing enabled."""
    global GLOBAL_MEMORY_MANAGER
    if GLOBAL_MEMORY_MANAGER is None:
        GLOBAL_MEMORY_MANAGER = MotifMemoryManager(enable_trace=True)
    return GLOBAL_MEMORY_MANAGER

# ─────────────────────────────────────────────────────────────
# Minimal sanity test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--- Running Minimal Sanity Test ---")
    mm = MotifMemoryManager(enable_trace=True)

    # Simulate 40 tick cycles with repeated access to two motifs (α, β)
    for t in range(40):
        if t % 5 == 0:
            mm.access("α")
        if t % 7 == 0:
            mm.access("β", boost=0.3)
        mm.update_cycle()

    print("\nFinal State (STMM, LTMM):")
    stmm_state, ltmm_state = mm.export_state()
    print(f"STMM: {stmm_state}")
    print(f"LTMM: {ltmm_state}")
    
    print("\nTrace (last 5 events):")
    for ev in mm.export_trace()[-5:]:
        print(ev)
    
    print("\nEmission Count (ρᵢ):")
    print(mm.export_density_report()["emission_count"])

# Public exports
__all__ = [
    "MotifMemoryTrace",
    "MotifMemoryManager",
    "get_global_memory_manager",
]

# End_of_File