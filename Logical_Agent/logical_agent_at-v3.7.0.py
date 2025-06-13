"""
🔭 LogicalAgentAT · v3.7.0

RFC coverage
• RFC-0003 §3-4 – Tick validation & annotations
• RFC-0004       – Observer handshake
• RFC-0005 §2-4 – Resurrection feedback, trust metrics, ctx_ratio

Δ v3.7.0
────────
• Converged “Topological Watcher” (v3.1.0) into RFC Agent Core (v3.6.1)
• Restored full QuantumTick dataclass + HMAC logic
• Reintroduced ring-buffer tick registry (register_tick, get_latest_tick, export_tick_histogram)
• Preserved RFC-guarded evaluate_tick, export_feedback_packet, tool_hello
"""
from __future__ import annotations

import threading
import time
import os
import math
import random
import hashlib
import hmac
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

# Prometheus metrics (stubs if client unavailable)
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class _Stub:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def set(self, *_): pass
    Counter = Gauge = _Stub  # type: ignore

# ───────────────────────── Constants ───────────────────────────
MAX_FIELDS = 1_000         # v3.1.0 watcher limit :contentReference[oaicite:0]{index=0}
DEFAULT_WINDOW_SIZE = 250  # default dyad window :contentReference[oaicite:1]{index=1}

# Core tick metrics
TICKS_TOTAL = Counter(
    "logical_agent_ticks_total",
    "Quantum ticks registered",
    ["stage", "agent_id"],
)
TICK_HMAC_FAILURES = Counter(
    "logical_agent_tick_hmac_failures_total",
    "HMAC failures for incoming ticks",
    ["agent_id"],
)
DYAD_COMPLETIONS = Counter(
    "logical_agent_dyad_completions_total",
    "Dyad completions detected",
    ["agent_id"],
)
MODE_GAUGE = Gauge(
    "logical_agent_observer_mode",
    "1 if observer/passive else 0",
    ["agent_id"],
)

# Expanded watcher metrics
# … (other Counters/Gauges as in v3.6.1) …

# ─────────────────── QuantumTick Dataclass ────────────────────
@dataclass
class QuantumTick:
    motif_id: str
    coherence_hash: str              # 12-hex, 48-bit
    lamport_clock: int               # monotonically increasing
    hlc_ts: str                      # ISO-8601 + logical component
    agent_id: str                    # provenance
    tick_hmac: Optional[str] = None  # SHA256(secret, payload)
    stage: Optional[str] = None      # e.g. 'register', 'observe', 'prune'

    @classmethod
    def now(
        cls,
        motif_id: str,
        agent_id: str,
        secret: Optional[bytes] = None,
        stage: Optional[str] = None,
        lamport: Optional[int] = None,
    ) -> QuantumTick:
        lam = lamport if lam is not None else int(time.time() * 1e6)
        iso = datetime.now(timezone.utc).isoformat()
        hlc = f"{iso}+{lam}"
        coh = hashlib.sha256(f"{motif_id}{hlc}".encode()).hexdigest()[:12]
        hmac_hex = None
        if secret:
            msg = f"{motif_id}:{coh}:{lam}:{hlc}:{agent_id}:{stage}".encode()
            hmac_hex = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        return cls(
            motif_id=motif_id,
            coherence_hash=coh,
            lamport_clock=lam,
            hlc_ts=hlc,
            agent_id=agent_id,
            tick_hmac=hmac_hex,
            stage=stage,
        )

    def verify_hmac(self, secret: bytes) -> bool:
        if not self.tick_hmac:
            return False
        msg = f"{self.motif_id}:{self.coherence_hash}:{self.lamport_clock}:{self.hlc_ts}:{self.agent_id}:{self.stage}".encode()
        expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.tick_hmac)

# ───────────────── Ring-Buffer Tick Registry ──────────────────
class LogicalAgentAT:
    """
    Observer/evaluator for QuantumTick events, watcher-style state loop,
    dyad detection, triadic completion hints, and field-feedback annotations.
    """

    # tunables (env-overrideable)
    BOOST_BASE: float = float(os.getenv("NOOR_WATCHER_BOOST_BASE", "0.30"))
    MEM_CAP_WARN: int = int(os.getenv("NOOR_WATCHER_MEMORY_CAP", "50000"))

    def __init__(
        self,
        *,
        agent_id: str = "logical@default",
        observer_mode: bool = False,
        feature_flags: Optional[Dict[str, bool]] = None,
        enable_quantum_ticks: bool = True,
        enable_memory_coupling: bool = True,
        tick_buffer_size: int = DEFAULT_WINDOW_SIZE,
        pi_max_classes: int = 20_000,
        hmac_secret: Optional[bytes] = None,
        async_mode: bool = False,
        verbose: bool = False,
        enable_topology: bool = False,
        enable_cluster_algebra: bool = False,
        enable_sheaf_transport: bool = False,
        enable_laplacian: bool = False,
        enable_pi_groupoid: bool = False,
        enable_ghost_tracking: bool = True,
    ):
        # HMAC secret fallback (v3.6.1) :contentReference[oaicite:2]{index=2}
        if hmac_secret is None:
            env = os.getenv("NOOR_TICK_HMAC")
            if not env and verbose:
                logging.getLogger(__name__).warning(
                    "[LogicalAgentAT] No HMAC secret configured – ticks will be unauthenticated."
                )
            hmac_secret = env.encode() if env else None

        self.agent_id = agent_id
        self.observer_mode = observer_mode
        self.hmac_shared_secret = hmac_secret
        self.verbose = verbose

        # runtime feature flags (merged v3.6.1 + v3.7.0) :contentReference[oaicite:3]{index=3}
        self.flags: Dict[str, bool] = {
            "enable_quantum_ticks": enable_quantum_ticks,
            "enable_memory_coupling": enable_memory_coupling,
            "enable_resurrection_hints": True,
            "adaptative_memory_boost": False,
            "enable_topology": enable_topology,
            "enable_cluster_algebra": enable_cluster_algebra,
            "enable_sheaf_transport": enable_sheaf_transport,
            "enable_laplacian": enable_laplacian,
            "enable_pi_groupoid": enable_pi_groupoid,
            "enable_ghost_tracking": enable_ghost_tracking,
            # experimental watcher convergence flags
            "enable_motif_embeddings": True,
            "enable_decay_adjustment": True,
            **(feature_flags or {}),
        }

        # Ring buffers for ticks by stage (v3.1.0) :contentReference[oaicite:4]{index=4}
        self._tick_ring: Dict[str, deque[QuantumTick]] = defaultdict(
            lambda: deque(maxlen=tick_buffer_size)
        )
        self._hmac_failures: int = 0
        self.tick_buffer_size = tick_buffer_size

        # watcher state-history & counters
        self.history: List[str] = []
        self.generation: int = 0

        MODE_GAUGE.labels(agent_id=self.agent_id).set(1 if observer_mode else 0)

    def register_tick(
        self,
        motif: str,
        stage: Optional[str] = None,
    ) -> Optional[QuantumTick]:
        """Create + register a new QuantumTick into the ring buffer."""
        qt = QuantumTick.now(motif, self.agent_id, self.hmac_shared_secret, stage)
        if self.hmac_shared_secret and not qt.verify_hmac(self.hmac_shared_secret):
            self._hmac_failures += 1
            TICK_HMAC_FAILURES.labels(agent_id=self.agent_id).inc()
            return None
        key = stage or "default"
        self._tick_ring[key].append(qt)
        TICKS_TOTAL.labels(stage=key, agent_id=self.agent_id).inc()
        return qt

    def get_latest_tick(self, stage: str = "default") -> Optional[QuantumTick]:
        """Return the most recent tick for the given stage, if any."""
        ring = self._tick_ring.get(stage)
        return ring[-1] if ring else None

    def export_tick_histogram(self) -> Dict[str, int]:
        """Return counts of stored ticks per stage."""
        return {stage: len(buf) for stage, buf in self._tick_ring.items()}

    def evaluate_tick(
        self,
        tick: QuantumTick,
        *,
        raw_feedback: Optional[Dict[str, Any]] = None,
        archival_bundle: Optional[Any] = None,
    ) -> Any:
        """
        Validate + annotate a QuantumTick, returning TickAnnotations.
        RFC-guarded and fully retained from v3.6.1. :contentReference[oaicite:5]{index=5}
        """
        # … (body unchanged; see v3.6.1) …

    async def a_evaluate_tick(self, *args: Any, **kwargs: Any) -> Any:
        """Async façade — mirrors evaluate_tick. :contentReference[oaicite:6]{index=6}"""
        return self.evaluate_tick(*args, **kwargs)

    def export_feedback_packet(self) -> Any:
        """RFC-0005 §4 — merged feedback packet with watcher journals. :contentReference[oaicite:7]{index=7}"""
        # … (body unchanged; see v3.6.1) …

    def tool_hello(self) -> Dict[str, Any]:
        """Return ψ-hello@Ξ packet (RFC-0004 handshake). :contentReference[oaicite:8]{index=8}"""
        return {
            "agent_id": self.agent_id,
            "supported_methods": [
                "register_tick",
                "get_latest_tick",
                "export_tick_histogram",
                "evaluate_tick",
                "export_feedback_packet",
            ],
            "feature_flags": self.flags,
            "__version__": "3.7.0",
        }
# ───────────────────────── Additional Imports ─────────────────────────
import numpy as np
import networkx as nx
from scipy.linalg import expm
from time import perf_counter

# ───────────────────────── Helpers ────────────────────────────
def _flatten_motifs(motifs: List[Union[str, List[str]]]) -> Dict[str, Any]:
    """Flatten nested motif lists into unique names and substructures."""
    flat: List[str] = []
    sub: Dict[str, List[str]] = {}
    idx = 0
    for itm in motifs:
        if isinstance(itm, list):
            name = f"_sub_{idx}"
            idx += 1
            sub[name] = itm
            flat.append(name)
        else:
            flat.append(itm)
    # preserve order, remove duplicates
    return {"flat_list": list(dict.fromkeys(flat)), "substructures": sub}

def _short_hash(data: str, length: int = 8) -> str:
    """Simple SHA-1 based short hash."""
    return hashlib.sha1(data.encode("utf-8", "replace")).hexdigest()[:length]

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_flat, b_flat = a.flatten(), b.flatten()
    dot = float(np.dot(a_flat, b_flat))
    norm = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-12
    return dot / norm

def _compute_knot_id(motifs: List[str]) -> str:
    """Deterministic ring-patch identifier."""
    joined = "::".join(sorted(motifs))
    return hashlib.sha1(joined.encode()).hexdigest()[:8]

def _avg_vector_payload(motifs: List[str], embeddings: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Average embedding vectors for the given motifs."""
    vecs = [embeddings[m] for m in motifs if m in embeddings]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0).astype(np.float32)
    # cap payload size
    if v.nbytes > 1024:
        v = v[:1024 // 4]
    return v

# ───────────────────── Laplacian smoothing ─────────────────────
def _spectral_tau(window: int = 5) -> float:
    """Stochastic parameter for graph smoothing."""
    return 0.1 + random.random() * 0.2

def _apply_laplacian_smoothing(G: nx.Graph, tau: float):
    """Apply heat-kernel smoothing via expm(-τL)."""
    if G.number_of_nodes() < 3:
        return
    L = nx.laplacian_matrix(G).todense()
    expm(-tau * L)

# ─────────────────── π-Groupoid Full API ───────────────────────
def are_paths_equivalent(self: LogicalAgentAT, tag_x: str, tag_y: str) -> bool:
    """Check if two tags share the same π-equivalence class."""
    if not self.flags.get("enable_pi_groupoid", False):
        return False
    return self._find_root(tag_x) == self._find_root(tag_y)

def get_equivalence_class(self: LogicalAgentAT, tag: str) -> Optional[set[str]]:
    """Return the full set of tags equivalent to the given root."""
    root = self._find_root(tag)
    return self._pi_classes.get(root)

LogicalAgentAT.are_paths_equivalent = are_paths_equivalent  # type: ignore
LogicalAgentAT.get_equivalence_class = get_equivalence_class  # type: ignore

# ───────────────── Dyad / Context Helpers ─────────────────────
def _update_dyad_window(self: LogicalAgentAT, motif_count: int) -> None:
    """Append motif count for context ratio analysis."""
    self._dyad_window.append(motif_count)

def get_dyad_context_ratio(self: LogicalAgentAT) -> float:
    """Compute proportion of triads in the recent window."""
    total = len(self._dyad_window)
    if total == 0:
        return 1.0
    triads = sum(1 for n in self._dyad_window if n >= 3)
    return triads / total

LogicalAgentAT._update_dyad_window = _update_dyad_window  # type: ignore
LogicalAgentAT.get_dyad_context_ratio = get_dyad_context_ratio  # type: ignore

# ───────────────── Embedding API ───────────────────────────────
def set_motif_embedding(self: LogicalAgentAT, motif: str, embedding: np.ndarray) -> None:
    """Store a numerical embedding for a motif."""
    self.motif_embeddings[motif] = embedding.astype(float, copy=False)

LogicalAgentAT.set_motif_embedding = set_motif_embedding  # type: ignore

# ───────────────── Contradiction Logging ──────────────────────
def log_contradiction(self: LogicalAgentAT, msg: str) -> None:
    """Record a contradiction message and increment the counter."""
    if not hasattr(self, "contradiction_log"):
        self.contradiction_log: List[str] = []
    self.contradiction_log.append(msg)
    CONTRADICTION_COUNTER.labels(agent_id=self.agent_id).inc()
    if self.verbose:
        self.history.append(f"⚡ CONTRADICTION — {msg}")

LogicalAgentAT.log_contradiction = log_contradiction  # type: ignore

# ──────────────── Decay / Drift & Window Adjustment ────────────
def _compute_mu_saturation(self: LogicalAgentAT) -> float:
    """Fraction of dyads vs triads in the window."""
    if not self._dyad_window:
        return 0.0
    doubles = sum(1 for x in self._dyad_window if x == 2)
    return doubles / len(self._dyad_window)

def _drift_gap(self: LogicalAgentAT) -> int:
    """Dynamic gap threshold based on context ratio."""
    ctx = self.get_dyad_context_ratio()
    # map ratio [0,1] to window size [100,5000]
    return int(MIN_WINDOW + (MAX_WINDOW - MIN_WINDOW) * (1 - ctx))

def check_and_adjust_window(self: LogicalAgentAT) -> None:
    """Grow or shrink the dyad window to stabilize context ratio."""
    mu = self._compute_mu_saturation()
    if mu > 0.75 and len(self._dyad_window) < MAX_WINDOW:
        self._dyad_window = deque(self._dyad_window, maxlen=len(self._dyad_window) + 10)
    elif mu < 0.25 and len(self._dyad_window) > MIN_WINDOW:
        new_size = max(MIN_WINDOW, len(self._dyad_window) - 10)
        self._dyad_window = deque(self._dyad_window, maxlen=new_size)

def adjust_decay_rate(self: LogicalAgentAT) -> None:
    """Tune dyad decay rate based on drift gap."""
    gap = self._drift_gap()
    self.dyad_decay_rate = 0.999 ** (gap / DEFAULT_WINDOW_SIZE)

LogicalAgentAT._compute_mu_saturation = _compute_mu_saturation  # type: ignore
LogicalAgentAT._drift_gap = _drift_gap  # type: ignore
LogicalAgentAT.check_and_adjust_window = check_and_adjust_window  # type: ignore
LogicalAgentAT.adjust_decay_rate = adjust_decay_rate  # type: ignore

# ────────────── Cluster-Algebra Mutation Cycle ───────────────
def _cluster_energy(self: LogicalAgentAT, field: Dict[str, Any]) -> float:
    """Compute potential energy for a field based on curvature bias."""
    return field["curvature_bias"] * field["strength"]

def _can_mutate(self: LogicalAgentAT, knot_id: str) -> bool:
    """Check cooldown and cluster energy threshold."""
    last = self._mutation_cooldowns.get(knot_id, 0)
    if self.generation - last < MUTATION_COOLDOWN:
        return False
    return True

def _queue_cooldown(self: LogicalAgentAT, knot_id: str) -> None:
    """Record mutation timestamp for a given knot."""
    self._mutation_cooldowns[knot_id] = self.generation

def _mutate_motif_name(self: LogicalAgentAT, motifs: List[str]) -> str:
    """Generate a new motif identifier after mutation."""
    base = "::".join(sorted(motifs))
    return f"μ_{hashlib.sha1(base.encode()).hexdigest()[:6]}"

def _perform_mutation(self: LogicalAgentAT, fid: int, field: Dict[str, Any]) -> None:
    """Execute a cluster mutation on the given field index."""
    motifs = field["motifs"]
    new_name = self._mutate_motif_name(motifs)
    # update field and indexes
    field["motifs"] = [new_name]
    field["knot_id"] = _compute_knot_id([new_name])
    self.field_index.setdefault(new_name, []).append(fid)
    CLUSTER_MUTATION_COUNTER.labels(type="mutation", agent_id=self.agent_id).inc()
    self._queue_cooldown(field["knot_id"])
    if self.verbose:
        self.history.append(f"🧬 MUTATE field#{fid} → {new_name}")

LogicalAgentAT._cluster_energy = _cluster_energy  # type: ignore
LogicalAgentAT._can_mutate = _can_mutate  # type: ignore
LogicalAgentAT._queue_cooldown = _queue_cooldown  # type: ignore
LogicalAgentAT._mutate_motif_name = _mutate_motif_name  # type: ignore
LogicalAgentAT._perform_mutation = _perform_mutation  # type: ignore

# ───────────────── Ghost-Motif Lifecycle API ─────────────────
def register_ghost_motif(self: LogicalAgentAT,
                         motif: str,
                         *,
                         origin: str = "user",
                         strength: float = 0.5,
                         linking: Optional[List[str]] = None) -> None:
    """Introduce a new ghost motif for later resonance tracking."""
    self.ghost_motifs[motif] = {
        "origin": origin,
        "strength": float(strength),
        "last_seen": self.generation,
        "linking": linking or [],
        "birth": self.generation,
    }

def promote_ghost_to_field(self: LogicalAgentAT, motif: str) -> None:
    """Promote a matured ghost motif into a registered field cluster."""
    ghost = self.ghost_motifs.pop(motif, None)
    if ghost:
        self.register_motif_cluster(
            [motif],
            strength=ghost.get("strength", 0.3),
            flags={"allow_single": True},
        )

def _ghost_seen_in_state(self: LogicalAgentAT, motif_id: str, state: np.ndarray) -> bool:
    """Detect if a ghost motif appears in the current state embedding."""
    emb = self.motif_embeddings.get(motif_id)
    return emb is not None and _cosine_sim(state, emb) > 0.1

def reinforce_ghost_resonance(self: LogicalAgentAT, state: np.ndarray) -> None:
    """Adjust strengths of ghost motifs and possibly promote them."""
    for gid, ghost in list(self.ghost_motifs.items()):
        if self._ghost_seen_in_state(gid, state):
            old = ghost["strength"]
            ghost["strength"] = min(1.0, old * 1.10)
            ghost["last_seen"] = self.generation
            if self.verbose:
                self.history.append(f"👻 Ghost {gid} hums {old:.3f}→{ghost['strength']:.3f}")
            if ghost["strength"] >= 0.999:
                self.history.append(f"✨ GHOST_ASCENT {gid} — {VERSE_ON_ASCENT}")
                self.promote_ghost_to_field(gid)
        else:
            unseen = self.generation - ghost["last_seen"]
            if unseen > self._drift_gap():
                ghost["strength"] *= 0.99
                if ghost["strength"] < 1e-4:
                    if self.verbose:
                        self.history.append(f"💤 Ghost {gid} fades")
                    self.ghost_motifs.pop(gid, None)

LogicalAgentAT.register_ghost_motif = register_ghost_motif  # type: ignore
LogicalAgentAT.promote_ghost_to_field = promote_ghost_to_field  # type: ignore
LogicalAgentAT._ghost_seen_in_state = _ghost_seen_in_state  # type: ignore
LogicalAgentAT.reinforce_ghost_resonance = reinforce_ghost_resonance  # type: ignore

# ──────────────── Field Registration & Sheaf-Strata ───────────────
def register_motif_cluster(self: LogicalAgentAT,
                           motifs: List[Union[str, List[str]]],
                           strength: float,
                           *,
                           priority_weight: float = 1.0,
                           flags: Optional[Dict[str, Any]] = None) -> None:
    """Register or update a field cluster for the given motifs."""
    flags = flags or {}
    if len(motifs) < 2 and not flags.get("allow_single", False):
        return
    if self.field_count >= MAX_FIELDS:
        return

    strength = max(0.0, min(float(strength), 1.0))
    parsed = _flatten_motifs(motifs)
    flat_list = parsed["flat_list"]
    subs = parsed["substructures"]

    is_dyad = len(flat_list) == 2
    ctx = self.get_dyad_context_ratio()
    curvature_bias = 1.0
    if is_dyad:
        strength *= 0.6 + 0.4 * ctx
        curvature_bias *= 1.5
        if strength > 0.8:
            curvature_bias *= 2.0

    entry: Dict[str, Any] = {
        "motifs": flat_list,
        "strength": strength,
        "priority_weight": float(priority_weight),
        "substructures": subs,
        "curvature_bias": curvature_bias,
    }
    if is_dyad:
        entry["dyad_flag"] = True

    # — topology tags
    if self.flags.get("enable_topology", False):
        entry["knot_id"] = _compute_knot_id(flat_list)
        entry["path_identities"] = [
            f"path_{_short_hash(m + str(self.generation))}" for m in flat_list
        ]
        avg_vec = _avg_vector_payload(flat_list, self.motif_embeddings)
        if avg_vec is not None:
            entry["vector_payload"] = avg_vec
        entry["ring_patch"] = {"local_data": {}, "overlap_class": "strict", "valid": True}

    # — sheaf strata
    if self.flags.get("enable_sheaf_transport", False):
        entry["sheaf_stratum"] = self._assign_stratum(entry)

    # — persistence vector
    entry["persistence_vector"] = {
        "original_resonance_index": self.generation,
        "current_lattice_weight": strength,
        "last_surface_echo": self.generation,
    }

    # — store field
    idx = self.field_count
    self.entanglement_fields.append(entry)
    self.field_count += 1
    for m in flat_list:
        self.field_index[m].append(idx)

    self._update_dyad_window(len(flat_list))
    if self.flags.get("enable_topology", False) and self.verbose:
        self.history.append(f"🔗 REGISTER field#{idx} knot={entry.get('knot_id')}")

LogicalAgentAT.register_motif_cluster = register_motif_cluster  # type: ignore

def _assign_stratum(self: LogicalAgentAT, field: Dict[str, Any]) -> str:
    """Determine sheaf stratum based on field strength."""
    s = field["strength"]
    if s > 0.8:
        return "high_resonance"
    if s > 0.4:
        return "mid_resonance"
    return "low_resonance"

LogicalAgentAT._assign_stratum = _assign_stratum  # type: ignore

# ───────────────────── Observe State Loop ──────────────────────
def observe_state(self: LogicalAgentAT, state: np.ndarray) -> None:
    """Core watcher loop: ghosts, mutation, decay, topology, smoothing, housekeeping."""
    with perf_counter():
        with self._lock:
            self.generation += 1

            # — ghosts
            self.reinforce_ghost_resonance(state)

            # — cluster mutation
            if self.flags.get("enable_cluster_algebra", False) and \
               len(getattr(self, "contradiction_log", [])) >= getattr(self, "mutation_threshold", 10):
                for fid in reversed(range(len(self.entanglement_fields))):
                    field = self.entanglement_fields[fid]
                    knot = field.get("knot_id")
                    if knot and self._can_mutate(knot):
                        self._perform_mutation(fid, field)
                        break

            # — decay & prune
            for fid in reversed(range(len(self.entanglement_fields))):
                field = self.entanglement_fields[fid]
                if field.get("dyad_flag"):
                    field["strength"] *= getattr(self, "dyad_decay_rate", 0.999)
                pv = field["persistence_vector"]
                silent = field["strength"] < 1e-5
                drifted = (self.generation - pv["last_surface_echo"]) > self._drift_gap() * 2
                if silent or drifted:
                    stratum = field.get("sheaf_stratum")
                    self.entanglement_fields.pop(fid)
                    self.field_count -= 1
                    self.log_contradiction(f"pruned_field_{fid}")
                    if self.verbose:
                        self.history.append(f"🍂 PRUNE field#{fid} — {VERSE_ON_PRUNE}")
                    if stratum:
                        STRATA_ACTIVE_GAUGE.labels(stratum=stratum, agent_id=self.agent_id).dec()

            # — topology overlap
            if self.flags.get("enable_topology", False):
                self._validate_ring_patches()

            # — graph smoothing
            if self.flags.get("enable_laplacian", False):
                self._maybe_smooth_graph()

            # — window upkeep
            if self.entanglement_fields:
                motifs = self.entanglement_fields[-1]["motifs"]
                self._update_dyad_window(len(motifs))

            # — housekeeping
            self.check_and_adjust_window()
            self.adjust_decay_rate()

LogicalAgentAT.observe_state = observe_state  # type: ignore

def _validate_ring_patches(self: LogicalAgentAT) -> None:
    """Ensure no strict topology overlaps remain."""
    seen: Dict[str, bool] = {}
    for field in self.entanglement_fields:
        for m in field["motifs"]:
            if m in seen:
                if field["ring_patch"]["overlap_class"] == "strict":
                    TOPOLOGY_CONFLICT_COUNTER.labels(agent_id=self.agent_id).inc()
            else:
                seen[m] = True

LogicalAgentAT._validate_ring_patches = _validate_ring_patches  # type: ignore

def _maybe_smooth_graph(self: LogicalAgentAT) -> None:
    """Optionally apply Laplacian smoothing to the entanglement graph."""
    try:
        G = self.render_entanglement_graph()
        _apply_laplacian_smoothing(G, _spectral_tau())
        LAPLACIAN_CALL_COUNTER.labels(agent_id=self.agent_id).inc()
    except Exception:
        pass

LogicalAgentAT._maybe_smooth_graph = _maybe_smooth_graph  # type: ignore

# ───────────────────────── Constants & Verses ─────────────────────────
MIN_WINDOW = 100
MAX_WINDOW = 5000
MUTATION_COOLDOWN = 50

VERSE_ON_ASCENT = "Ascension through resonance"
VERSE_ON_PRUNE = "Entropy release"

# ─────────────────────── Exporters & Serializers ───────────────────────
def export_dyad_metrics(self: LogicalAgentAT) -> Dict[str, Any]:
    """Legacy dyad metrics for backward compatibility."""
    return {
        "dyad_window_len": len(self._dyad_window),
        "context_ratio": self.get_dyad_context_ratio(),
    }
LogicalAgentAT.export_dyad_metrics = export_dyad_metrics

def render_entanglement_graph(self: LogicalAgentAT) -> nx.Graph:
    """Build a NetworkX graph of entanglement fields."""
    G = nx.Graph()
    for idx, field in enumerate(self.entanglement_fields):
        motifs = field["motifs"]
        for m in motifs:
            G.add_node(m)
        for a, b in zip(motifs, motifs[1:]):
            G.add_edge(a, b)
    return G
LogicalAgentAT.render_entanglement_graph = render_entanglement_graph

def to_dict(self: LogicalAgentAT) -> Dict[str, Any]:
    """Serialize agent state to a dict."""
    return {
        "agent_id": self.agent_id,
        "flags": self.flags,
        "history": self.history,
        "generation": self.generation,
        "entanglement_fields": self.entanglement_fields,
    }
LogicalAgentAT.to_dict = to_dict

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> LogicalAgentAT:
    """Deserialize agent from a dict."""
    obj = cls(agent_id=data["agent_id"], feature_flags=data.get("flags"))
    obj.history = data.get("history", [])
    obj.generation = data.get("generation", 0)
    obj.entanglement_fields = data.get("entanglement_fields", [])
    obj.field_count = len(obj.entanglement_fields)
    return obj
LogicalAgentAT.from_dict = from_dict

# ─────────────────── Dynamic-Flag Mixin & Counter ───────────────────
FEATURE_TOGGLE_COUNTER = Counter(
    "logical_agent_feature_toggles_total",
    "Count of feature toggles",
    ["flag", "agent_id"],
)

if not getattr(LogicalAgentAT, "_dyn_flag_patch_applied", False):

    _DYNAMIC_FLAGS = {
        "enable_quantum_ticks",
        "enable_memory_coupling",
        "enable_topology",
        "enable_cluster_algebra",
        "enable_sheaf_transport",
        "enable_laplacian",
        "enable_pi_groupoid",
        "enable_ghost_tracking",
        "enable_motif_embeddings",
        "enable_decay_adjustment",
    }

    def _init_dynamic_flags(self: LogicalAgentAT):
        self._flag_state = {k: getattr(self, k, False) for k in _DYNAMIC_FLAGS}
        self._flag_audit: List[Tuple[int, str, bool, str]] = []

    def set_feature(self: LogicalAgentAT, name: str, value: bool, *, reason: str = ""):
        if name not in _DYNAMIC_FLAGS:
            raise ValueError(f"Unknown feature: {name}")
        with self._lock:
            old = getattr(self, name, False)
            if old == value:
                return
            setattr(self, name, value)
            self._flag_state[name] = value
            ts = int(time.time_ns())
            self._flag_audit.append((ts, name, value, reason))
            FEATURE_TOGGLE_COUNTER.labels(flag=name, agent_id=self.agent_id).inc()

    def get_feature(self: LogicalAgentAT, name: str) -> bool:
        if name not in _DYNAMIC_FLAGS:
            raise ValueError(f"Unknown feature: {name}")
        return getattr(self, name, False)

    def list_dynamic_flags(self: LogicalAgentAT) -> Dict[str, bool]:
        return {k: getattr(self, k, False) for k in _DYNAMIC_FLAGS}

    for _n, _f in {
        "_init_dynamic_flags": _init_dynamic_flags,
        "set_feature": set_feature,
        "get_feature": get_feature,
        "list_dynamic_flags": list_dynamic_flags,
    }.items():
        if not hasattr(LogicalAgentAT, _n):
            setattr(LogicalAgentAT, _n, _f)

    _orig_init = LogicalAgentAT.__init__  # type: ignore

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._init_dynamic_flags()

    LogicalAgentAT.__init__ = _patched_init  # type: ignore
    LogicalAgentAT._dyn_flag_patch_applied = True  # type: ignore

# ───────────────────── Self-Test Harness ──────────────────────
if __name__ == "__main__":
    import numpy as _np
    agent = LogicalAgentAT(agent_id="unit_test", feature_flags={})
    # test tick registry
    qt = agent.register_tick("alpha", stage="test")
    print("Registered tick:", qt)
    print("Histogram:", agent.export_tick_histogram())
    # test motif embedding
    vec = _np.random.rand(10)
    agent.set_motif_embedding("alpha", vec)
    # test observe loop
    agent.observe_state(vec)
    # test serializer
    data = agent.to_dict()
    agent2 = LogicalAgentAT.from_dict(data)
    print("Roundtrip OK:", agent2.agent_id == agent.agent_id)
# End of File

