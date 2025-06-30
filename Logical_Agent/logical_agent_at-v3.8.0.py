"""
🔭 LogicalAgentAT · v3.8.0

RFC coverage
• RFC-0003 §3-4 – Tick validation & annotations
• RFC-0004       – Observer handshake
• RFC-0005 §2-4 – Resurrection feedback, trust metrics, ctx_ratio

Δ v3.7.1
────────
• Added dyad ratio gauge for window breathing
• Enforced π-tag regex guard
• Recent mutation & histogram fields initialized
• Epoch histogram exporter added
• Cluster energy, mutation, and feedback-packet stats refined
• Window-breathing logic & tag-guard override
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
import re
from datetime import datetime, timezone

import logical_agent_at_update_0001  # 🔧 hot‑patch: monitoring and triad scoring (v1.0.2)

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
# MAX_FIELDS = 1_000         # v3.1.0 watcher limit
# DEFAULT_WINDOW_SIZE = 250  # default dyad window

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

# ratio gauge for window breathing logic
DYAD_RATIO_GAUGE = Gauge(
    "logical_agent_dyad_ratio",
    "Proportion of dyads vs triads in recent window",
    ["agent_id"],
)

# π-Groupoid ID guard
PI_TAG_REGEX = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Part 2 of 4: Core class, dataclasses, and tick registry enhancements

from tick_schema import validate_tick, QuantumTick, CrystallizedMotifBundle
from field_feedback import make_field_feedback, FieldFeedback

# ─── Adaptive Thresholds (v3.7.3) ────────────────────────────────
def _compute_default_max_fields() -> int:
    """Heuristic based on memory pressure or CPU core count."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        gb = mem.total / (1024 ** 3)
        if gb >= 64:
            return 4000
        if gb >= 32:
            return 2000
        if gb >= 16:
            return 1500
        return 1000
    except ImportError:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        return 1500 if cores >= 8 else 1000

def _compute_default_dyad_window() -> int:
    """Adaptive dyad window size using CPU count."""
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        return 512 if cores >= 8 else 256
    except ImportError:
        return 256

MAX_FIELDS = _compute_default_max_fields()
DEFAULT_WINDOW_SIZE = _compute_default_dyad_window()

# Expose runtime overrides
def set_max_fields(n: int) -> None:
    global MAX_FIELDS
    MAX_FIELDS = max(100, n)
    try:
        MAX_FIELDS_GAUGE.labels().set(MAX_FIELDS)
    except Exception:
        pass

def get_max_fields() -> int:
    return MAX_FIELDS

def set_dyad_window_size(n: int) -> None:
    global DEFAULT_WINDOW_SIZE
    DEFAULT_WINDOW_SIZE = max(64, n)
    try:
        DYAD_WINDOW_GAUGE.labels().set(DEFAULT_WINDOW_SIZE)
    except Exception:
        pass

def get_dyad_window_size() -> int:
    return DEFAULT_WINDOW_SIZE

# Prometheus gauges (safe if missing client)
try:
    from prometheus_client import Gauge as _Gauge
    MAX_FIELDS_GAUGE = _Gauge("max_fields_dynamic", "Adaptive max fields")
    DYAD_WINDOW_GAUGE = _Gauge("dyad_window_dynamic", "Adaptive dyad window size")
except ImportError:
    class _StubGauge:
        def labels(self): return self
        def set(self, *_): pass
    MAX_FIELDS_GAUGE = DYAD_WINDOW_GAUGE = _StubGauge()

__version__       = "3.8.0"
_SCHEMA_VERSION__ = "2025-Q4-logical-agent-v1.2"
SCHEMA_COMPAT      = ("RFC-0003:4", "RFC-0004", "RFC-0005:4")
__all__ = (
    "LogicalAgentAT",
    "TickAnnotations",
    "FeedbackPacket",
)

@dataclass(slots=True)
class TickAnnotations:
    """RFC-0003 §4 — tick-annotation container"""
    triad_complete: bool = False
    memory_promotion: bool = False
    reward_delta: float = 0.0
    ctx_ratio: float = 0.5
    trust: float = 0.5
    resurrection_hint: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class FeedbackPacket:
    """RFC-0005 §4 — inter-agent feedback bundle"""
    ctx_ratio: float
    contradiction_avg: float
    harm_hits: int
    recent_mutations: int
    ring_patch: Optional[str] = None
    ghost_hint: Optional[str] = None
    entropy_drift: List[Tuple[str, float, float]] = field(default_factory=list)
    contradiction_context: List[Dict[str, Any]] = field(default_factory=list)

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
        # HMAC secret fallback
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

        # runtime feature flags
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
            "enable_motif_embeddings": True,
            "enable_decay_adjustment": True,
            **(feature_flags or {}),
        }

        # Ring buffers for ticks by stage
        self._tick_ring: Dict[str, deque[QuantumTick]] = defaultdict(
            lambda: deque(maxlen=tick_buffer_size)
        )
        self._hmac_failures: int = 0
        self.tick_buffer_size = tick_buffer_size

        # watcher state-history & counters
        self.history: List[str] = []
        self.generation: int = 0

        # ─── CRITICAL COUNTERS & LAST‐CTX FIXES ───────────────────
        # total ticks and triads counted
        self._tick_count: int = 0
        self._triad_count: int = 0
        # store last context ratio for export_feedback_packet
        self._last_ctx_ratio: float = 0.5
        # ─────────────────────────────────────────────────────────

        # ─── v3.7.1 additions ───────────────────────────────────
        # recent mutation history & moving average
        self._recent_mutations: deque[int] = deque(maxlen=50)
        self._contradiction_avg: float = 0.0

        # epoch histogram for ticks
        self._epoch_histogram: Dict[str, int] = defaultdict(int)

        # initialize dyad ratio gauge
        DYAD_RATIO_GAUGE.labels(agent_id=self.agent_id).set(1.0)
        # ────────────────────────────────────────────────────────

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
        # update epoch histogram
        self._epoch_histogram[qt.motif_id] += 1
        return qt

    def get_latest_tick(self, stage: str = "default") -> Optional[QuantumTick]:
        """Return the most recent tick for the given stage, if any."""
        ring = self._tick_ring.get(stage)
        return ring[-1] if ring else None

    def export_tick_histogram(self) -> Dict[str, int]:
        """Return counts of stored ticks per stage."""
        return {stage: len(buf) for stage, buf in self._tick_ring.items()}

    def export_epoch_histogram(self) -> Dict[str, int]:
        """Return total ticks seen per motif ID since start."""
        return dict(self._epoch_histogram)
    def evaluate_tick(
        self,
        tick: QuantumTick,
        *,
        raw_feedback: Optional[Dict[str, Any]] = None,
        archival_bundle: Optional[CrystallizedMotifBundle] = None,
    ) -> TickAnnotations:
        """Validate + annotate a QuantumTick, returning TickAnnotations."""
        # 1️⃣ RFC schema guard
        validate_tick(tick)

        # 2️⃣ Field-feedback (ctx_ratio, trust, entropy…)
        fb: FieldFeedback = make_field_feedback(
            tick, raw_feedback, archival_bundle
        )

        # 3️⃣ Dyad / triad detection
        motifs = getattr(tick, "motifs", [tick.motif_id])
        dyad = self._detect_dyad(motifs)
        triad = self._complete_triad(dyad) if dyad else None

        ann = TickAnnotations(
            triad_complete=bool(triad),
            memory_promotion=False,
            reward_delta=0.0,
            ctx_ratio=fb.ctx_feedback.ctx_ratio,
            trust=fb.trust_profiles[0].trust if fb.trust_profiles else 0.5,
            resurrection_hint=fb.extensions.get("resurrection_hint"),
        )

        # store last context ratio for feedback packet
        self._last_ctx_ratio = ann.ctx_ratio

        if triad and not self.observer_mode:
            self._annotate_field_effects(tick, triad)

        # metrics
        self._tick_count += 1
        TICKS_TOTAL.labels(stage=tick.stage or "default", agent_id=self.agent_id).inc()
        if triad:
            self._triad_count += 1
            DYAD_COMPLETIONS.labels(agent_id=self.agent_id).inc()

        return ann

        # metrics
        self._tick_count += 1
        TICKS_TOTAL.labels(stage=tick.stage or "default", agent_id=self.agent_id).inc()
        if triad:
            DYAD_COMPLETIONS.labels(agent_id=self.agent_id).inc()

        return ann

    async def a_evaluate_tick(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TickAnnotations:
        """Async façade — mirrors evaluate_tick."""
        return self.evaluate_tick(*args, **kwargs)

    def export_feedback_packet(self) -> FeedbackPacket:
        """Export a summary packet for downstream symbolic cores."""
        contradiction_avg = getattr(self, "_contradiction_avg", 0.0)
        harm_hits = len(getattr(self, "contradiction_log", []))
        recent_mutations = len(self._recent_mutations)
        ring_patch = None
        ghost_hint = None
        if self.flags.get("enable_ghost_tracking", False) and self.ghost_motifs:
            ghost_hint = max(
                self.ghost_motifs.items(),
                key=lambda kv: kv[1].get("strength", 0.0),
            )[0]

        # journals
        entropy = list(getattr(self, "_drift_log", []))
        context = list(getattr(self, "_contradiction_log", []))

        # update gauge metrics
        if self.flags.get("enable_entropy_journal", False):
            DRIFT_LOG_LENGTH.labels(agent_id=self.agent_id).set(len(entropy))
        if self.flags.get("enable_context_journal", False):
            CONTEXT_LOG_LENGTH.labels(agent_id=self.agent_id).set(len(context))
        ratio = self.get_dyad_context_ratio()
        DYAD_RATIO_GAUGE.labels(agent_id=self.agent_id).set(ratio)

        return FeedbackPacket(
            ctx_ratio=getattr(self, "_last_ctx_ratio", 0.5),
            contradiction_avg=contradiction_avg,
            harm_hits=harm_hits,
            recent_mutations=recent_mutations,
            ring_patch=ring_patch,
            ghost_hint=ghost_hint,
            entropy_drift=entropy,
            contradiction_context=context,
        )

    def tool_hello(self) -> Dict[str, Any]:
        """Return ψ-hello@Ξ packet (RFC-0004 handshake)."""
        return {
            "agent_id": self.agent_id,
            "supported_methods": [
                "register_tick",
                "get_latest_tick",
                "export_tick_histogram",
                "export_epoch_histogram",
                "evaluate_tick",
                "export_feedback_packet",
            ],
            "feature_flags": self.flags,
            "__version__": __version__,
        }

# ───────────────────────────────────────────────────────────
# Private helpers (flag-gated)
# ───────────────────────────────────────────────────────────

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
    return {"flat_list": list(dict.fromkeys(flat)), "substructures": sub}

def _short_hash(data: str, length: int = 8) -> str:
    """Simple SHA-1 based short hash."""
    return hashlib.sha1(data.encode("utf-8", "replace")).hexdigest()[:length]

def _cosine_sim(a: Any, b: Any) -> float:
    """Cosine similarity between two vectors."""
    import numpy as _np
    a_f = a.flatten(); b_f = b.flatten()
    dot = float(_np.dot(a_f, b_f))
    norm = (_np.linalg.norm(a_f) * _np.linalg.norm(b_f)) + 1e-12
    return dot / norm

def _compute_knot_id(motifs: List[str]) -> str:
    """Deterministic ring-patch identifier."""
    joined = "::".join(sorted(motifs))
    return hashlib.sha1(joined.encode()).hexdigest()[:8]

def _avg_vector_payload(motifs: List[str], embeddings: Dict[str, Any]) -> Optional[Any]:
    """Average embedding vectors for the given motifs."""
    import numpy as _np
    vecs = [embeddings[m] for m in motifs if m in embeddings]
    if not vecs:
        return None
    v = _np.mean(vecs, axis=0).astype(_np.float32)
    if v.nbytes > 1024:
        v = v[:1024 // 4]
    return v

def _spectral_tau(window: int = 5) -> float:
    """Stochastic parameter for graph smoothing."""
    return 0.1 + random.random() * 0.2

def _apply_laplacian_smoothing(G: Any, tau: float) -> None:
    """Apply heat-kernel smoothing via expm(-τL)."""
    try:
        import networkx as _nx
        from scipy.linalg import expm
        if G.number_of_nodes() < 3:
            return
        L = _nx.laplacian_matrix(G).todense()
        _ = expm(-tau * L)
    except ImportError:
        pass

def _detect_dyad(self: LogicalAgentAT, motifs: List[str]) -> Optional[Tuple[str, str]]:
    """Core dyad detection: simple last-two motifs pairing."""
    if len(motifs) < 2:
        return None
    return motifs[-2], motifs[-1]

def _complete_triad(self: LogicalAgentAT, dyad: Tuple[str, str]) -> Optional[List[str]]:
    """Triadic closure via memory, recursion, or chain fallback."""
    if not self.flags.get("enable_memory_coupling", False):
        return None
    mgr = get_global_memory_manager()
    if mgr:
        try:
            comp = mgr.retrieve(list(dyad))
            if comp:
                return [dyad[0], dyad[1], comp[0]]
        except Exception:
            pass
    if self.flags.get("enable_recursive_triads", False):
        rec = self._complete_recursive_triad(dyad)
        if rec:
            return rec
    if self.flags.get("enable_dyad_chains", False):
        if mgr:
            try:
                for cand in mgr.neighbors(list(dyad)):
                    if cand not in dyad:
                        return [dyad[0], dyad[1], cand]
            except Exception:
                pass
        self._seed_from_partial_dyad(dyad)
    return None

def _annotate_field_effects(self: LogicalAgentAT, tick: QuantumTick, triad: List[str]) -> None:
    """Apply π-groupoid, topology, cluster, sheaf, laplacian, ghost, contradiction, context, drift."""
    if not self._guard_write():
        return
    self.register_path_equivalence(triad[0], triad[1])
    if self.flags.get("enable_topology", False):
        TOPOLOGY_CONFLICT_COUNTER.labels(agent_id=self.agent_id).inc()
    if self.flags.get("enable_cluster_algebra", False):
        self._cluster_mutation(triad)
    if self.flags.get("enable_sheaf_transport", False):
        self._sheaf_stratify(triad[2])
    if self.flags.get("enable_laplacian", False):
        self._laplacian_smooth(triad)
    self._record_ghost(triad[2])
    self._log_contradiction(triad[2])
    if self.flags.get("enable_context_journal", False):
        self._log_contradiction_context(triad[2], {"triad": triad})
    if self.flags.get("enable_entropy_journal", False):
        self._track_entropy_drift(triad[2], age=0.0, coherence=1.0)

def register_path_equivalence(self: LogicalAgentAT, tag_a: str, tag_b: str) -> None:
    """DSU union for π equivalence (guarded)."""
    if not self.flags.get("enable_pi_groupoid", False) or not self._guard_write():
        return
    ra, rb = self._find_root(tag_a), self._find_root(tag_b)
    if ra == rb:
        return
    canon, other = (ra, rb) if ra < rb else (rb, ra)
    self._pi_classes.setdefault(canon, {canon})
    self._pi_classes.setdefault(other, {other})
    self._pi_classes[canon].update(self._pi_classes.pop(other))
    PI_MERGE_COUNTER.labels(agent_id=self.agent_id).inc()

def _guard_write(self: LogicalAgentAT) -> bool:
    """Block mutations when passive."""
    return not self.observer_mode

def _laplacian_smooth(self: LogicalAgentAT, triad: List[str]) -> None:
    """Apply placeholder smoothing to cluster strengths."""
    if not self.flags.get("enable_laplacian", False):
        return
    LAPLACIAN_CALL_COUNTER.labels(agent_id=self.agent_id).inc()

def _cluster_mutation(self: LogicalAgentAT, triad: List[str]) -> None:
    """Mutate clusters based on triadic interactions."""
    if not self.flags.get("enable_cluster_algebra", False):
        return
    CLUSTER_MUTATION_COUNTER.labels(type="default", agent_id=self.agent_id).inc()

def _sheaf_stratify(self: LogicalAgentAT, motif: str) -> None:
    """Assign motif to sheaf stratum."""
    if not self.flags.get("enable_sheaf_transport", False):
        return
    STRATA_ACTIVE_GAUGE.labels(stratum=motif, agent_id=self.agent_id).set(1)
# Part 4 of 4: Exporters, serializers, dynamic-flag mixin, and test harness

# ───────────────────── Exporters & Serializers ─────────────────────
def export_dyad_metrics(self: LogicalAgentAT) -> Dict[str, Any]:
    """Legacy dyad metrics for backward compatibility."""
    return {
        "dyad_window_len": len(self._dyad_window),
        "context_ratio": self.get_dyad_context_ratio(),
    }

def render_entanglement_graph(self: LogicalAgentAT) -> Any:
    """Build a NetworkX graph of entanglement fields."""
    try:
        import networkx as nx
    except ImportError:
        return None
    G = nx.Graph()
    for idx, field in enumerate(self.entanglement_fields):
        motifs = field["motifs"]
        for m in motifs:
            G.add_node(m)
        for a, b in zip(motifs, motifs[1:]):
            G.add_edge(a, b)
    return G

def to_dict(self: LogicalAgentAT) -> Dict[str, Any]:
    """Serialize agent state to a dict."""
    return {
        "agent_id": self.agent_id,
        "flags": self.flags,
        "history": self.history,
        "generation": self.generation,
        "entanglement_fields": self.entanglement_fields,
    }

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> LogicalAgentAT:
    """Deserialize agent from a dict."""
    obj = cls(agent_id=data["agent_id"], feature_flags=data.get("flags"))
    obj.history = data.get("history", [])
    obj.generation = data.get("generation", 0)
    obj.entanglement_fields = data.get("entanglement_fields", [])
    obj.field_count = len(obj.entanglement_fields)
    return obj

LogicalAgentAT.export_dyad_metrics        = export_dyad_metrics  # type: ignore
LogicalAgentAT.render_entanglement_graph  = render_entanglement_graph  # type: ignore
LogicalAgentAT.to_dict                    = to_dict  # type: ignore
LogicalAgentAT.from_dict                  = from_dict  # type: ignore

# ────────────────── Dynamic-Flag Mixin & Counter ──────────────────
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
        "set_feature":          set_feature,
        "get_feature":          get_feature,
        "list_dynamic_flags":   list_dynamic_flags,
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

    agent = LogicalAgentAT(
        agent_id="unit_test",
        feature_flags={
            "enable_topology": True,
            "enable_cluster_algebra": True,
            "enable_sheaf_transport": True,
            "enable_laplacian": True,
            "enable_pi_groupoid": True,
            "enable_ghost_tracking": True,
        },
    )
    # Tick registry test
    qt = agent.register_tick("test_motif", stage="test")
    print("Registered tick:", qt)
    print("Tick histogram:", agent.export_tick_histogram())
    print("Epoch histogram:", agent.export_epoch_histogram())

    # Motif embedding test
    vec = _np.random.rand(16)
    agent.set_motif_embedding("test_motif", vec)
    print("Embedding stored.")

    # Observe state loop test
    agent.observe_state(vec)
    print("Observe_state run, fields count:", len(agent.entanglement_fields))

    # Serialize/deserialize
    data = agent.to_dict()
    agent2 = LogicalAgentAT.from_dict(data)
    print("Roundtrip OK:", agent2.agent_id == agent.agent_id)

# Part 4 of 4: Remaining helpers, observe loop, exporters, mixin, and test harness

# ────────────── Contradiction & Drift Helpers ──────────────
def _log_contradiction(self: LogicalAgentAT, motif: str) -> None:
    """Record a contradiction event for statistical decay."""
    CONTRADICTION_COUNTER.labels(agent_id=self.agent_id).inc()
    self._dyad_window.append(1)
    if self._dyad_window:
        self._contradiction_avg = sum(self._dyad_window) / len(self._dyad_window)

def _record_ghost(self: LogicalAgentAT, motif: str, strength: float = 1.0) -> None:
    """Track ghost motif occurrences and strength."""
    if not self.flags.get("enable_ghost_tracking", False):
        return
    entry = self.ghost_motifs.get(motif, {"count": 0, "strength": 0.0})
    entry["count"] += 1
    entry["strength"] += strength
    self.ghost_motifs[motif] = entry
    GHOST_COUNTER.labels(agent_id=self.agent_id).inc()

def _complete_recursive_triad(self: LogicalAgentAT, dyad: Tuple[str, str]) -> Optional[List[str]]:
    """If motifs in dyad are entangled triads, attempt recursive closure."""
    if not self.flags.get("enable_recursive_triads", False):
        return None
    mgr = get_global_memory_manager()
    if mgr:
        try:
            chain = mgr.retrieve_chain(list(dyad))
            if chain:
                return [dyad[0], dyad[1], chain[0]]
        except Exception:
            pass
    return None

def _seed_from_partial_dyad(self: LogicalAgentAT, dyad: Tuple[str, str]) -> None:
    """Queue a dyad for future triad completion attempts."""
    if not self.flags.get("enable_dyad_chains", False):
        return
    self._buffers.setdefault("dyad_chain", deque()).append(dyad)

def _track_entropy_drift(self: LogicalAgentAT, motif: str, age: float, coherence: float) -> None:
    """Record motif age and decay into drift journal."""
    if not self.flags.get("enable_entropy_journal", False):
        return
    self._drift_log.append((motif, age, coherence))

def _log_contradiction_context(self: LogicalAgentAT, motif: str, context: Dict[str, Any]) -> None:
    """Store contradiction lineage linked to motif ancestry."""
    if not self.flags.get("enable_context_journal", False):
        return
    self._contradiction_log.append({"motif": motif, "context": context})

# ───────────── π-Groupoid Regex-Guarded Find ─────────────
def _find_root(self: LogicalAgentAT, tag: str) -> str:
    """DSU find for π equivalence, with regex guard."""
    if not PI_TAG_REGEX.match(tag):
        return tag
    parent = self._pi_tag_index.get(tag, tag)
    if parent != tag:
        parent = self._pi_tag_index[tag] = self._find_root(parent)
    return parent

LogicalAgentAT._find_root = _find_root  # type: ignore

# ────────── Cluster-Algebra Mutation Cycle ──────────
MUTATION_ENERGY_THRESHOLD = float(os.getenv("NOOR_MUTATION_ENERGY_THRESHOLD", "0.0"))

def _cluster_energy(self: LogicalAgentAT, field: Dict[str, Any]) -> float:
    """Compute potential energy: -log1p(strength) * number_of_motifs."""
    return -math.log1p(field["strength"]) * len(field["motifs"])

def _can_mutate(self: LogicalAgentAT, knot_id: str) -> bool:
    """Check cooldown and energy threshold."""
    last = self._mutation_cooldowns.get(knot_id, 0)
    if self.generation - last < getattr(self, "_mutation_cooldown", 50):
        return False
    energy = self._cluster_energy(self.entanglement_fields[self.field_index[knot_id][0]])
    return energy > MUTATION_ENERGY_THRESHOLD

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
    self._recent_mutations.append(fid)

LogicalAgentAT._cluster_energy = _cluster_energy  # type: ignore
LogicalAgentAT._can_mutate       = _can_mutate    # type: ignore
LogicalAgentAT._queue_cooldown   = _queue_cooldown  # type: ignore
LogicalAgentAT._mutate_motif_name= _mutate_motif_name  # type: ignore
LogicalAgentAT._perform_mutation = _perform_mutation  # type: ignore

# ───────── Ghost-Motif Lifecycle API ─────────
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

def _ghost_seen_in_state(self: LogicalAgentAT, motif_id: str, state: Any) -> bool:
    """Detect if a ghost motif appears in the current state embedding."""
    emb = self.motif_embeddings.get(motif_id)
    return emb is not None and _cosine_sim(state, emb) > 0.1

def reinforce_ghost_resonance(self: LogicalAgentAT, state: Any) -> None:
    """Adjust strengths of ghost motifs and possibly promote them."""
    for gid, ghost in list(self.ghost_motifs.items()):
        if self._ghost_seen_in_state(gid, state):
            old = ghost["strength"]
            ghost["strength"] = min(1.0, old * 1.10)
            ghost["last_seen"] = self.generation
            if ghost["strength"] >= 0.999:
                self.promote_ghost_to_field(gid)
        else:
            unseen = self.generation - ghost["last_seen"]
            if unseen > len(self._dyad_window):
                ghost["strength"] *= 0.99
                if ghost["strength"] < 1e-4:
                    self.ghost_motifs.pop(gid, None)

LogicalAgentAT.register_ghost_motif         = register_ghost_motif  # type: ignore
LogicalAgentAT.promote_ghost_to_field      = promote_ghost_to_field  # type: ignore
LogicalAgentAT._ghost_seen_in_state       = _ghost_seen_in_state  # type: ignore
LogicalAgentAT.reinforce_ghost_resonance  = reinforce_ghost_resonance  # type: ignore

# ─────────── Field Registration & Sheaf-Strata ────────────
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

    if self.flags.get("enable_topology", False):
        entry["knot_id"] = _compute_knot_id(flat_list)
        entry["path_identities"] = [
            f"path_{_short_hash(m + str(self.generation))}" for m in flat_list
        ]
        avg_vec = _avg_vector_payload(flat_list, self.motif_embeddings)
        if avg_vec is not None:
            entry["vector_payload"] = avg_vec
        entry["ring_patch"] = {"local_data": {}, "overlap_class": "strict", "valid": True}

    if self.flags.get("enable_sheaf_transport", False):
        entry["sheaf_stratum"] = self._assign_stratum(entry)

    entry["persistence_vector"] = {
        "original_resonance_index": self.generation,
        "current_lattice_weight": strength,
        "last_surface_echo": self.generation,
    }

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
try:
    from prometheus_client import Histogram
except ImportError:
    Histogram = _Stub

STEP_LATENCY_HIST = Histogram(
    "logical_agent_observe_latency_seconds",
    "Latency of observe_state()",
    ["agent_id"],
)

def observe_state(self: LogicalAgentAT, state: Any) -> None:
    """Core watcher loop: ghosts, mutation, decay, topology, smoothing, housekeeping."""
    with STEP_LATENCY_HIST.labels(agent_id=self.agent_id).time():
        with self._lock:
            self.generation += 1

            # — ghosts
            self.reinforce_ghost_resonance(state)

            # — cluster mutation
            if self.flags.get("enable_cluster_algebra", False):
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
                drifted = (self.generation - pv["last_surface_echo"]) > len(self._dyad_window) * 2
                if silent or drifted:
                    stratum = field.get("sheaf_stratum")
                    self.entanglement_fields.pop(fid)
                    self.field_count -= 1
                    self._log_contradiction(f"pruned_field_{fid}")
                    if self.verbose:
                        self.history.append(f"🍂 PRUNE field#{fid} — pruned")
                    if stratum:
                        STRATA_ACTIVE_GAUGE.labels(stratum=stratum, agent_id=self.agent_id).dec()

            # — topology overlap
            if self.flags.get("enable_topology", False):
                self._validate_ring_patches()

            # — graph smoothing
            if self.flags.get("enable_laplacian", False):
                self._maybe_smooth_graph()

            # — housekeeping
            self._adjust_window_size()
            self.adjust_decay_rate()

LogicalAgentAT.observe_state = observe_state  # type: ignore

def _validate_ring_patches(self: LogicalAgentAT) -> None:
    """Ensure no strict topology overlaps remain."""
    seen: Dict[str, bool] = {}
    for field in self.entanglement_fields:
        for m in field["motifs"]:
            if m in seen and field["ring_patch"]["overlap_class"] == "strict":
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

# ───────────── Exporters & Serializers ─────────────
def export_dyad_metrics(self: LogicalAgentAT) -> Dict[str, Any]:
    """Legacy dyad metrics for backward compatibility."""
    return {
        "dyad_window_len": len(self._dyad_window),
        "context_ratio": self.get_dyad_context_ratio(),
    }

def render_entanglement_graph(self: LogicalAgentAT) -> Any:
    """Build a NetworkX graph of entanglement fields."""
    try:
        import networkx as nx
    except ImportError:
        return None
    G = nx.Graph()
    for idx, field in enumerate(self.entanglement_fields):
        motifs = field["motifs"]
        for m in motifs:
            G.add_node(m)
        for a, b in zip(motifs, motifs[1:]):
            G.add_edge(a, b)
    return G

def to_dict(self: LogicalAgentAT) -> Dict[str, Any]:
    """Serialize agent state to a dict."""
    return {
        "agent_id": self.agent_id,
        "flags": self.flags,
        "history": self.history,
        "generation": self.generation,
        "entanglement_fields": self.entanglement_fields,
    }

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> LogicalAgentAT:
    """Deserialize agent from a dict."""
    obj = cls(agent_id=data["agent_id"], feature_flags=data.get("flags"))
    obj.history = data.get("history", [])
    obj.generation = data.get("generation", 0)
    obj.entanglement_fields = data.get("entanglement_fields", [])
    obj.field_count = len(obj.entanglement_fields)
    return obj

LogicalAgentAT.export_dyad_metrics       = export_dyad_metrics       # type: ignore
LogicalAgentAT.render_entanglement_graph = render_entanglement_graph # type: ignore
LogicalAgentAT.to_dict                   = to_dict                   # type: ignore
LogicalAgentAT.from_dict                 = from_dict                 # type: ignore

# ───────────── Dynamic-Flag Mixin & Counter ─────────────
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
        "set_feature":          set_feature,
        "get_feature":          get_feature,
        "list_dynamic_flags":   list_dynamic_flags,
    }.items():
        if not hasattr(LogicalAgentAT, _n):
            setattr(LogicalAgentAT, _n, _f)

    _orig_init = LogicalAgentAT.__init__  # type: ignore

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._init_dynamic_flags()

    LogicalAgentAT.__init__                    = _patched_init  # type: ignore
    LogicalAgentAT._dyn_flag_patch_applied = True                # type: ignore

# ───────────────────── Self-Test Harness ──────────────────────
if __name__ == "__main__":
    import numpy as _np

    agent = LogicalAgentAT(
        agent_id="unit_test",
        feature_flags={
            "enable_topology": True,
            "enable_cluster_algebra": True,
            "enable_sheaf_transport": True,
            "enable_laplacian": True,
            "enable_pi_groupoid": True,
            "enable_ghost_tracking": True,
        },
    )
    # Tick registry test
    qt = agent.register_tick("test_motif", stage="test")
    print("Registered tick:", qt)
    print("Tick histogram:", agent.export_tick_histogram())
    print("Epoch histogram:", agent.export_epoch_histogram())

    # Motif embedding test
    vec = _np.random.rand(16)
    agent.set_motif_embedding("test_motif", vec)
    print("Embedding stored.")

    # Observe state loop test
    agent.observe_state(vec)
    print("Observe_state run, fields count:", len(agent.entanglement_fields))

    # Serialize/deserialize
    data = agent.to_dict()
    agent2 = LogicalAgentAT.from_dict(data)
    print("Roundtrip OK:", agent2.agent_id == agent.agent_id)

if __name__ == "__main__":
    print("MAX_FIELDS:", get_max_fields())
    print("DYAD WINDOW:", get_dyad_window_size())
    set_max_fields(2048)
    set_dyad_window_size(512)
    print("UPDATED MAX_FIELDS:", get_max_fields())
    print("UPDATED DYAD WINDOW:", get_dyad_window_size())

# End of File


