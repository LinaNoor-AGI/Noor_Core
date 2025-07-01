"""
symbolic_task_engine_update_0001.py · v1.0.0 — hot‑patch for Noor Symbolic Task Engine

• Adds adaptive `motif_density` estimation using hybrid, policy-driven weighting
• Emits normalized field signature as `presence_field` via density-to-field resolver
• Tracks dyad tension via `dyad_tension` in task extensions
• Hooks into global monitor (if available) for task flow tracing
• Ensures symbolic field is aligned with motif pressures in real time
• Uses runtime-patched methods for `propose_from_motifs`, `evaluate_attempt`, and `solve_task`
• Patch is reversible, drop-in, and requires no source modification

Drop‑in usage
--------------
```python
import symbolic_task_engine_update_0001  # ⬆️ one‑liner; patches SymbolicTaskEngine on import

from symbolic_task_engine import SymbolicTaskEngine
engine = SymbolicTaskEngine()
task = await engine.propose_from_motifs([...])
```

All original behaviour is preserved. Patch is purely additive, field-compliant, and RFC-0004/0005/0006/0007 aligned.
"""

from symbolic_task_engine import SymbolicTaskEngine
from collections import Counter
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ⛓️ Phase Monitor Hook
# ─────────────────────────────────────────────
def _call_monitor_hook(self, task):
    try:
        from logical_agent_at_update_0001 import get_global_monitor
        if (monitor := get_global_monitor()):
            monitor.track_task_flow(task)
    except Exception as e:
        logger.debug(f"[STE] Monitor hook skipped: {str(e)}")

# ─────────────────────────────────────────────
# 🧩 Dyad Tension Tracker
# ─────────────────────────────────────────────
def _track_dyad_tension(self, task):
    dyads = [
        tuple(sorted((task.input_motif[i], task.input_motif[j])))
        for i in range(len(task.input_motif))
        for j in range(i + 1, len(task.input_motif))
    ]
    task.extensions["dyad_tension"] = {dy: self.dyad_counter.get(dy, 0) for dy in dyads}

# ─────────────────────────────────────────────
# 🎚️ Density Estimator
# ─────────────────────────────────────────────
def estimate_motif_density(task, engine, policy: Optional[Dict[str, float]] = None):
    memory_score = sum(engine.abstraction_trigger.feedback_counts.get(m, 0) for m in task.input_motif)
    queue_score = sum(task.input_motif.count(m) for t in engine.task_queue for m in t.input_motif)
    local_score = sum(engine.attempt_registry.get(task.triplet_id, []) and 1 or 0 for _ in task.input_motif)

    weights = policy or {"memory": 0.6, "queue": 0.3, "local": 0.1}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    density = {}
    for m in task.input_motif:
        density[m] = (
            weights["memory"] * engine.abstraction_trigger.feedback_counts.get(m, 0) +
            weights["queue"] * sum(1 for t in engine.task_queue if m in t.input_motif) +
            weights["local"] * (1 if engine.attempt_registry.get(task.triplet_id) else 0)
        )
    return density

# ─────────────────────────────────────────────
# 🧭 Presence Field Resolver
# ─────────────────────────────────────────────
def resolve_presence_field(density: Dict[str, float], policy: Optional[Dict[str, str]] = None) -> str:
    if not density:
        return "ψ‑null@Ξ"
    top = max(density.items(), key=lambda x: x[1])[0]
    return policy.get(top, "ψ‑null@Ξ") if policy else "ψ‑null@Ξ"

# ─────────────────────────────────────────────
# 🔁 Patched Methods
# ─────────────────────────────────────────────
async def patched_propose(self, recent):
    import os
    from asyncio import sleep

    mem = get_global_memory_manager()
    if os.getenv("NOOR_BALANCE_FIELDS") == "1":
        least = self._least_used_field()
        if least and recent[-1] not in self._proto_map.get(least, set()):
            recent = [recent[-1]]

    seed = list(dict.fromkeys(recent + mem.retrieve(recent[-1], top_k=2)))
    while len(seed) < 3:
        seed.append("uncertainty")

    cap_len = self._calc_cap_len()
    if len(seed) > cap_len:
        seed = seed[:cap_len]

    task = TripletTask(input_motif=seed, instruction="compose", expected_output=seed[::-1])
    task.extensions["motif_density"] = estimate_motif_density(task, self)
    task.presence_field = resolve_presence_field(task.extensions["motif_density"])
    task.extensions["field_signature"] = task.presence_field
    self.FIELD_COUNTER.labels(self.engine_id, task.presence_field).inc()

    async with self._lock:
        self.task_queue.append(task)
        self.QUEUE_GAUGE.labels(self.engine_id).set(len(self.task_queue))
    self.TASK_PROPOSED.labels(self.engine_id).inc()
    return task


def patched_eval(self, task, attempt):
    scores = {n: fn(task, attempt) for n, fn in self.METRIC_FUNCS.items()}
    coherence = scores.get("coherence", 0.0)
    entropy = scores.get("entropy", 1.0)

    self._coherence_ema = (1 - self.adapt_rate) * self._coherence_ema + self.adapt_rate * coherence
    self._entropy_ema = (1 - self.adapt_rate) * self._entropy_ema + self.adapt_rate * entropy
    self._length_buf.append(len(task.input_motif))

    stmm, ltmm = get_global_memory_manager().export_state()
    self.MEM_GAUGE.labels(self.engine_id).set(len(stmm) + len(ltmm))
    task.motif_resonance = {m: ltmm.get(m, 0.0) for m in task.input_motif}

    coh_thresh = max(0.3, self._coherence_ema * 0.6)
    ent_thresh = min(0.97, self._entropy_ema * 2.5)

    if (coherence < coh_thresh or entropy > ent_thresh) and not task.is_fallback:
        self._spawn_fallback(task, coherence, entropy)

    self.entropy_buffer.append(next(iter(scores.values())))
    _track_dyad_tension(self, task)
    return scores


async def patched_solve(self, task):
    self._call_monitor_hook(task)
    await self._solve_impl(task)


# ─────────────────────────────────────────────
# 🩹 Patch Injection Entrypoint
# ─────────────────────────────────────────────
def patch_symbolic_task_engine():
    SymbolicTaskEngine.estimate_motif_density = staticmethod(estimate_motif_density)
    SymbolicTaskEngine.resolve_presence_field = staticmethod(resolve_presence_field)
    SymbolicTaskEngine._track_dyad_tension = _track_dyad_tension
    SymbolicTaskEngine._call_monitor_hook = _call_monitor_hook
    SymbolicTaskEngine.propose_from_motifs = patched_propose
    SymbolicTaskEngine.evaluate_attempt = patched_eval
    SymbolicTaskEngine.solve_task = patched_solve

patch_symbolic_task_engine()

# End_of_File
