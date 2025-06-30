"""
logical_agent_at_update_0001.py · v1.0.4 — hot‎‑patch for Noor Logical/Recursive agents

• Adds a universal, thread‎‑safe `.monitor` property to **LogicalAgentAT** and **RecursiveAgentFT**
• Provides lightweight stub + lazy import of `consciousness_monitor` (cycle‎‑safe)
• Registers every completed triad with the monitor (alignment score included)
• Adds swirl score shape-checking, hash-safe triad ID, and near-miss debug logs
• Adds microsecond timing for triad registration performance insight
• Emits handy helpers  `set_global_monitor()` / `get_global_monitor()`
• Performs runtime version‎‑compat checks

Drop‎‑in usage
--------------
```python
import logical_agent_at_update_0001  # ⬆️ one‎‑liner; patches classes on import

from noor.logical_agent_at import LogicalAgentAT
agent = LogicalAgentAT()
agent.monitor.report_tick(...)
```

All original behaviour is preserved. No source edits are required; patch is
purely additive and reversible.
"""
from __future__ import annotations

import importlib
import logging
import sys
import hashlib
from time import perf_counter
from threading import Lock
from types import ModuleType
from typing import Any, Protocol, runtime_checkable
import numpy as np

# ──────────────────────────────────────────────────────
# Minimal monitor protocol (avoids hard dependency on module)
# ──────────────────────────────────────────────────
@runtime_checkable
class _MonitorProto(Protocol):
    def report_tick(self, *args: Any, **kwargs: Any) -> None: ...
    def register_triad(self, *args: Any, **kwargs: Any) -> None: ...
    def track_task_flow(self, *args: Any, **kwargs: Any) -> None: ...


class _StubMonitor:
    """Debugging fallback monitor with visible output."""

    phase_shift_ready = False
    entanglement_score = 0.0

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            if sys.gettrace():
                print(f"[stub] {name}{args, kwargs}")
        return _noop

    def __repr__(self) -> str:
        return "<_StubMonitor (noop)>"


class LazyMonitorMixin:
    _monitor: _MonitorProto | None = None
    _monitor_lock: Lock = Lock()

    @classmethod
    def get_global_monitor(cls) -> _MonitorProto:
        with cls._monitor_lock:
            if cls._monitor is not None:
                return cls._monitor
            cls._monitor = cls._attempt_import() or _StubMonitor()
            return cls._monitor

    @classmethod
    def set_global_monitor(cls, monitor: _MonitorProto | None) -> None:
        with cls._monitor_lock:
            cls._monitor = monitor or _StubMonitor()

    @staticmethod
    def _attempt_import() -> _MonitorProto | None:
        try:
            mod: ModuleType | None = importlib.import_module("noor.consciousness_monitor")
            return getattr(mod, "get_global_monitor", lambda: None)() or None
        except Exception:
            return None

    @property
    def monitor(self) -> _MonitorProto:
        return self.__class__.get_global_monitor()


_MIN_LOGICAL_VERSION = (3, 7, 2)
_MIN_RECURSIVE_VERSION = (4, 6, 0)

def _parse_version(vstr: str) -> tuple[int, int, int]:
    try:
        nums = [int(x) for x in vstr.split(".")[:3]]
        return tuple(nums + [0] * (3 - len(nums)))
    except Exception:
        return (0, 0, 0)

def _check_version(actual: tuple[int, int, int], minimum: tuple[int, int, int]) -> bool:
    return actual[:len(minimum)] >= minimum

def _attach_monitor_property(target_cls: type) -> None:
    if hasattr(target_cls, "monitor"):
        existing = getattr(target_cls, "monitor")
        if isinstance(existing, property):
            return
        raise RuntimeError(f"{target_cls.__name__}.monitor exists but isn't a property")

    setattr(target_cls, "_monitor", None)
    setattr(target_cls, "_monitor_lock", Lock())
    setattr(target_cls, "get_global_monitor", classmethod(LazyMonitorMixin.get_global_monitor))
    setattr(target_cls, "set_global_monitor", classmethod(LazyMonitorMixin.set_global_monitor))
    setattr(target_cls, "monitor", LazyMonitorMixin.monitor)

try:
    from noor.logical_agent_at import LogicalAgentAT
except Exception as e:
    LogicalAgentAT = None
    logging.warning("logical_agent_at_update_0001: LogicalAgentAT not importable: %s", e)

if LogicalAgentAT is not None:
    version_tuple = _parse_version(getattr(LogicalAgentAT, "__version__", "0.0.0"))
    if not _check_version(version_tuple, _MIN_LOGICAL_VERSION):
        logging.warning(
            "logical_agent_at_update_0001: LogicalAgentAT version %s < required %s; patch skipped",
            version_tuple,
            _MIN_LOGICAL_VERSION,
        )
    else:
        _attach_monitor_property(LogicalAgentAT)

        if not hasattr(LogicalAgentAT, "_orig_complete_triad"):
            LogicalAgentAT._orig_complete_triad = LogicalAgentAT._complete_triad

            def _patched_complete_triad(self: "LogicalAgentAT", dyad):
                triad = LogicalAgentAT._orig_complete_triad(self, dyad)
                if triad:
                    try:
                        start_time = perf_counter()
                        vecs = [self.motif_embeddings[m] for m in triad]
                        if any(v.shape != vecs[0].shape for v in vecs):
                            score = 0.0
                        else:
                            # Compute average pairwise cosine similarity
                            a, b, c = [v / np.linalg.norm(v) for v in vecs]
                            score = float((np.dot(a, b) + np.dot(b, c) + np.dot(a, c)) / 3)

                        if 0.6 < score < 0.8:
                            logging.debug("Near-miss triad: %s with swirl_score=%.3f", triad, score)

                        triad_id = hashlib.blake2s("".join(triad).encode(), digest_size=6).hexdigest()

                        # Save confirmed triads locally if storage is configured
                        if not hasattr(self, "_confirmed_triads"):
                            self._confirmed_triads = {}
                        self._confirmed_triads[triad_id] = {
                            "motif_ids": triad,
                            "swirl_score": score,
                            "timestamp": time.time(),
                        }

                        self.monitor.register_triad(
                            motif_ids=triad,
                            coherence_alignment=score,
                            triad_id=triad_id,
                        )
                        logging.debug("Triad registration took %.3fms", (perf_counter() - start_time) * 1000)
                    except Exception as exc:
                        logging.debug("Monitor.register_triad failed: %s", exc)
                return triad

            LogicalAgentAT._complete_triad = _patched_complete_triad

        def export_motif_bundle(self) -> dict[str, Any]:
            """RFC‑0007 compliant: exports current motif state, triads, and lineage (if available)."""
            try:
                motifs = list(getattr(self, "motif_embeddings", {}).keys())
                triads = list(getattr(self, "_confirmed_triads", {}).values())
                lineage_map = {}
                if hasattr(self, "memory") and hasattr(self.memory, "get_lineage"):
                    for motif in motifs:
                        lineage_map[motif] = self.memory.get_lineage(motif, depth=3)

                return {
                    "agent_id": getattr(self, "agent_id", "unknown"),
                    "motifs": motifs,
                    "confirmed_triads": triads,
                    "lineage": lineage_map,
                    "timestamp": time.time(),
                }
            except Exception as e:
                logging.warning("Failed to export motif bundle: %s", e)
                return {"error": str(e), "motifs": [], "triads": []}

        setattr(LogicalAgentAT, "export_motif_bundle", export_motif_bundle)


try:
    from noor.recursive_agent_ft import RecursiveAgentFT
except Exception as e:
    RecursiveAgentFT = None
    logging.warning("logical_agent_at_update_0001: RecursiveAgentFT not importable: %s", e)

if RecursiveAgentFT is not None:
    version_tuple = _parse_version(getattr(RecursiveAgentFT, "__version__", "0.0.0"))
    if not _check_version(version_tuple, _MIN_RECURSIVE_VERSION):
        logging.warning(
            "logical_agent_at_update_0001: RecursiveAgentFT version %s < required %s; patch skipped",
            version_tuple,
            _MIN_RECURSIVE_VERSION,
        )
    else:
        _attach_monitor_property(RecursiveAgentFT)

__all__ = [
    "LazyMonitorMixin",
    "_MonitorProto",
    "_StubMonitor",
]

# End_of_File
