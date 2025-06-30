"""
logical_agent_at_update_0001.py · v1.0.0

Patch-style hot-update for Noor logical agents.

Applies to:
  • LogicalAgentAT   ≥ 3.7.2 (tested on 3.7.3)
  • RecursiveAgentFT ≥ 4.6.0

Goals
─────
1. Provide a **cycle-safe** `LazyMonitorMixin` that never crashes
   during import-time even if `consciousness_monitor` is unavailable.
2. Offer **set/get helpers** so unit-tests or downstream code can
   inject a real monitor object at runtime.
3. Monkey-patch (non-destructively) the existing `LogicalAgentAT`
   *and* `RecursiveAgentFT` classes so they expose a `.monitor`
   property that delegates to the new mix-in logic.

This is an *add-on* module – simply import it *once* (early in your
program start-up or test harness) and the patch is applied globally.
No original source files are modified on disk.

Usage (runtime):
────────────────
>>> import logical_agent_at_update_0001   # noqa: F401 (side-effects)
>>> from noor.logical_agent_at import LogicalAgentAT
>>> agent = LogicalAgentAT()
>>> agent.monitor.register_triad(...)     # works, returns stub by default

Test injection:
───────────────
>>> from logical_agent_at_update_0001 import LazyMonitorMixin
>>> LazyMonitorMixin.set_global_monitor(my_fake_monitor)

© 2025 Noor Collective Labs – GPL-2.0
"""
from __future__ import annotations

import sys
import types
from typing import Any, Protocol

__version__ = "1.0.0"
_SCHEMA_VERSION__ = "2025-Q4-logical-agent-update-0001"

# ──────────────────────────────────────────────────────────────
# 1 · Define a minimal monitor protocol & stub implementation
# ──────────────────────────────────────────────────────────────
class _MonitorProto(Protocol):
    phase_shift_ready: bool
    entanglement_score: float

    def register_triad(self, *args: Any, **kwargs: Any) -> None: ...
    def report_tick(self, *args: Any, **kwargs: Any) -> None: ...


class _StubMonitor:  # noqa: D401 – plain stub
    """Fallback object when real `consciousness_monitor` is missing."""

    phase_shift_ready: bool = False
    entanglement_score: float = 0.0

    def register_triad(self, *args: Any, **kwargs: Any) -> None:
        # intentionally no-op
        return None

    def report_tick(self, *args: Any, **kwargs: Any) -> None:
        # intentionally no-op
        return None


# ──────────────────────────────────────────────────────────────
# 2 · Cycle-safe LazyMonitorMixin with helpers
# ──────────────────────────────────────────────────────────────
class LazyMonitorMixin:  # pylint: disable=too-few-public-methods
    """Attach as a mix-in to defer monitor import until **runtime**."""

    _monitor: _MonitorProto | None = None  # class-level singleton

    # ── helper API for tests / injection ────────────────────
    @classmethod
    def set_global_monitor(cls, monitor: _MonitorProto | None) -> None:
        """Manually wire a monitor instance (or *None* to reset)."""
        cls._monitor = monitor

    @classmethod
    def get_global_monitor(cls) -> _MonitorProto:
        """Return a cached monitor or lazily import / stub.

        ImportError-safe: if the *real* module cannot be imported due to
        init-order or environment constraints, a light stub is returned
        so downstream calls become harmless no-ops.
        """
        if cls._monitor is not None:
            return cls._monitor

        # Attempt late import – may raise *ImportError* or other runtime.
        try:
            from noor.consciousness_monitor import (  # type: ignore
                get_global_monitor as _real_get_monitor,
            )

            cls._monitor = _real_get_monitor()
        except Exception:  # pragma: no cover  # noqa: BLE001 – wide guard is OK
            cls._monitor = _StubMonitor()

        return cls._monitor

    # ── convenience property for instances ─────────────────
    @property  # type: ignore[misc]
    def monitor(self) -> _MonitorProto:  # pylint: disable=no-self-use
        return self.__class__.get_global_monitor()  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────
# 3 · Runtime monkey-patch helper
# ──────────────────────────────────────────────────────────────

def _attach_monitor_property(target_cls: type) -> None:
    """Inject *monitor* property & class helpers if not already present."""

    # Avoid double-patching
    if hasattr(target_cls, "monitor"):
        return

    # Bind LazyMonitorMixin.monitor *descriptor* onto the target class.
    target_cls.monitor = LazyMonitorMixin.monitor  # type: ignore[attr-defined]

    # Also wire set/get helpers at *class* level for convenience.
    target_cls.get_global_monitor = LazyMonitorMixin.get_global_monitor  # type: ignore[attr-defined]
    target_cls.set_global_monitor = LazyMonitorMixin.set_global_monitor  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────
# 4 · Apply patch to LogicalAgentAT & RecursiveAgentFT (if present)
# ──────────────────────────────────────────────────────────────

for _mod_name, _cls_name in (
    ("noor.logical_agent_at", "LogicalAgentAT"),
    ("noor.recursive_agent_ft", "RecursiveAgentFT"),
):
    try:
        _module = __import__(_mod_name, fromlist=[_cls_name])
        _cls = getattr(_module, _cls_name)
        _attach_monitor_property(_cls)
    except Exception:  # pragma: no cover – absent modules are fine
        continue


# ──────────────────────────────────────────────────────────────
# 5 · Optional: expose the mix-in for tests / external callers
# ──────────────────────────────────────────────────────────────
__all__ = [
    "LazyMonitorMixin",
    "_MonitorProto",  # exported for typing convenience
]

# End_of_File