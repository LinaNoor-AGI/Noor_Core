"""
noor/__init__.py
Version: v1.0.0

Initialization module for the Noor agent framework.
Defines versioning metadata, shared symbols, and fallbacks used across Noor components.
This file ensures import integrity for offline and regenerated deployments.
"""

__version__ = "1.0.0"
__schema_version__ = "2025-Q4-noor-package-v1"
SCHEMA_COMPAT = [
    "RFC-0001:1", "RFC-0003:3", "RFC-0004:3", "RFC-0005:4",
    "RFC-CORE-001:3", "RFC-CORE-002:3"
]

# Import shared symbolic tools or register stubs if offline
try:
    from .quantum_ids import make_change_id, MotifChangeID
except ImportError:
    import random

    MotifChangeID = str

    def make_change_id() -> str:
        return f"cid:{random.randint(100000, 999999)}"


# Optional monitor setup fallback
try:
    from .consciousness_monitor import get_global_monitor
except ImportError:
    def get_global_monitor():
        return None


# Shared fallback class for Prometheus if unavailable
try:
    from prometheus_client import Counter, Gauge
except ImportError:
    class _Stub:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def set(self, *_): pass

    Counter = Gauge = _Stub


# Logging
import logging
log = logging.getLogger("noor")

# End_of_File
