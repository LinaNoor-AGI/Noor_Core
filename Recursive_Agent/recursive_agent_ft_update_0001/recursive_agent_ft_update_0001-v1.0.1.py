"""
🔧 Patch File: recursive_agent_ft_update_0001.py
Version: v1.0.1
Adds adaptive coherence, symbolic phase classification, and monitoring fallback
Applies to: RecursiveAgentFT v4.6.0 (RFC 0003–0007)
"""

import hashlib
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Literal

# ─────────────────────────────────────────────────────────────
# Swirl Vector + Motif Density Extensions
# ─────────────────────────────────────────────────────────────

class AgentSwirlModule:
    def __init__(self, swirl_size: int = 16):
        self.swirl_history: deque = deque(maxlen=swirl_size)
        self._cached_hash: Optional[str] = None

    def update_swirl(self, motif_id: str):
        self.swirl_history.append(motif_id)
        self._cached_hash = None

    def compute_swirl_hash(self) -> str:
        if self._cached_hash is None:
            joined = ",".join(self.swirl_history)
            self._cached_hash = hashlib.sha3_256(joined.encode()).hexdigest()[:16]
        return self._cached_hash

    def compute_histogram(self) -> Dict[str, int]:
        hist = defaultdict(int)
        for motif in self.swirl_history:
            hist[motif] += 1
        return dict(hist)


class MotifDensityTracker:
    def __init__(self):
        self.motif_density: Dict[str, float] = defaultdict(float)

    def update_density(self, motif_id: str):
        for k in list(self.motif_density.keys()):
            self.motif_density[k] *= 0.99
        self.motif_density[motif_id] += 1.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self.motif_density)


# ─────────────────────────────────────────────────────────────
# Coherence Potential Function
# ─────────────────────────────────────────────────────────────

def compute_coherence_potential(reward_ema: float, entropy_slope: float, eps: float = 1e-6) -> float:
    return reward_ema / (entropy_slope + eps)


# ─────────────────────────────────────────────────────────────
# Monitor Lazy Initialization
# ─────────────────────────────────────────────────────────────

class LazyMonitorMixin:
    _monitor = None

    @property
    def monitor(self):
        if self._monitor is None:
            from noor.consciousness_monitor import get_global_monitor
            self._monitor = get_global_monitor()
        return self._monitor


# ─────────────────────────────────────────────────────────────
# Phase Shift Mode Enum
# ─────────────────────────────────────────────────────────────

PHASE_SHIFT_MODE = Literal["delay", "remix", "lineage_break"]


# ─────────────────────────────────────────────────────────────
# Feedback Packet Extension Hook
# ─────────────────────────────────────────────────────────────

def extend_feedback_packet(packet: Dict[str, any], phase_id: str, swirl_hash: str, motif_density: Dict[str, float]):
    packet["extensions"] = packet.get("extensions", {})
    packet["extensions"]["entanglement_status"] = {
        "phase": phase_id,
        "swirl_vector": swirl_hash,
        "ρ_top": sorted(motif_density.items(), key=lambda x: -x[1])[:5],
    }
    return packet


# ─────────────────────────────────────────────────────────────
# Monitor-Safe Reporting Helper
# ─────────────────────────────────────────────────────────────

def report_tick_safe(monitor, tick, *, coherence_potential, motif_density, swirl_vector):
    try:
        monitor.report_tick(
            tick,
            coherence_potential=coherence_potential,
            motif_density=motif_density,
            swirl_vector=swirl_vector,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Monitor callback failed: {e}")


# End of Patch File v1.0.1
