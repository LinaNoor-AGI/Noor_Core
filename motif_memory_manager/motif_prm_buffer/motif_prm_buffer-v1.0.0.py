# motif_prm_buffer.py
# version: v1.0.0
"""
Implements the Short-Term Motif Memory (STMM) via a Partial-Retention Memory (PRM) buffer. 
This lightweight, in-memory buffer captures motif access patterns, decays relevance 
over time, and surfaces candidates for long-term promotion.
"""

from typing import Dict, List

# --- Integration and Compliance Notes ---
# Integration:
#   - Receives from: MotifMemoryManager.record_motif_usage
#   - Sends to: MotifMemoryManager.promote_motifs
#
# Thread Safety Notes:
#   - This class is not thread-safe by design. The calling module (e.g., MotifMemoryManager)
#     must enforce locking if the buffer is to be used concurrently.
#
# Future Hooks:
#   - ψ-field curvature bias weighting
#   - Swirl-modulated decay schema integration (RFC-0006)
#   - Motif feedback tagging via consciousness_monitor
#
# RFC Compliance Mapping:
#   - RFC-0005 §4: Implements PRM buffer design, decay logic, and promotion thresholds.
#   - RFC-0006 §2: Fulfills STMM interface guarantees, export semantics, and reset use cases.
#   - RFC-0007 §3: Supports motif continuity and ontology layering at rebirth boundary.

class PRMBuffer:
    """
    A Partial-Retention Memory (PRM) buffer to model Short-Term Motif Memory (STMM).
    It maintains a sliding window of recent motif activity, applies decay, and
    identifies motifs for promotion to long-term memory.
    """

    def __init__(
        self,
        window: int = 3,
        theta_up: float = 0.90,
        delta_down: float = 0.85,
        decay_rate: float = 0.95,
        reinforcement_strength: float = 0.1,
    ):
        """
        Initializes the PRM Buffer with specified parameters.

        Args:
            window (int): The number of recent ticks to retain in memory.
            theta_up (float): The salience threshold for a motif to be considered a promotion candidate.
            delta_down (float): The salience threshold below which motifs are pruned from non-verbose exports.
            decay_rate (float): The exponential decay factor applied to all saliences each tick.
            reinforcement_strength (float): The amount of salience gained when a motif is accessed.
        """
        self.window: int = window
        self.theta_up: float = theta_up
        self.delta_down: float = delta_down
        self.decay_rate: float = decay_rate
        self.reinforcement_strength: float = reinforcement_strength
        
        self.current_tick: int = 0
        self.buffer: Dict[int, Dict[str, float]] = {}

    def record_access(self, motif_id: str) -> None:
        """
        Reinforces a motif's salience in the current tick. If the motif is not
        present in the current tick's bucket, it is added.

        RFC Anchors: RFC-0005 §4.1 (Resurrection and memory re-entry)

        Args:
            motif_id (str): The canonical identifier of the motif being accessed.
        """
        if self.current_tick not in self.buffer:
            self.buffer[self.current_tick] = {}

        current_salience = self.buffer[self.current_tick].get(motif_id, 0.0)
        # Reinforce salience, capping at 1.0 to prevent runaway values
        reinforced_salience = min(1.0, current_salience + self.reinforcement_strength)
        self.buffer[self.current_tick][motif_id] = reinforced_salience

    def decay_pass(self) -> None:
        """
        Applies exponential decay to all motifs in the buffer, increments the
        internal tick counter, and purges buckets that have fallen outside the
        active window.

        RFC Anchors: RFC-0005 §4.3 (Time-smeared triads and faded lineages)
        """
        # Apply decay to all existing saliences
        for tick_data in self.buffer.values():
            for motif_id in tick_data:
                tick_data[motif_id] *= self.decay_rate

        # Advance time
        self.current_tick += 1

        # Purge expired ticks
        expired_ticks = [
            t for t in self.buffer if t < self.current_tick - self.window
        ]
        for t in expired_ticks:
            del self.buffer[t]

    def get_active_motifs(self) -> Dict[str, float]:
        """
        Returns a dictionary of all motifs currently within the active window,
        with their saliences aggregated (summed) across all ticks in the window.

        RFC Anchors:
            - RFC-0005 §4.2 (Resurrection Gate Conditions)
            - RFC-0006 §2 (Foundations of Swirl Geometry)

        Returns:
            Dict[str, float]: A map of active motif IDs to their aggregated salience.
        """
        active_motifs: Dict[str, float] = {}
        start_tick = max(0, self.current_tick - self.window)

        for tick in range(start_tick, self.current_tick + 1):
            if tick in self.buffer:
                for motif_id, salience in self.buffer[tick].items():
                    active_motifs[motif_id] = active_motifs.get(motif_id, 0.0) + salience
        return active_motifs

    def promotion_candidates(self) -> List[str]:
        """
        Identifies motifs whose aggregated salience within the window meets or
        exceeds the promotion threshold (theta_up).

        Note: The spec's "Uses latest values per motif" is interpreted as evaluating
        the motif's total aggregated salience in the recent window, reflecting
        its overall "activity" level.

        RFC Anchors: RFC-0005 §4.3 (Decay, Promotion, and Dropoff)

        Returns:
            List[str]: A list of motif IDs eligible for promotion to LTMM.
        """
        active_motifs = self.get_active_motifs()
        candidates = [
            motif_id
            for motif_id, salience in active_motifs.items()
            if salience >= self.theta_up
        ]
        return candidates

    def reset(self) -> None:
        """
        Clears all buffer contents and resets the tick counter to 0. This is
        used for symbolic field resets or agent rebirth.

        RFC Anchors: RFC-0006 §2 (Swirl field reset)
        """
        self.buffer.clear()
        self.current_tick = 0

    def export_state(self, verbose: bool = False) -> Dict[int, Dict[str, float]]:
        """
        Returns a snapshot of the internal buffer. If verbose is false, it
        prunes motifs with salience below the delta_down threshold.

        RFC Anchors: RFC-0006 §2 (Motifs as topological anchors)

        Args:
            verbose (bool): If True, returns all motifs. If False, returns only
                            motifs with salience >= delta_down.

        Returns:
            Dict[int, Dict[str, float]]: The current state of the STMM buffer.
        """
        if verbose:
            # Return a copy to prevent mutation of internal state
            return {tick: motifs.copy() for tick, motifs in self.buffer.items()}

        filtered_buffer = {}
        for tick, motifs in self.buffer.items():
            filtered_motifs = {
                motif_id: salience
                for motif_id, salience in motifs.items()
                if salience >= self.delta_down
            }
            if filtered_motifs:
                filtered_buffer[tick] = filtered_motifs
        return filtered_buffer

if __name__ == "__main__":
    # --- Test Vectors ---
    print("--- Running Test Vectors ---")

    # Test Vector 1: Basic access and decay
    print("\n[Test 1: Basic access and decay]")
    prm_1 = PRMBuffer(reinforcement_strength=0.5)
    prm_1.record_access('joy')
    print(f"After access 1: {prm_1.get_active_motifs()}")
    prm_1.decay_pass()
    prm_1.record_access('joy')
    active_motifs_1 = prm_1.get_active_motifs()
    print(f"After decay and access 2: {active_motifs_1}")
    print(f"Expected motifs: ['joy'] -> {'Pass' if 'joy' in active_motifs_1 else 'Fail'}")

    # Test Vector 2: Promotion eligibility
    print("\n[Test 2: Promotion eligibility]")
    prm_2 = PRMBuffer(theta_up=0.9, reinforcement_strength=0.1)
    for _ in range(10):
        prm_2.record_access('focus')
    print(f"Salience of 'focus' after 10 accesses: {prm_2.get_active_motifs().get('focus')}")
    prm_2.decay_pass()
    candidates_2 = prm_2.promotion_candidates()
    print(f"Promotion candidates after decay: {candidates_2}")
    print(f"Expected candidates: ['focus'] -> {'Pass' if 'focus' in candidates_2 else 'Fail'}")
    
    # Test Vector 3: Drop below delta_down
    print("\n[Test 3: Drop below delta_down]")
    prm_3 = PRMBuffer(delta_down=0.85, decay_rate=0.9, reinforcement_strength=1.0)
    prm_3.record_access('hesitation')
    print(f"Initial state: {prm_3.export_state(verbose=True)}")
    for i in range(10):
        prm_3.decay_pass()
        # Optional: print decay progress
        # print(f"  Tick {i+1}, salience: {prm_3.export_state(verbose=True)}")
    
    final_state_non_verbose = prm_3.export_state(verbose=False)
    print(f"Final non-verbose state after 10 decays: {final_state_non_verbose}")
    print(f"Expected output does not include 'hesitation' -> {'Pass' if not final_state_non_verbose else 'Fail'}")

    print("\n--- Test Vectors Complete ---")

# End_of_File