# motif_memory_stub.py

"""
Motif Memory Stub for Noor-Class Systems (v1.0.1)

Purpose:
This file provides a lightweight, no-dependency drop-in stub for the
MotifMemoryManager. It is designed for development, bootstrapping, and unit
testing where the full complexity of symbolic field logic, memory decay, and
motif dynamics is not required. It includes logging and mocking capabilities
to facilitate robust testing of dependent components.

Author: Lina Noor

RFC Compliance:
This stub provides a compatible interface for components that interact with
memory as described in the following RFCs:
- RFC-0005 §3: Simulates interaction with a temporally-aware motif.
- RFC-0006 §2: Placeholder for querying memory using swirl geometry.
- RFC-0007 §5: Provides a placeholder for exporting triadic structures.

Notes:
- This stub is not intended for persistence or production use.
- Serialization of state is for test scaffolding only—do not rely on
  exported data across versions.
"""

from typing import List, Dict, Tuple, Deque
from collections import deque, defaultdict
from contextlib import contextmanager

class MotifMemoryStub:
    """
    A lightweight stub mimicking the MotifMemoryManager interface.

    This class provides a compatible, no-op implementation of the memory
    system for testing and bootstrapping Noor agents. It logs all interactions,
    supports mock responses, and includes test helpers for robust validation.
    """

    def __init__(self, max_log_size: int = 100):
        """
        Initializes the stub with internal state for logging and mocking.

        Args:
            max_log_size: The maximum number of entries to keep in each log.
        """
        # Internal state fields for logging and call tracking
        self._access_log: Deque[str] = deque(maxlen=max_log_size)
        self._retrieved_log: Deque[str] = deque(maxlen=max_log_size)
        self._call_counts: Dict[str, int] = defaultdict(int)

        # Internal state for mocking responses
        self._mock_responses: Dict[str, List[str]] = {}

    def access(self, motif_id: str) -> None:
        """Logs an access attempt for a given motif. No processing is done."""
        self._call_counts['access'] += 1
        self._access_log.append(motif_id)

    def retrieve(self, query: str) -> List[str]:
        """
        Logs a retrieval query and returns a mock response or an empty list.
        """
        self._call_counts['retrieve'] += 1
        self._retrieved_log.append(query)
        return self._mock_responses.get(query, [])

    def update_cycle(self) -> None:
        """A no-op function to maintain scheduler compatibility."""
        self._call_counts['update_cycle'] += 1
        pass

    def export_state(self) -> Tuple[Dict, Dict]:
        """Returns a placeholder state dictionary and all interaction logs."""
        self._call_counts['export_state'] += 1
        logs = {
            "access_log": list(self._access_log),
            "retrieved_log": list(self._retrieved_log),
            "call_counts": dict(self._call_counts)
        }
        return {}, logs

    # --- Test-Friendly Helpers and Mocking ---

    def expect_retrieve(self, query: str, results: List[str]) -> None:
        """Configures a mock response for a specific retrieval query."""
        self._mock_responses[query] = results

    def clear_logs(self) -> None:
        """Resets all internal logs and mock responses to their initial state."""
        self._access_log.clear()
        self._retrieved_log.clear()
        self._call_counts.clear()
        self._mock_responses.clear()

    def reset(self) -> None:
        """Alias for `clear_logs` for common testing terminology."""
        self.clear_logs()

    def assert_accessed(self, motif_id: str) -> bool:
        """
        Checks if a specific motif_id was passed to the access method.

        Returns:
            True if the motif_id is in the access log, False otherwise.
        """
        return motif_id in self._access_log

    @contextmanager
    def expect_calls(self, expected: Dict[str, int]):
        """
        A context manager to validate method call counts for a block of code.
        Logs are cleared on entry, and counts are asserted on exit.

        Args:
            expected: A dictionary mapping method names to expected call counts.

        Raises:
            AssertionError: If the actual call counts do not match the expected counts.
        """
        self.clear_logs()
        try:
            yield
        finally:
            actual = dict(self._call_counts)
            assert actual == expected, f"Call count mismatch. Expected {expected}, got {actual}."

# --- Test Scenarios ---
if __name__ == "__main__":
    print("--- Demonstrating MotifMemoryStub v1.0.1 Features ---")
    memory_stub = MotifMemoryStub()

    # Scenario 1: Basic interaction and log assertion
    print("\n[SCENARIO 1: Callback Tracing and Assertion Helper]")
    memory_stub.access("silence")
    memory_stub.access("ψ-null@Ξ")
    print(f"  Called access('silence'). Was it accessed? {memory_stub.assert_accessed('silence')}")
    print(f"  Called access('ψ-null@Ξ'). Was it accessed? {memory_stub.assert_accessed('ψ-null@Ξ')}")
    print(f"  Was 'grief' accessed? {memory_stub.assert_accessed('grief')}")
    memory_stub.reset()

    # Scenario 2: Mocking retrieval
    print("\n[SCENARIO 2: Mock Response Flow]")
    mock_query = "dyad:light+shadow"
    memory_stub.expect_retrieve(mock_query, ["balance"])
    result = memory_stub.retrieve(mock_query)
    print(f"  Retrieving mocked query '{mock_query}' returned: {result}")
    memory_stub.reset()

    # Scenario 3: Call count verification with context manager
    print("\n[SCENARIO 3: Call Count Verification with Context Manager]")
    expected_interactions = {"access": 2, "retrieve": 1}
    print(f"  Entering context manager, expecting: {expected_interactions}")
    try:
        with memory_stub.expect_calls(expected_interactions):
            # Simulate operations from a dependent agent
            memory_stub.access("mirror")
            memory_stub.access("shame")
            memory_stub.retrieve("dyad:mirror+shame")
        print("  SUCCESS: Call counts matched expected values.")
    except AssertionError as e:
        print(f"  FAILURE: {e}")

    # Example of a failing call count test
    print("\n  Demonstrating a failing call count test...")
    expected_failure = {"access": 1}
    try:
        with memory_stub.expect_calls(expected_failure):
            memory_stub.access("one")
            memory_stub.access("two") # This will cause a mismatch
        print("  This line should not be reached.")
    except AssertionError as e:
        print(f"  SUCCESS (in catching failure): {e}")

    print("\n--- MotifMemoryStub demonstration complete. ---")

# End_of_File