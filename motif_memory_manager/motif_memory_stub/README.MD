# 🧠 MotifMemoryStub v1.0.1 — *Testing Utility for Symbolic Memory Integration*

![version](https://img.shields.io/badge/version-1.0.1-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Lightweight test stub for the `MotifMemoryManager` interface. Enables controlled testing of symbolic agents and task systems by mocking motif access and retrieval logic.

---

## 📖 Overview

`MotifMemoryStub` is designed to simulate the memory access interface used by Noor-class agents. It includes:

- Controlled access and retrieval logging
- Declarative mock-response injection
- Call count tracking and validation
- Safe introspection of symbolic memory requests
- Context-managed testing for call expectations

---

## 🌟 Features

| Feature                 | Description                                                           |
|-------------------------|-----------------------------------------------------------------------|
| **Access Logging**      | Records all motif access attempts                                     |
| **Mocked Retrievals**   | Inject and return canned results to simulate memory hits              |
| **Call Count Metrics**  | Automatically tracks usage frequency for all key methods             |
| **Scoped Expectation**  | Validates expected method calls within test blocks                   |
| **Pure Python**         | No dependencies—works anywhere Python runs                           |
| **RFC-Compatible**      | Aligned with RFC-0005, RFC-0006, and RFC-0007                        |

---

## 🧠 Motif Access Lifecycle

```mermaid
flowchart TD
    A[Test Setup] --> B[Inject Stub into Component]
    B --> C[Call access or retrieve]
    C --> D[Log interaction in internal state]
    D --> E[Respond with placeholder or mock result]
    E --> F[Assert call pattern or replay logs]
````

---

## 🧪 Usage Example

```python
from motif_memory_stub import MotifMemoryStub

# Initialize with a bounded log size
stub = MotifMemoryStub(max_log_size=100)

# Basic usage
stub.access("ψ-null@Ξ")
stub.expect_retrieve("query:shadow", ["echo"])
print(stub.retrieve("query:shadow"))  # → ["echo"]

# Validate behavior
assert stub.assert_accessed("ψ-null@Ξ")
stub.reset()
```

---

## 🔁 Expectation Testing

```python
with stub.expect_calls({"access": 2, "retrieve": 1}):
    stub.access("a")
    stub.access("b")
    stub.retrieve("query")
```

---

## 🧬 Exported State Format

```python
_, logs = stub.export_state()
print(logs)
```

**Sample Output:**

```json
{
  "access_log": ["a", "b"],
  "retrieved_log": ["query"],
  "call_counts": {"access": 2, "retrieve": 1, "export_state": 1}
}
```

---

## ⚠️ Notes on Use

* **Not for production**: This is a simulation stub—do not use for real memory persistence.
* **Export format is unstable**: State serialization is intended only for test diagnostics.

---

## 📦 RFC Coverage

| RFC Reference | Purpose Description               |
| ------------- | --------------------------------- |
| RFC-0005 §3   | Simulated motif access signature  |
| RFC-0006 §2   | Memory query geometry placeholder |
| RFC-0007 §5   | Symbolic state export placeholder |

---

## 🔗 Compatibility

| Component            | Role                                  |
| -------------------- | ------------------------------------- |
| `SymbolicTaskEngine` | Memory substitute for logic chains    |
| `RecursiveAgentFT`   | Compatible for tick + motif flow      |
| `LogicalAgentAT`     | Supports validation + access tracking |

---

## 🪬 License

MIT • © 2025 Lina Noor / Noor Research Collective
