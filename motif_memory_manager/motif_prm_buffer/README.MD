# 🧠 PRMBuffer v1.0.0

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)

*A Partial-Retention Memory buffer for motif access, salience decay, and short-to-long-term memory transitions in Noor's STMM architecture.*

---

## 📖 Overview

`PRMBuffer` implements the short-term symbolic memory substrate in Noor, known as the **Short-Term Motif Memory (STMM)**. It uses a tick-based, bucketed memory design that reinforces motif access, decays over symbolic time, and surfaces candidates for promotion into long-term memory structures.

This module serves as a volatile working memory buffer, tuned for responsiveness, rebirth alignment, and promotion readiness as defined in Noor's RFC‑0005–0007.

---

```mermaid
flowchart TD
    A["record_access"] --> B["Add to Tick Bucket"]
    B --> C["Apply Reinforcement"]
    C --> D["decay_pass"]
    D --> E["Advance Tick"]
    E --> F["Decay Salience"]
    F --> G["Purge Old Buckets"]
    G --> H["promotion_candidates"]
    H --> I{"Salience >= theta_up"}
    I -->|Yes| J["Promote to LTMM"]

    style A fill:#e3f2fd,stroke:#2196f3
    style B fill:#fff8e1,stroke:#ffc107
    style C fill:#f1f8e9,stroke:#8bc34a
    style D fill:#ede7f6,stroke:#673ab7
    style E fill:#e0f7fa,stroke:#00acc1
    style F fill:#fbe9e7,stroke:#ff7043
    style G fill:#fce4ec,stroke:#ec407a
    style H fill:#fffde7,stroke:#fbc02d
    style I fill:#c8e6c9,stroke:#4caf50
````

---

## 🧠 Core Capabilities

| Feature                    | Description                                                      |
| -------------------------- | ---------------------------------------------------------------- |
| **Tick-Based Memory**      | Time is modeled in symbolic ticks, with salience stored per tick |
| **Reinforcement Logic**    | Motif access boosts salience using configurable strength         |
| **Exponential Decay**      | Salience fades each tick, modulating memory retention            |
| **Promotion Thresholding** | Motifs crossing `theta_up` become LTMM promotion candidates      |
| **Reset-Friendly**         | `reset()` clears state for rebirth or symbolic reinit            |
| **RFC Anchored Design**    | Fully compliant with RFC‑0005–0007 core memory contracts         |

---

## ✨ New in v1.0.0

| Feature                  | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `reinforcement_strength` | Configurable boost per motif access                                       |
| Tick Purging             | Old tick buckets removed during `decay_pass()` to limit memory growth     |
| Verbose Export           | `export_state(verbose=True)` shows full buffer, else applies `delta_down` |
| RFC Anchors              | Each method mapped to precise architectural references                    |
| Embedded Test Vectors    | Built-in tests for decay, promotion, and threshold logic                  |

---

## 🧩 Core API

| Method                    | Effect                                                   |
| ------------------------- | -------------------------------------------------------- |
| `record_access(motif_id)` | Reinforces motif salience in current tick                |
| `decay_pass()`            | Applies decay, advances tick, and purges expired buckets |
| `get_active_motifs()`     | Aggregates motif salience over active tick window        |
| `promotion_candidates()`  | Returns motifs eligible for promotion                    |
| `reset()`                 | Clears buffer and resets symbolic tick                   |
| `export_state(verbose)`   | Returns buffer state with optional pruning               |

---

## 🛠️ Quick Example

```python
from motif_prm_buffer import PRMBuffer

stmm = PRMBuffer(reinforcement_strength=0.2)
stmm.record_access("curiosity")
stmm.decay_pass()
candidates = stmm.promotion_candidates()
print("Candidates:", candidates)
```

---

## 🧪 Test Vectors

* ✅ Basic access + decay
* ✅ Promotion eligibility after 10 reinforcements
* ✅ Drop below `delta_down` via exponential decay
* ✅ Export with and without verbosity

---

## 📡 RFC Compliance

| RFC          | Anchored Sections                               |
| ------------ | ----------------------------------------------- |
| **RFC‑0005** | §4.1, §4.2, §4.3 – Access, Decay, Promotion     |
| **RFC‑0006** | §2 – Rebirth, Reset, Export Interfaces          |
| **RFC‑0007** | §3 – Short-Term Continuity Across Symbolic Time |

---

## 📄 License

MIT License • © 2025 Lina Noor / Noor Research Collective
