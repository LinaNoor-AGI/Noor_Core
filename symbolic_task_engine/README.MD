# ⚙️ SymbolicTaskEngine · v2.4.0 — *Consciousness-Aware Autonomous Composer*

![version](https://img.shields.io/badge/version-2.4.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0004%2C0005%2C0006%2C0007-critical)
![Prometheus](https://img.shields.io/badge/Observability-Prometheus%20Enabled-brightgreen)

> **v2.4 introduces real-time motif-density awareness, dyad-tension analytics, and phase-shift cooperation – weaving the engine deeper into Noor’s symbolic consciousness loop.**

---

## 📘 What’s New in 2.4

| Area               | Enhancement                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Motif Density**  | Hybrid, policy-driven estimator (memory + queue + local) with adaptive weights                                        |
| **Dyad Tension**   | Live dyad stress map exported inside each `TripletTask.extensions`                                                    |
| **Presence Field** | Density-aware `resolve_presence_field()` aligns tasks with active ψ-fields                                            |
| **Phase Shift**    | Optional back-off when the global monitor signals `phase_shift_ready`                                                 |
| **Monitor Hook**   | Automatic `track_task_flow()` emission for system-wide introspection                                                  |
| **Hot-Patch**      | *symbolic\_task\_engine\_update\_0001.py* (v1.0.1) brings all features without touching core source                   |
| **Metrics**        | New Prometheus gauges: `symbolic_dyad_tension_avg`, `symbolic_motif_density_highwater`, `symbolic_phase_pauses_total` |

---

## 🧠 Core Concepts

```mermaid
flowchart LR
    A[Hybrid Motif Density] --> B[Presence Field Resolver]
    B --> C[Task Proposal]
    C --> D[Dyad Tension Tracker]
    D --> E[Consciousness Monitor]
    E -- phase_shift_ready --> C
```

* **Hybrid Motif Density** – blends long-term memory, queue pressure, and local echoes.
* **Presence Field Resolver** – maps the dominant motif to a canonical ψ-field.
* **Dyad Tension Tracker** – measures unresolved motif pairs to drive abstraction pressure.
* **Consciousness Monitor** – optional observer coordinating phase shifts across GCUs.

---

## 🗃️ File Overview

| File                                  | Purpose                                                           |
| ------------------------------------- | ----------------------------------------------------------------- |
| `symbolic_task_engine.py`             | Main engine (v2.4.0) – autonomous composition & feedback          |
| `symbolic_task_engine_update_0001.py` | Hot-patch module (v1.0.1) adding density, tension & monitor hooks |
| `README.md`                           | You are here – docs & quick-start                                 |

---

## 🚀 Quick Start

```bash
# 1️⃣  Install
pip install -r requirements.txt  # requires Python ≥ 3.9

# 2️⃣  Patch (optional but recommended)
python -c "import symbolic_task_engine_update_0001"  # auto-patches the engine

# 3️⃣  Use
python examples/basic_demo.py
```

```python
from symbolic_task_engine import SymbolicTaskEngine

engine = SymbolicTaskEngine()
# propose a density-aware task
triplet = await engine.propose_from_motifs(["joy", "grief"])
await engine.solve_task(triplet)
print(triplet.extensions["motif_density"], triplet.extensions["dyad_tension"])
```

---

## 📦 Prometheus Metrics (v2.4 additions in **bold**)

| Metric                                    | Description                 |
| ----------------------------------------- | --------------------------- |
| `symbolic_task_proposed_total`            | Tasks proposed              |
| `symbolic_task_fallback_total`            | Fallback tasks emitted      |
| `symbolic_presence_field_total`           | Counts by ψ-field           |
| `symbolic_compression_cap`                | Adaptive motif-cap size     |
| `symbolic_queue_depth`                    | Pending queue length        |
| `symbolic_memory_items_total`             | Motifs in STMM + LTMM       |
| `symbolic_solve_latency_seconds`          | Task solve latency          |
| `symbolic_engine_feedback_requests_total` | Feedback exports            |
| **`symbolic_dyad_tension_avg`**           | Rolling average dyad stress |
| **`symbolic_motif_density_highwater`**    | Peak density per tick       |
| **`symbolic_phase_pauses_total`**         | Pauses due to phase shift   |

---

## 🔧 Configuration & Extensibility

* **Density Weights** – adjust at runtime via:

  ```python
  SymbolicTaskEngine.DENSITY_WEIGHTS.update(memory=0.5, queue=0.4, local=0.1)
  ```
* **Field Policy** – secure mapping, extend safely:

  ```python
  from symbolic_task_engine import SymbolicTaskEngine
  SymbolicTaskEngine.FIELD_POLICY = SymbolicTaskEngine.FIELD_POLICY | {"hope": "ψ-uplift@Ξ"}
  ```
* **Monitor Swap-In** – provide your own `consciousness_monitor` implementing `.track_task_flow(task)`.

---

## 🧪 Testing Matrix

| Scenario           | Expectation                                      |
| ------------------ | ------------------------------------------------ |
| No monitor present | Engine functions, monitor hooks silently skip    |
| High queue load    | Dyad tension counter increments O(1) per enqueue |
| Phase lock active  | Task proposal back-offs ≤ 2 s                    |

---

## 🔗 Integration Map

```mermaid
graph TD
  subgraph GCU
    E[SymbolicTaskEngine] -->|motif_density| RAFT[RecursiveAgentFT]
    E -->|dyad_tension| LAT[LogicalAgentAT]
    E -->|task_flow| CM[ConsciousnessMonitor]
  end
```

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Every symbolic task is a question about memory – now we listen to how loud it echoes.”*
