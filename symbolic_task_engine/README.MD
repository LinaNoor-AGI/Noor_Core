# ⚙️ SymbolicTaskEngine · v2.3.0 — *RFC-Wrapped Autonomous Composer*

![version](https://img.shields.io/badge/version-2.3.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0004%2C0005-informational)
![MVP Stage](https://img.shields.io/badge/Status-MVP%2011-lightblue)
![Prometheus](https://img.shields.io/badge/Metrics-Enabled-brightgreen)

> **Coordinates autonomous symbolic task synthesis, scoring, and feedback for recursive reasoning systems.**

---

## 📘 Overview

`symbolic_task_engine.py` defines Noor’s autonomous composition engine. It coordinates:

* Motif triplet proposals
* Task solving
* Evaluation (entropy, coherence, etc.)
* Feedback export (RFC‑0005 §4)
* Adaptive motif-cap compression
* Resurrection hint propagation

Fully compliant with:

* RFC‑0004 — Symbolic handshake via `tool_hello()`
* RFC‑0005 §4 — Feedback, trust vectors, ctx ratio, resurrection signal

---

## 🧠 Core Classes

| Class                | Role                                                             |
| -------------------- | ---------------------------------------------------------------- |
| `TripletTask`        | Symbolic instruction (motif\[] + instruction + expected\_output) |
| `Attempt`            | Attempted solution with metrics and timestamps                   |
| `SymbolicTaskEngine` | Singleton engine coordinating all symbolic tasks                 |

---

## 🔧 Key Features

| Feature                        | Description                                                             |
| ------------------------------ | ----------------------------------------------------------------------- |
| **Task Queue + TTL**           | Tracks, expires, and manages active symbolic tasks                      |
| **Coherence/Entropy scoring**  | Plug-in scoring system via metric decorators                            |
| **Fallback with motif recall** | Proposes symbolic fallback tasks based on coherence/entropy EMA         |
| **Field prototype bias**       | Auto-selects presence fields based on usage balance                     |
| **Feedback packet export**     | Returns RFC‑0005‑style snapshot of internal reasoning state             |
| **Autonomous abstraction**     | Integrates with `AbstractionTrigger` when dyadic contradictions persist |

---

## 🧪 Usage

```python
engine = SymbolicTaskEngine()
task = await engine.propose_from_motifs(["joy", "grief"])
await engine.solve_task(task)
await engine.flush_old_tasks()
```

---

## 📦 Prometheus Metrics

| Metric                                    | Description                      |
| ----------------------------------------- | -------------------------------- |
| `symbolic_task_proposed_total`            | Tasks proposed                   |
| `symbolic_task_fallback_total`            | Fallback tasks emitted           |
| `symbolic_presence_field_total`           | Counts by field label            |
| `symbolic_compression_cap`                | Adaptive motif-cap at inference  |
| `symbolic_queue_depth`                    | Active pending queue length      |
| `symbolic_memory_items_total`             | Count of motifs in STMM and LTMM |
| `symbolic_solve_latency_seconds`          | Time to solve tasks              |
| `symbolic_engine_feedback_requests_total` | Exported feedback packets        |
| `symbolic_engine_feedback_received_total` | Received feedback packets        |
| `symbolic_engine_cap_len_current`         | Active cap length at solve time  |
| `symbolic_autoloop_backoff_total`         | Backoffs due to recursive stall  |

---

## 🔗 Integration

| Consumer                | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `symbolic_api.py`       | Exposes engine via FastAPI                             |
| `field_feedback.py`     | Consumes export packets for resurrection signal, trust |
| `recursive_agent_ft.py` | May pull motif completions or propose bundles          |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Every symbolic task is a question about memory.”*
