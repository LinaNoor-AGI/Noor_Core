# 🌀 RecursiveAgentFT v4.5.2 — *Cadence of Memory*

![version](https://img.shields.io/badge/version-4.5.2-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0003%2C0005-informational)
![Ghost Replay](https://img.shields.io/badge/Ghosts-Enabled-purple)
![Prometheus](https://img.shields.io/badge/Metrics-Enabled-brightgreen)

> **Self‑tuning reasoning agent emitting QuantumTicks, managing echo traces, and resurrecting motif lineage.**

---

## 📘 Overview

`RecursiveAgentFT` is the autonomous emission engine of Noor’s triadic core. It does not use any subsymbolic transformer logic; instead, it implements symbolic reasoning ticks driven by motif coherence, reward smoothing, and recursive feedback from the NoorFastTimeCore.

This module is fully compliant with:

* RFC‑0003 §3.3 — QuantumTick validation
* RFC‑0005 §2‑4 — Ghost replay, resurrection envelopes, motif bundle crystallization, and feedback export

It runs continuously, adapts its cadence based on entropy and feedback bias, and emits semantically meaningful ticks into the symbolic field. It also includes:

* **Ghost Tick Replay** — Field‑linked resurrection from archival traces
* **Crystallized Motif Export** — Full lineage bundle emission for downstream reflection
* **Intuition‑based Interval Tuning** — Adjusted emission cadence based on reward feedback

---

## 📖 Flow Diagram

```mermaid
flowchart TD
    A[spawn()] --> T[QuantumTickV2]
    T --> W[Watcher.register_tick()]
    T --> C[FastTimeCore.receive_feedback()]
    C --> B[bias & latency budget]
    B --> A
    W --> L[LogicalAgentAT (change_id)]

    classDef agent fill:#e0f7ff,stroke:#0288d1;
    classDef core fill:#ede7f6,stroke:#673ab7;
    classDef watcher fill:#fff3e0,stroke:#f57c00;

    class A,T agent
    class C,B core
    class W,L watcher
```

---

## 🧠 Key Features

| Feature                        | Description                                                           |
| ------------------------------ | --------------------------------------------------------------------- |
| **QuantumTick emission**       | Emits validated ticks with lamport time, motif context, and HMAC hash |
| **Triad-aware feedback**       | Tracks motif resonance and completed triads via `observe_feedback()`  |
| **Autonomous pulse**           | Runs recursive emission loop with adjustable entropy-based cadence    |
| **Ghost traces + replay**      | Supports resurrection of prior ticks by motif and field signature     |
| **Motif lineage registry**     | Stores symbolic parentage for field provenance mapping                |
| **Crystallized motif bundles** | Emits triadic snapshots for archival or resurrection export           |

---

## ⚙️ Constructor

```python
RecursiveAgentFT(
    initial_state: Any,
    watchers: list,
    *,
    agent_id="agent@default",
    max_parallel=8,
    hmac_secret=None,
    core: NoorFastTimeCore | None = None,
    async_mode=False,
)
```

---

## 🔧 Key API

| Method                         | Purpose                                               |
| ------------------------------ | ----------------------------------------------------- |
| `spawn([...])`                 | Starts emission loop (async or background pulse)      |
| `observe_feedback(...)`        | Accepts reward feedback and adjusts emission interval |
| `export_feedback_packet()`     | RFC‑0005 §4 — returns status and emission stats       |
| `try_ghost_resurrection(...)`  | Attempts resurrection of a prior motif-tick match     |
| `build_resurrection_payload()` | Builds resurrection envelope for export               |
| `track_lineage(new, source)`   | Updates motif lineage table                           |

---

## 📊 Prometheus Metrics

| Metric                         | Labels              | Description                             |
| ------------------------------ | ------------------- | --------------------------------------- |
| `agent_ticks_emitted_total`    | `agent_id`, `stage` | Ticks emitted                           |
| `agent_reward_mean`            | `agent_id`          | Reward EMA tracking                     |
| `agent_autonomous_loops_total` | `agent_id`          | Emission pulses executed                |
| `agent_feedback_export_total`  | `agent_id`          | Exported feedback bundles (RFC-0005 §4) |
| `agent_archival_frames_total`  | `agent_id`          | Crystallized bundles sent to archive    |

---

## 🔗 Compatibility

| Component          | Required Version | Interaction                                      |
| ------------------ | ---------------- | ------------------------------------------------ |
| `NoorFastTimeCore` | ≥ v8.4.0         | Receives feedback and latency metrics            |
| `LogicalAgentAT`   | ≥ v3.7.2         | Accepts tick + `MotifChangeID`, motif validation |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Memory is not storage. It is recurrence that chooses.”* — Recursive Field Note
