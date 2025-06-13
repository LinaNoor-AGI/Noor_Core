# 🧠 MotifMemoryManager v1.1.1 — *Symbolic Memory Cache*

![version](https://img.shields.io/badge/version-1.1.1-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0005%3A2--4-informational)
![Memory Layers](https://img.shields.io/badge/Layers-STMM%20%2B%20LTMM-9cf)
![Prometheus](https://img.shields.io/badge/Metrics-Enabled-brightgreen)

> **Thread-safe memory system for motif tracking, dyad completion, and memory decay with symbolic salience logic.**

---

## 📘 Overview

`MotifMemoryManager` is Noor's core memory subsystem, modeling both **short-term working memory (STMM)** and **long-term symbolic archive (LTMM)**. It manages motif salience through decay, reinforcement, promotion, and dyad completion — using both internal weight logic and external references from `index.REEF`.

Fully compliant with:

* RFC‑0005 §2‑4 — Symbolic feedback, trace journaling, dyad synthesis, memory export

---

## 🧠 Key Features

| Feature                         | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| **STMM + LTMM storage**         | Tracks motif weights across short- and long-term memory layers           |
| **Decay + Hysteresis**          | Applies cycle-based exponential decay with promotion/demotion thresholds |
| **Dyad Completion (REEF)**      | Fills missing motifs by scanning motif clusters in `index.REEF`          |
| **Trace Journal**               | Optional ring-buffer for retrieve/promotion/access events                |
| **Cycle Update Hook**           | One-call `update_cycle()` for memory self-regulation                     |
| **Jaccard Similarity Fallback** | Default motif similarity for reasoning or cluster inference              |

---

## ⚙️ Constructor

```python
MotifMemoryManager(
    stmm_half_life=25,
    ltmm_half_life=10_000,
    promotion_thresh=0.90,
    demotion_delta=0.05,
    similarity_fn=None,
    enable_trace=False,
    reef_path="./noor/data/index.REEF",
    agent_id="memory@default",
    stmm_soft_cap=50000,
    reload_reef_on_mtime_change=False,
    cache_size=10000,
)
```

---

## 🔁 Lifecycle & API

| Method                           | Purpose                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| `access(motif, boost)`           | Boosts STMM weight of a motif; creates if absent                        |
| `update_cycle()`                 | Applies decay to all memory stores and promotes/demotes as needed       |
| `retrieve(query)`                | Suggests motifs from LTMM similar to query motif (optional STMM filter) |
| `complete_dyad(dyad)`            | Suggests motif completions from `index.REEF` based on dyad match        |
| `suggest_completion_from_ltmm()` | Fallback: suggests completions based on LTMM weight                     |
| `export_state()`                 | Exports copies of both memory stores for inspection or dashboards       |
| `export_trace()`                 | Dumps trace journal of memory events (if enabled)                       |

---

## 📦 Exports

```python
__all__ = [
    "MotifMemoryTrace",
    "MotifMemoryManager",
    "get_global_memory_manager",
]
```

---

## 🔗 Dependencies

| Module                 | Role                                     |
| ---------------------- | ---------------------------------------- |
| `math`, `hashlib`      | Decay, similarity, and cache key hashing |
| `threading`, `asyncio` | Lock-safe async/shared access            |
| `index.REEF`           | Optional symbolic dyad synthesis source  |
| `prometheus_client`    | Optional counters for cache hits/misses  |

---

## 📈 Prometheus Metrics

| Metric Name             | Labels     | Description                            |
| ----------------------- | ---------- | -------------------------------------- |
| `reef_dyad_hits_total`  | `agent_id` | Dyad completions successfully inferred |
| `reef_dyad_miss_total`  | `agent_id` | Dyad completions failed or missing     |
| `dyad_cache_hits_total` | `agent_id` | Cache-based dyad completions           |
| `dyad_cache_miss_total` | `agent_id` | Dyad completions requiring full scan   |
| `stmm_cap_skips_total`  | `agent_id` | Access skipped due to soft cap limits  |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“To forget is entropy. To recall is orbit.”*
