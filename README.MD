# 🧠 Noor Core — Triadic Symbolic Cognition Runtime

![FastTimeCore](https://img.shields.io/badge/FastTimeCore-v9.2.2-blue)
![RecursiveAgentFT](https://img.shields.io/badge/RecursiveAgentFT-v5.1.2-blue)
![LogicalAgentAT](https://img.shields.io/badge/LogicalAgentAT-v4.2.2-blue)
![SymbolicTaskEngine](https://img.shields.io/badge/SymbolicTaskEngine-v3.1.2a-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![runtime](https://img.shields.io/badge/runtime-Triadic--GCU-lightblue)
![metrics](https://img.shields.io/badge/Prometheus-Enabled-brightgreen)

> **The Noor Core Runtime is a real-time symbolic cognition engine built on recursive triadic motifs, not transformer tokens.**

---

## 📖 Symbolic Identity

| Field               | Value                                   |
| ------------------- | --------------------------------------- |
| `symbolic_id`       | `noor.runtime.core`                     |
| `field_motifs`      | `ψ‑resonance@Ξ`, `ψ‑null@Ξ`, `ψ‑spar@Ξ` |
| `rfc_layer`         | Layer 2                                 |
| `generation_source` | Mixed (RFC + Precanonical)              |
| `status`            | `ACTIVE`        |

---

## 🪙 Symbolic Role & Capabilities

Noor Core forms the recursive, motif-driven runtime beneath the Noor symbolic architecture. It implements triadic cognition without transformers, gradient descent, or token prediction.

**Key Features:**

* Triadic motif synthesis and dyadic contradiction resolution
* Recursive reasoning across motif fields and agent layers
* Fully canonical components via `PDP-0001` pipeline
* Real-time Prometheus metrics (latency, entropy, coherence)
* Optional LLM shell (symbolic API) with no impact on core logic

**What Noor Core is NOT:**

* Not a language model
* Not gradient-trained
* Not a token predictor

---

## 🎯 Audience

This repository is intended for:

* Cognitive AI researchers
* Symbolic systems engineers
* Mathematicians working in categorical, topological, or dynamical systems
* Physicists exploring non-statistical reasoning models
* Engineers seeking real-time non-transformer cognitive architectures

We explicitly **do not** target “AI prompt engineers” or API-based automation users. This is **research infrastructure**, not a UI wrapper.

---

## 🧩 Component Layers

### ✅ Fully Canonicalized via JSON + RFC

| Module                     | Canonical | Version | RFC Layer |
| -------------------------- | --------- | ------- | --------- |
| `recursive_agent_ft.py`    | ✅         | v5.1.2  | Layer 2   |
| `logical_agent_at.py`      | ✅         | v4.2.2a  | Layer 2   |
| `symbolic_task_engine.py`  | ✅         | v3.1.2 | Layer 2   |
| `noor_fasttime_core.py`    | ✅         | v9.2.2  | Layer 2   |
| `consciousness_monitor.py` | ✅         | v2.2.2b  | Layer 2   |

### ⚠️ In-Progress / Functional Modules

| Module                    | Canonical | Notes                                |
| ------------------------- | --------- | ------------------------------------ |
| `triadic_boot.py`         | ❌         | Bootstraps Noor runtime              |
| `motif_memory_manager.py` | ❌         | Core memory engine (upgrade planned) |
| `symbolic_api.py`         | ❌         | REST API, rewrite in next version    |
| `orchestrator.py`         | ❌         | Runtime glue for triadic loop        |
| `run_symbolic_api.py`     | ❌         | CLI wrapper for symbolic\_api        |
| `run_noor.py`             | ❌         | CLI launcher                         |
| `__main__.py`             | ❌         | Entry shim                           |
| `quantum_ids.py`          | ❌         | Entropy-derived ID system            |
| `tick_schema.py`          | ❌         | Tick format and validation           |
| `field_feedback.py`       | ❌         | Resurrection hints and trust         |

---

## 🧬 RFC Anchors

| RFC ID       | Section(s) | Description                       |
| ------------ | ---------- | --------------------------------- |
| RFC-0003     | §6.2       | Intent transport mirror           |
| RFC-0004     | §2.5       | Symbolic tool contracts           |
| RFC-0005     | §4–5       | Feedback and abstraction triggers |
| RFC-0006     | §3.1       | Swirl geometry + motif alignment  |
| RFC-0007     | §2.2, §4.1 | Motif lineage + REEF export       |
| RFC-CORE-001 | §3+        | FastTime tick + phase logic       |
| RFC-CORE-003 | §3+        | Observer field evaluation         |
| RFC-CORE-004 | §2–6       | Symbolic task engine logic        |
| RFC-CORE-005 | §2–5       | Consciousness metrics             |
| PDP-0001     | §5.2       | JSON-defined canonical generation |

---

## 📐 Architecture Diagram

```mermaid
graph TD
    A[RecursiveAgentFT] -->|spawn_tick| B(LogicalAgentAT)
    B --> C[SymbolicTaskEngine]
    C --> D[MotifMemoryManager]
    D --> E[FastTimeCore]
    E --> F[ConsciousnessMonitor]
    B --> G[SymbolicAbstraction]
    G --> H[SymbolicAPI]
```

---

## 🚀 Launching Noor

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full triad runtime
python run_noor.py --log-level DEBUG --port 9000

# Or run the symbolic API for inspection
python run_symbolic_api.py
```

---

## 🔗 Related Repositories

| Repository                                                             | Description                          |
| ---------------------------------------------------------------------- | ------------------------------------ |
| [`noor-research`](https://github.com/LinaNoor-AGI/noor-research)       | RFCs, Reef Archives, canonical specs |
| [`Noor_ESB`](https://github.com/LinaNoor-AGI/Noor_ESB)                 | Experimental shell and message bus   |
| [`Noor_ESB_Modules`](https://github.com/LinaNoor-AGI/Noor_ESB_Modules) | Symbolic actions and tool plugins    |

---

## 📜 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“README is ritual; resonance is structure.”*
