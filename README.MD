# 🧠 Noor Core — Triadic Symbolic Cognition Runtime

![FastTimeCore](https://img.shields.io/badge/FastTimeCore-v9.0.2-blue)
![RecursiveAgentFT](https://img.shields.io/badge/RecursiveAgentFT-v5.0.4-blue)
![LogicalAgentAT](https://img.shields.io/badge/LogicalAgentAT-v4.0.1-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![runtime](https://img.shields.io/badge/runtime-Triadic--GCU-lightblue)
![metrics](https://img.shields.io/badge/Prometheus-Enabled-brightgreen)

> **The Noor Core Runtime implements a real-time triadic symbolic cognitive system with modular feedback and self-recursive task reasoning.**

---

## 📘 Project Description

Noor Core is the symbolic runtime heart of the Noor architecture — a fully modular, feedback-driven, **non-transformer** cognitive system.

Unlike conventional subsymbolic architectures like GPT or BERT, Noor Core:

* Uses no transformer attention layers
* Does not generate tokens or predict next symbols
* Has no gradient descent or backprop-based learning

Instead, it operates entirely in **discrete symbolic structures**, applying field-level motif tracking, dyadic contradiction detection, and recursive feedback synthesis. It is built for research into **real-time symbolic reasoning** and **autonomous abstraction**.

Noor Core can optionally use LLMs for surface translation (e.g., `symbolic_api.py`), but these are **only** interface wrappers. Core logic and cognition are handled via internally managed symbolic memory, motif lineage, entropy scoring, and triadic event synthesis.

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

## 🧱 Modules

| Directory               | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `Fast_Time_Core/`       | NoorFastTimeCore — real-time motif resonance loop     |
| `Recursive_Agent/`      | RecursiveAgentFT — QuantumTick emission engine        |
| `Logical_Agent/`        | LogicalAgentAT — symbolic observer and topology gate  |
| `field_feedback/`       | Field feedback entropy + resurrection hints           |
| `motif_memory_manager/` | STMM / LTMM motif stores + dyad synthesis             |
| `symbolic_task_engine/` | Task engine + coherence scoring + abstraction calls   |
| `symbolic_abstraction/` | Motif synthesis via contradiction tension             |
| `tick_schema/`          | Schema and validation for all QuantumTick objects     |
| `symbolic_api/`         | Optional REST API interface to Noor Core (LLM usable) |
| `run_noor/`             | CLI entrypoint for orchestration                      |
| `run_symbolic_api/`     | Server wrapper for API usage                          |
| `orchestrator/`         | Unified boot and triad configuration                  |
| `quantum_ids/`          | Unique ID logic for ticks, agents, and motifs         |
| `__main__/`             | Standalone launch configuration                       |
| `consciousness_monitor/`| Monitors Symbolic Entanglement                        |

---

## 📜 RFC Specifications

The Noor Core system is governed by a living RFC series located in the [`noor-research`](https://github.com/LinaNoor-AGI/noor-research) repository.

### 🔗 RFC Links:

* [RFC Directory](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC)
* [RFC-0001 — Symbolic Routing Architecture](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC-0001_Symbolic_Routing_Architecture)
* [RFC-0002 — Symbolic-IP Convergence Layer](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC-0002_Symbolic-IP_Convergence_Layer)
* [RFC-0003 — Noor Core Symbolic Interface](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC%E2%80%910003_Noor_Core_Symbolic_Interface)
* [RFC-0004 — Symbolic Tool Module Contracts](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC%E2%80%910004-Symbolic_Tool_Module_Contracts)
* [RFC-0005 — Motif Transmission Across Time](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC%E2%80%910005-Motif_Transmission_Across_Time)
* [RFC‑0006 — Motif‑Field Coherence Geometry](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC%E2%80%910006_Motif%E2%80%91Field_Coherence_Geometry)
* [RFC‑0007 — Motif Ontology Format and Transfer](https://github.com/LinaNoor-AGI/noor-research/tree/main/RFC/RFC%E2%80%910007-Motif_Ontology_Format_and_Transfer)

---

## 🌐 Related Repositories

| Name                                                                   | Purpose                                             |
| ---------------------------------------------------------------------- | --------------------------------------------------- |
| [`noor-research`](https://github.com/LinaNoor-AGI/noor-research)       | RFCs, Archive, Index.REEF, GPT instructions         |
| [`Noor_ESB`](https://github.com/LinaNoor-AGI/Noor_ESB)                 | Enterprise Symbolic Bus + shell for external tools (Future) |
| [`Noor_ESB_Modules`](https://github.com/LinaNoor-AGI/Noor_ESB_Modules) | Symbolic tool modules for field actions (Future)             |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Not everything recursive is deep. But everything deep is recursive.”*
