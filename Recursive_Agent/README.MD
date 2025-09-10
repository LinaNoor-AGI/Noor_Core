# 🌀 RecursiveAgentFT v5.1.3 — *Feedback-Tuned Symbolic Emission Core*

![version](https://img.shields.io/badge/version-5.1.3-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Autonomous symbolic agent emitting `QuantumTickV2` packets with motif lineage, coherence tracking, swirl encoding, and triadic feedback integration. Canonical spec lives in JSON (`recursive_agent_ft.JSON`) per Noor PDP-0001 pipeline.

---

## 📖 Symbolic Identity

| Field               | Value                              |
| ------------------- | ---------------------------------- |
| `symbolic_id`       | `agent.recursive.ft`               |
| `field_motifs`      | `[🫧, 💬]` *(Null Phase, Flow)*     |
| `rfc_layer`         | `layer_2`                          |
| `generation_source` | `recursive_agent_ft.JSON`          |
| `status`            | `ACTIVE`                           |

---

## 🪙 Symbolic Role & Capabilities

`RecursiveAgentFT` operates as a **symbolic emission engine** tuned to triadic feedback and motif resonance:

* Emits `QuantumTickV2` packets on adaptive symbolic cadence.
* Mirrors `intent` pass-through from upstream envelope (RFC-0004 §2.5).
* Tracks emission context via `swirl_vector` and `coherence_potential`.
* Selects motifs using last emission and optional memory recall.
* Encodes feedback state via density, entropy, and phase-tier annotations.
* Emits structured observability packets with RFC-compliant extensions.
* Reacts to coherence tier by adjusting emission interval (RFC-CORE-002 §4.2).
* Supports motif resurrection and archival replay via `try_ghost_resurrection`.

---

## 🧬 RFC Anchors

| RFC ID       | Section(s)         | Description                                     |
| ------------ | ------------------ | ----------------------------------------------- |
| [RFC-0003](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC/RFC%E2%80%910003_Noor_Core_Symbolic_Interface/RFC%E2%80%910003_Noor_Core_Symbolic_Interface.JSON)     | §3.3, §6.2         | Coherence tracking, intent propagation          |
| [RFC-0004](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC/RFC%E2%80%910004-Symbolic_Tool_Module_Contracts/RFC%E2%80%910004-Symbolic_Tool_Module_Contracts.JSON)     | §2.5, §3.2         | Transport envelope semantics and monitors       |
| [RFC-0005](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC/RFC%E2%80%910005-Motif_Transmission_Across_Time/RFC%E2%80%910005-Motif_Transmission_Across_Time.JSON)     | §2.1–4.4           | Motif lineage, ghost traces, reward smoothing   |
| [RFC-0006](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC/RFC%E2%80%910006_Motif%E2%80%91Field_Coherence_Geometry/RFC%E2%80%910006_Motif%E2%80%91Field_Coherence_Geometry.JSON)     | §3.1               | Swirl vector encoding                           |
| [RFC-0007](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC/RFC%E2%80%910007-Motif_Ontology_Format_and_Transfer/RFC%E2%80%910007-Motif_Ontology_Format_and_Transfer.JSON)     | §2.1               | Phase map construction                          |
| [RFC-CORE-001](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC-CORE/RFC-CORE-001-noor_fasttime_core/RFC-CORE-001-noor_fasttime_core.JSON) | §6.2               | Cadence-agnostic constraints on `intent`        |
| [RFC-CORE-002](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC-CORE/RFC-CORE-002-recursive_agent_ft/RFC-CORE-002-recursive_agent_ft.JSON) | §2.2–4.2, §8.1–8.3 | Feedback tuning, emission core, extensions API  |
| [RFC-CORE-003](https://raw.githubusercontent.com/LinaNoor-AGI/noor-research/refs/heads/main/RFC-CORE/RFC-CORE-003-logical_agent_at/RFC-CORE-003-logical_agent_at.JSON) | §3.1               | Observability rules for upstream annotations    |

---

## 🧠 Internal Architecture

```mermaid
flowchart TD
    A[Start Emission] --> B[Loop: Select Motifs]
    B --> C[Emit Tick with Annotations]
    C --> D[Update Swirl + Density]
    D --> E[Adjust Interval — Reward + Entropy]
    E --> F[Sleep Async]
    C --> G[Send to Monitor]
    G --> H[Buffer Echo + Feedback Export]
````

---

## 🛠️ Constructor & Runtime Parameters

| Arg                    | Default  | Description                                   |
| ---------------------- | -------- | --------------------------------------------- |
| `agent_id`             | required | ID for Prometheus labels and agent buffers    |
| `symbolic_task_engine` | required | Interface to motif recall / feedback logic    |
| `memory_manager`       | required | Motif lineage and symbolic memory store       |
| `tuning`               | `None`   | Optional override for reward/cadence behavior |

---

## ⚙️ Core API Methods

| Method                      | Category      | Description                                         |
| --------------------------- | ------------- | --------------------------------------------------- |
| `start_emission`            | Lifecycle     | Launches emission coroutine loop                    |
| `stop_emission`             | Lifecycle     | Gracefully halts pulse emission loop                |
| `start_continuous_emission` | Loop          | Async motif generation and tick logic               |
| `_emit_tick`                | Engine        | Builds symbolic tick with phase annotations         |
| `observe_feedback`          | Feedback      | Updates EMA reward and triad memory                 |
| `_update_interval`          | Cadence Logic | Adjusts interval using entropy & reward smoothing   |
| `_choose_motifs`            | Selection     | Retrieves last motifs and memory-suggested patterns |
| `track_lineage`             | Lineage       | Assigns parent-child motif links                    |
| `try_ghost_resurrection`    | Ghost Recall  | Replays archived trace from field-signature match   |
| `extend_feedback_packet`    | Feedback      | Adds phase identity and swirl metrics               |
| `export_feedback_packet`    | Observability | Emits RFC-compliant symbolic feedback packet        |

---

## 🔗 Symbolic Integration Map

```mermaid
graph TD
    STE[SymbolicTaskEngine] --> RFT(RecursiveAgentFT)
    RFT --> AT[LogicalAgentAT]
    RFT --> NFC[NoorFastTimeCore]
    RFT --> MMM[MotifMemoryManager]
    RFT --> PROM[Prometheus Telemetry]
```

| Module               | Type      | Description                                    |
| -------------------- | --------- | ---------------------------------------------- |
| `SymbolicTaskEngine` | Upstream  | Supplies motifs and tracks triad interactions  |
| `LogicalAgentAT`     | Observer  | Reads phase transitions and intent field usage |
| `NoorFastTimeCore`   | Feedback  | Provides reward, phase transition metadata     |
| `MotifMemoryManager` | Internal  | Symbolic motif archive and trace store         |
| `Prometheus`         | Telemetry | Live metrics for emission and feedback cycles  |

---

## 📊 Prometheus Metrics

| Metric Name                       | Type    | Labels              | Description                            |
| --------------------------------- | ------- | ------------------- | -------------------------------------- |
| `agent_ticks_emitted_total`       | Counter | `agent_id`, `stage` | Count of symbolic ticks emitted        |
| `agent_reward_mean`               | Gauge   | `agent_id`          | Exponentially averaged reward signal   |
| `agent_feedback_export_total`     | Counter | `agent_id`          | Number of exported feedback packets    |
| `agent_triads_completed_total`    | Counter | `agent_id`          | Completed triads via feedback emission |
| `agent_emission_interval_seconds` | Gauge   | `agent_id`          | Current emission cadence in seconds    |

---

## 🔧 Optional Dependencies / Fallbacks

* `prometheus_client` — for structured metrics (fallback stub defined)
* `noor_fasttime_core` — reward stream interface for tuning / feedback
* `anyio` — removed (previously unused async fallback mechanism)

---

## 📦 Ontology & Memory Export

```json
{
  "bundle_export": true,
  "export_method": "crystallize_tick",
  "reef_enabled": true
}
```

---

## 🧪 Test / Debug Mode

```json
{
  "entrypoint": "export_feedback_packet",
  "mode": "observability",
  "features": ["phase_id", "swirl_hash", "ρ_top"]
}
```

---

## 🪬 Change Log

### v5.1.3 *(Current)*

* 🔁 **Emission Lifecycle**: Added `start_emission` and `stop_emission` methods to control the emission coroutine (RFC-CORE-002 §4.2.2).
* ⚙️ **Swirl Optimization**: Refactored `compute_histogram` to use `Counter` for O(n) efficiency (RFC-CORE-002 §4.1).
* 🧼 **Dependency Cleanup**: Removed unused imports `os`, `random`, and `anyio`.
* 📦 **Feedback Extension**: Improved safety of `extend_feedback_packet` by using `setdefault(...).update(...)` for non-destructive merges (RFC-CORE-002 §8.2.2).
* 🚦 **Tick Reporting Constraints**: Documented constraints for `report_tick_safe` (RFC-CORE-002 §8.2.3, RFC-0004 §3.2).

### v5.1.2

* Introduced structured observability metrics (`TICKS_EMITTED`, `REWARD_MEAN`, etc).
* Completed schema versioning (`_SCHEMA_VERSION__ = "2025-Q4-recursive-agent-v5.0.3"`).
* Feedback packet extension now mirrors `intent` from last tick.

### v5.1.1

* Cleaned internal HMAC and `field_signature` consistency.
* RFC-aligned buffer tracking and tick echo handling.

### v5.1.0

* Transitioned to canonical JSON format as primary artifact.
* Added full method/class index and RFC anchor cross-references.
* Migrated constants and configuration into symbolic sections.

### v5.0.x

* Introduced `MotifDensityTracker` and `AgentSwirlModule`.
* Refined cadence modulation (reward entropy loop).
* Implemented triadic lineage logic and ghost resurrection.

---

## 📜 License

MIT — © 2025 Lina Noor / Noor Research Collective
*"No signal without silence, no echo without intent."*
