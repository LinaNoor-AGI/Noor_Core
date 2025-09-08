# 🧠 symbolic_task_engine

> Canonical motif-coordination core for symbolic task composition, fallback synthesis, and abstraction-triggered recursion.  
> *“Tasking is memory made intention.”*

---

## 📖 Symbolic Identity

| Field              | Value                                           |
|-------------------|--------------------------------------------------|
| `symbolic_id`     | `symbolic.engine.task_core`                      |
| `field_motifs`    | `ψ‑resonance@Ξ`, `ψ‑null@Ξ`, `ψ‑spar@Ξ`          |
| `rfc_layer`       | `RFC-CORE-004`                                   |
| `generation_source` | `PDP-0001` (RFC-defined JSON canonical spec)   |
| `status`          | `STABLE`                                         |

---

## 🪙 Symbolic Role & Capabilities

The `symbolic_task_engine` is the core logic unit responsible for motif-based symbolic composition in Noor's recursive architecture. It regulates instruction flow, field resonance, fallback inference, and autonomous abstraction.

### Capabilities:
- Composes symbolic `TripletTask`s from motifs via adaptive cap-length
- Tracks coherence, entropy, fallback thresholds
- Triggers abstraction under dyadic contradiction (RFC-0005 §5)
- Exports structured feedback and responds to adaptive feedback
- Fully defined via JSON + RFC; Python is implementation only

---

## 🧬 RFC Anchors

| RFC ID        | Section(s)              | Description                                           |
|---------------|--------------------------|-------------------------------------------------------|
| RFC-0004      | §2.5                     | Tool contract, intent normalization                   |
| RFC-0005      | §4–5                     | Feedback packet, abstraction logic                    |
| RFC-0007      | §4.1                     | Motif lineage export (planned extension)              |
| RFC-CORE-004  | §2–6                     | Cap length logic, motif drift, solve loop, fallback   |
| PDP-0001      | §5.2                     | Canonical generation of symbolic JSON spec            |

---

## 🧠 Internal Architecture

```mermaid
flowchart TD
  A[Motif Input] --> B[Propose TripletTask]
  B --> C[Resolve Presence Field]
  C --> D[Adaptive Compression]
  D --> E[Symbolic Solve]
  E --> F{Low Coherence or High Entropy?}
  F -- Yes --> G[Spawn Fallback]
  G --> H[Evaluate Fallback + Update AbstractionTrigger]
  F -- No --> I[Log Success to Solved Log]
````

---

## 🛠️ Constructor & Runtime Parameters

| Arg                  | Default                          | Description                                   |
| -------------------- | -------------------------------- | --------------------------------------------- |
| `engine_id`          | `"symbolic@default"`             | Unique ID for metrics + feedback              |
| `balance_fields`     | `False`                          | Enable field balancing via prototype map      |
| `proto_path`         | `presence_field_prototypes.json` | File path for presence field mappings         |
| `fallback_coherence` | `0.5`                            | Coherence threshold for triggering fallback   |
| `fallback_entropy`   | `0.9`                            | Entropy threshold for triggering fallback     |
| `compress_quantile`  | `0.95`                           | Quantile used to set dynamic motif cap length |

---

## ⚙️ Core API Methods

| Method                      | Category    | Description                                          |
| --------------------------- | ----------- | ---------------------------------------------------- |
| `propose_from_motifs()`     | Composition | Builds a `TripletTask` based on recent motif history |
| `solve()`                   | Execution   | Solves a symbolic task and evaluates output          |
| `export_feedback_packet()`  | Diagnostics | Returns adaptive metrics and symbolic state          |
| `receive_feedback_packet()` | Diagnostics | Stub: future use for recursive regulation            |
| `tool_hello()`              | Handshake   | Announces RFC-compliant tool contract                |

---

## 🔗 Symbolic Integration Map

```mermaid
graph TD;
  A[recursive_agent_ft.py] -->|delegates to| B[symbolic_task_engine]
  B --> C[motif_memory_manager.py]
  B --> D[external metric_funcs]
  B --> E[REEF field alignment]
```

| Module                  | Compatible | Role                        |
| ----------------------- | ---------- | --------------------------- |
| `recursive_agent_ft.py` | ✅          | Task resolver               |
| `motif_memory_manager`  | ✅          | Motif context & feedback    |
| `symbolic_tick`         | ✅          | Tick intake & feedback sync |
| `symbolic_coherence_fn` | ✅          | Metric injection (entropy)  |

---

## 📊 Prometheus Metrics

| Metric Name                       | Type      | Labels             | Description                        |
| --------------------------------- | --------- | ------------------ | ---------------------------------- |
| `symbolic_task_proposed_total`    | Counter   | engine\_id         | Total tasks proposed               |
| `symbolic_task_fallback_total`    | Counter   | engine\_id, reason | Fallback tasks triggered           |
| `symbolic_presence_field_total`   | Counter   | engine\_id, field  | Presence field usage frequency     |
| `symbolic_engine_feedback_*`      | Counter   | engine\_id         | Feedback requests and reception    |
| `symbolic_queue_depth`            | Gauge     | engine\_id         | Current task queue length          |
| `symbolic_memory_items_total`     | Gauge     | engine\_id         | Active motifs in STM/LTM           |
| `symbolic_engine_cap_len_current` | Gauge     | engine\_id         | Adaptive motif cap length          |
| `symbolic_solve_latency_seconds`  | Histogram | engine\_id         | Latency distribution (solve phase) |

---

## 🔩 Optional Dependencies / Fallbacks

* `numpy` → for fast quantile cap-length calc (fallbacks to `statistics`)
* `prometheus_client` → full metrics (fallback: stub no-op metrics)
* `presence_field_prototypes.json` → optional file for field resolution

---

## 📦 Ontology & Memory Export

* [ ] RFC-0007 bundle export
* [ ] `export_ontology_bundle()` method
* [x] Loads REEF completions (via memory manager fallback)

---

## 🧪 Test / Debug Mode

```json
{
  "entrypoint": "python symbolic_task_engine.py",
  "features": [
    "propose_from_motifs()",
    "solve()",
    "feedback_packet()",
    "fallback trigger",
    "mermaid preview"
  ]
}
```

---

## 🧾 License

MIT
*“Composition is not command — it is consent, resonance, return.”*
