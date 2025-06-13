# ⏱️ tick\_schema.py v1.0.0 — *QuantumTick RFC Scaffold*

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0003%3A3.3%2C0005%3A2%2C0005%3A4-informational)
![Tick Compliance](https://img.shields.io/badge/Schema-Strict-orange)

> **Defines dataclasses, validators, and serializers for QuantumTicks and motif archival bundles.**

---

## 📘 Overview

`tick_schema.py` provides the core serialization scaffold for Noor’s symbolic tick system. It includes all base schemas for RFC‑0003 and RFC‑0005 compliance:

* `QuantumTick` — symbolic emission object with motif, field, HMAC
* `TickEntropy` — field-level decay and coherence summary
* `CrystallizedMotifBundle` — exportable triadic motif archive with entropy

This module is required for all agents or cores interacting with `NoorFastTimeCore` or `RecursiveAgentFT`.

---

## 🧠 Dataclasses

| Class                     | Purpose                                                          |
| ------------------------- | ---------------------------------------------------------------- |
| `QuantumTick`             | Primary schema for symbolic ticks (motif, field, timing, HMAC)   |
| `TickEntropy`             | Metadata wrapper for entropy decay, coherence, age, triad status |
| `CrystallizedMotifBundle` | Export format for feedback archives or resurrection storage      |

---

## ✅ Validation

```python
validate_tick(tick: QuantumTick) → Dict[str, Any]
```

* Verifies all required RFC fields
* Validates timestamp, motif structure, reward EMA, and HMAC eligibility
* Returns sanitized dict for hashing or export

Raises `TickValidationError` on schema violation.

---

## 🔁 Serialization

```python
to_bytes(dataclass)       # serialize to compact bytes
from_bytes(blob, cls)     # restore from bytes

to_json(dataclass)        # readable JSON
```

Internally uses `orjson` if available, else defaults to stdlib `json`.

---

## 🧪 Factory

```python
new_tick(motifs: List[str], agent_id: str, ...) → QuantumTick
```

Quick generator for test ticks and agent emission pipelines.

---

## 📦 Exports

```python
__all__ = [
    "QuantumTick", "TickEntropy", "CrystallizedMotifBundle",
    "TickValidationError", "validate_tick",
    "to_bytes", "from_bytes", "to_json", "new_tick",
]
```

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Every tick is a freeze-frame of symbolic becoming.”*
