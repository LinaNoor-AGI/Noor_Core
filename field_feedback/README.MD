# 🛰️ field\_feedback.py v1.0.0 — *Field Resonance Parser*

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![RFCs](https://img.shields.io/badge/RFCs-0003%3A4%2C0005%3A4-informational)
![Resurrection Hints](https://img.shields.io/badge/Entropy--Aware-yes-purple)
![Prometheus](https://img.shields.io/badge/Metrics-Upstream%20Ready-yellowgreen)

> **Extracts context feedback, computes trust vectors, and provides resurrection summaries for field feedback pipelines.**

---

## 📘 Overview

`field_feedback.py` is a lightweight symbolic helper module used by Noor’s feedback loop to:

* parse `ctx_ratio` and detect malformed or invalid feedback
* calculate trust from motif signature vectors
* attach entropy metadata and flag motifs for resurrection or decay

It does **not** handle inference, prediction, or state transitions — it provides **RFC-compliant summaries** for use by agents, observers, or archival tools.

Fully compliant with:

* RFC‑0003 §4 — Feedback validation and context parsing
* RFC‑0005 §4 — Resurrection flagging, entropy summarization, feedback packet encoding

---

## 🧠 Core Data Types

```python
@dataclass
class FieldFeedback:
    tick_id: str
    ctx_feedback: CtxFeedback
    trust_profiles: List[TrustProfile]
    entropy_summary: Dict[str, float]
    extensions: Dict[str, Any]
```

---

## ⚙️ Key Functions

| Function                    | Description                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| `parse_ctx_ratio()`         | Validates `ctx_ratio` (0–1 clamp, fallback = 0.5)                          |
| `compute_field_trust()`     | Heuristic trust score from tick vector and signature depth                 |
| `summarize_entropy()`       | Extracts decay, age, triad completion, coherence from `CrystallizedBundle` |
| `resurrection_hint()`       | Returns "resurrected", "faded", or `None` based on entropy state           |
| `make_field_feedback()`     | One-call generation of a valid `FieldFeedback` structure                   |
| `to_json()` / `from_json()` | Serialize/parse feedback structures                                        |

---

## 🔁 Sample Use

```python
from tick_schema import new_tick
from field_feedback import make_field_feedback, to_json

qt = new_tick(["bind@Ξ"], agent_id="agent@demo")
feedback = make_field_feedback(qt)
print(to_json(feedback))
```

---

## 📦 Exported Symbols

```python
__all__ = (
    "CtxFeedback",
    "TrustProfile",
    "FieldFeedback",
    "parse_ctx_ratio",
    "compute_field_trust",
    "summarize_entropy",
    "resurrection_hint",
    "make_field_feedback",
    "to_json",
    "from_json",
)
```

---

## 🔗 Dependencies

| Module         | Role                               |
| -------------- | ---------------------------------- |
| `tick_schema`  | Provides tick + entropy structures |
| `dataclasses`  | Internal type definitions          |
| `math`, `json` | Normalization and serialization    |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“Feedback is not signal. It is the field’s reflection of memory.”*
