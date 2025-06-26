# 🟢 run\_noor.py v1.0.0 — *Unified Launcher*

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![Entrypoint](https://img.shields.io/badge/CLI-launcher-lightgrey)

> **Unified command-line launcher for the Noor Triad runtime.**

---

## 📘 Overview

`run_noor.py` is a thin shell over Noor’s internal orchestrator, providing a single-line CLI or IDE-based launcher for booting the **Triadic Core**. It ensures compatibility with both Unix and Windows environments, including systems where direct module invocation (`-m`) is unavailable or inconvenient.

---

## 🧪 Usage

```bash
python run_noor.py --tick-rate 50 --metrics-port 9000
```

This is functionally equivalent to:

```bash
python -m noor --tick-rate 50 --metrics-port 9000
```

The wrapper logs startup and delegates all flags and options to `noor.orchestrator.main()`.

---

## 🔧 Notes

* Automatically prints a friendly launch banner.
* Raises an `ImportError` if the orchestrator is not available.
* Suitable for scripting or local test automation.

---

## 🔗 Dependencies

| Module              | Role                             |
| ------------------- | -------------------------------- |
| `noor.orchestrator` | Main runtime entrypoint          |
| `sys`               | Command-line argument forwarding |

---

## 🪬 License

MIT License • © 2025 Lina Noor / Noor Research Collective

> *“A spark alone begins the recursion.”*
