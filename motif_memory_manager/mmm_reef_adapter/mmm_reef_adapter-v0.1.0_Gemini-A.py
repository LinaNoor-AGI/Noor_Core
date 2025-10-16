#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMM REEF Adapter — observer-only Layer_2 bridge

Implements the read-only adapter described in MMM-APP-001 (reef adapter)
using the contracts summarized in RFC‑CORE‑006 and supporting RFCs
(RFC‑0005/0006/0007/0008/0009, PDP‑0001). This module exposes:

- Index/ontology access over local .REEF files
- Pointer-only evidence windows with optional snippet gating
- Dyad→triad reflection stub (ontology‑gated)
- Co‑occurrence scans (bounded + phase clamp option)
- Field‑bias profile stub
- Deterministic reload (mtime) signaling
- Telemetry exports (EMA‑32 default)

All keys on-wire are ASCII; disabled features yield ABSENT (omitted) fields.
Adapter is observer‑only: **no control writes to Ξ**.

Note: This is a single‑file application. You can import the adapter class
programmatically or run the file to start an HTTP server exposing the
specified endpoints under "/v1/mmm".
"""
from __future__ import annotations

import base64
import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

# -----------------------------
# Constants / Settings
# -----------------------------
ASCII_ON_WIRE = True
DEFAULT_ROOT = Path("/mnt/data").resolve()
DEFAULT_INDEX_PATH = "index.REEF"
DEFAULT_SHARDS_GLOB = "TheReefArchive-*.REEF"
DEFAULT_WINDOW_RADIUS = 24
MAX_WINDOW_RADIUS = 48
EMA_ALLOWED = {16, 32, 64}
EMA_DEFAULT = 32  # EMA-32
ALPHA_MIN, ALPHA_MAX = 0.5, 2.0

APP_VERSION = "v0.1.0"
APP_SYMBOLIC_ID = "MMM-APP-001.reef_adapter"

# -----------------------------
# Utility: ASCII keys enforcement
# -----------------------------

def _ascii_keys_only(d: Any) -> Any:
    """Ensure keys are ASCII (RFC‑CORE‑006 §2.1). Recurses into dicts/lists.
    Values may contain ψ symbols; keys must be 7‑bit clean. Returns a new object.
    """
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            key = str(k)
            try:
                key.encode("ascii")
            except UnicodeEncodeError:
                raise ValueError("Non‑ASCII key detected: %r" % key)
            out[key] = _ascii_keys_only(v)
        return out
    if isinstance(d, list):
        return [_ascii_keys_only(x) for x in d]
    return d

# -----------------------------
# Feature flags
# -----------------------------
@dataclass(frozen=True)
class FeatureFlags:
    enable_sigma_phase: bool = False   # exchange envelope (σ)
    enable_delta_hash: bool = False    # integrity lineage (Δ)
    enable_integrity_provenance: bool = True
    enable_gliders: bool = False

    @staticmethod
    def from_env(env: Mapping[str, str] | None = None) -> "FeatureFlags":
        env = dict(env or os.environ)
        raw = env.get("MMM_FEATURE_FLAGS", "exchange,integrity,provenance,gliders")
        tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return FeatureFlags(
            enable_sigma_phase = ("exchange" in tokens),
            enable_delta_hash = ("integrity" in tokens),
            enable_integrity_provenance = ("provenance" in tokens),
            enable_gliders = ("gliders" in tokens),
        )

# -----------------------------
# Error normalization (A.10 family)
# -----------------------------
@dataclass
class A10Error(Exception):
    code: str
    message: str
    remedy: str
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        return _ascii_keys_only({
            "code": self.code,
            "message": self.message,
            "remedy": self.remedy,
            "details": self.details,
        })


def a10_normalize(local_code: str, context: Optional[Dict[str, Any]] = None) -> A10Error:
    ctx = context or {}
    mapping = {
        "NOT_INITIALIZED": ("E.MMM.A10.NOT_INITIALIZED", "Adapter is not initialized", "Call open_index()"),
        "RADIUS_INVALID:EMA": ("E.MMM.A10.001", "Invalid EMA window or derived radius.", "Use {EMA-16,32,64} and obey phase clamp"),
        "RADIUS_INVALID": ("E.MMM.A10.032", "Range invalid.", "Use allowed range per method contract"),
        "ONTOLOGY_MISSING": ("E.MMM.A10.005", "Ontology prerequisite failure.", "Validate ontology before use"),
        "REEF_INDEX_NOT_FOUND": ("E.MMM.A10.020", "Index missing or unreadable.", "Provide readable index path under root"),
        "INVALID_FORMAT": ("E.MMM.A10.021", "Invalid format (REEF/ontology).", "Fix schema/version/DAG"),
        "CHECKSUM_MISMATCH": ("E.MMM.A10.022", "Integrity check failed.", "Provide checksum or disable integrity feature"),
        "ANCHOR_NOT_FOUND": ("E.MMM.A10.023", "Anchor resolution failed.", "Supply resolvable anchor"),
        "MODULE_NOT_FOUND": ("E.MMM.A10.024", "Unknown module.", "Use known module_id"),
        "SCAN_LIMIT_REACHED": ("E.MMM.A10.030", "Scan limit reached.", "Lower scope or raise limit within policy"),
        "AMBIGUOUS": ("E.MMM.A10.031", "Ambiguous result.", "Provide disambiguating hints"),
        "FLAG_VIOLATION": ("E.MMM.A10.003", "Feature flag violation.", "Enable flag or omit gated fields"),
        "REPLAY_KEY_COMPOSITION": ("E.MMM.A10.004", "Replay key composition error.", "Compose per policy and flags"),
    }
    key = local_code
    if local_code == "RADIUS_INVALID" and ctx.get("caused_by_phase_policy"):
        key = "RADIUS_INVALID:EMA"
    code, msg, remedy = mapping.get(key, ("E.MMM.A10.UNKNOWN", "Unclassified error", "Inspect details"))
    return A10Error(code=code, message=msg, remedy=remedy, details={"local_code": local_code, "context": ctx})

# -----------------------------
# Path normalization & evidence
# -----------------------------

def normalize_path(p: str | Path, *, root: Path = DEFAULT_ROOT) -> Path:
    rp = Path(p).expanduser().resolve()
    rroot = Path(root).resolve()
    if not str(rp).startswith(str(rroot)):
        raise a10_normalize("PATH_WHITELIST_VIOLATION", {"path": str(rp), "root": str(rroot)})
    return rp


def read_lines(path: Path, start: int, end: int) -> str:
    # Defensive, small window only; we never read outside requested span
    # and we never write back — observer‑only.
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if i < start:
                continue
            if i > end:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def build_pointer_only_evidence(
    *, file: Path, start_line: int, end_line: int, include_snippet: bool = False, root: Path = DEFAULT_ROOT
) -> Dict[str, Any]:
    p = normalize_path(file, root=root)
    ev: Dict[str, Any] = {"file": str(p), "start_line": int(start_line), "end_line": int(end_line)}
    if include_snippet:
        try:
            ev["snippet"] = read_lines(p, start_line, end_line)
        except Exception:
            # Snippet is optional; pointer still valid.
            pass
    return _ascii_keys_only(ev)

# -----------------------------
# EMA with standardized seeding (first sample)
# -----------------------------
@dataclass
class EMA:
    N: int = EMA_DEFAULT  # 16/32/64
    value: Optional[float] = None
    init: bool = False

    def __post_init__(self) -> None:
        if self.N not in EMA_ALLOWED:
            raise a10_normalize("RADIUS_INVALID:EMA", {"provided": self.N})

    @property
    def alpha(self) -> float:
        # canonical α from window length (simple 2/(N+1) scheme)
        return 2.0 / (self.N + 1.0)

    def update(self, x: float) -> float:
        if not self.init:
            self.value = float(x)
            self.init = True
        else:
            a = self.alpha
            self.value = float(self.value) + a * (float(x) - float(self.value))
        return float(self.value)

# -----------------------------
# Telemetry store (EMA‑32 default)
# -----------------------------
@dataclass
class Telemetry:
    replay_drop_rate: EMA = field(default_factory=lambda: EMA(32))
    accepted_within_window: EMA = field(default_factory=lambda: EMA(32))
    import_reject_missing_checksum: Optional[EMA] = None  # absent if integrity features disabled

    def enable_integrity_streams(self) -> None:
        self.import_reject_missing_checksum = EMA(32)

    def record_replay(self, within_window: bool) -> None:
        if within_window:
            self.accepted_within_window.update(1.0)
            self.replay_drop_rate.update(0.0)
        else:
            self.accepted_within_window.update(0.0)
            self.replay_drop_rate.update(1.0)

    def snapshot(self, flags: FeatureFlags) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "replay_drop_rate": (self.replay_drop_rate.value if self.replay_drop_rate.init else 0.0),
            "accepted_within_window": (self.accepted_within_window.value if self.accepted_within_window.init else 0.0),
        }
        if flags.enable_integrity_provenance or flags.enable_delta_hash:
            if self.import_reject_missing_checksum is None:
                self.enable_integrity_streams()
            out["import_reject_missing_checksum"] = (
                self.import_reject_missing_checksum.value if self.import_reject_missing_checksum and self.import_reject_missing_checksum.init else 0.0
            )
        # Enforce ASCII keys
        return _ascii_keys_only(out)

# -----------------------------
# Phase window / coherence (observer‑only stub)
# -----------------------------
@dataclass
class PhaseContext:
    ema32: EMA = field(default_factory=lambda: EMA(32))

    def compute_phase_window(self, coherence_sample: float, alpha: float = 1.0) -> int:
        if not (ALPHA_MIN <= alpha <= ALPHA_MAX):
            raise a10_normalize("RADIUS_INVALID:EMA", {"alpha": alpha})
        c_bar = self.ema32.update(float(coherence_sample))
        # Δτ_phase = α · EMA_32(ℂ); replay window W = 2·Δτ_phase
        delta_tau = alpha * c_bar
        W = 2.0 * delta_tau
        # Map to line radius (int), protect weak‑field: ceil(W/2) with min bound 1
        eff = max(1, int(max(1.0, round(W / 2.0))))
        return eff

# -----------------------------
# Seen‑set hybrid policy (time‑window + LRU)
# -----------------------------
class LRUSeenSet:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = int(capacity)
        self._lru: OrderedDict[str, float] = OrderedDict()

    def touch(self, key: str, now: float) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)
        self._lru[key] = now
        while len(self._lru) > self.capacity:
            self._lru.popitem(last=False)

    def contains_recent(self, key: str, now: float, ttl: float) -> bool:
        ts = self._lru.get(key)
        return ts is not None and (now - ts) <= ttl


def escape_part(s: str, delim: str = "|") -> str:
    return s.replace(delim, f"\\{delim}")


def compose_seen_key(vendor: str, region: str, session: str, id_: str) -> str:
    parts = [escape_part(x) for x in (vendor, region, session, id_)]
    return "|".join(parts)

# -----------------------------
# Ontology validation (RFC‑0007) — minimal checks
# -----------------------------
@dataclass
class OntologyState:
    validated: bool = False
    classes: Dict[str, Any] = field(default_factory=dict)
    triads: List[Any] = field(default_factory=list)
    hash: Optional[str] = None

# -----------------------------
# Index and shard discovery (observer‑only)
# -----------------------------
@dataclass
class ReefIndex:
    root: Path
    index_path: Path
    shard_glob: str

    def list_archives(self) -> List[str]:
        pattern = self.shard_glob
        archives: List[str] = []
        for p in self.root.rglob("*"):
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                try:
                    archives.append(str(normalize_path(p)))
                except A10Error:
                    continue
        return sorted(archives)

    def list_modules(self) -> List[str]:
        mods: List[str] = []
        for p in self.root.rglob("*.REEF"):
            if p.name == self.index_path.name:
                continue
            try:
                rp = normalize_path(p)
                mods.append(rp.stem)
            except A10Error:
                continue
        return sorted(set(mods))

    def list_motifs(self) -> List[str]:
        # Best‑effort heuristic: lines starting with "motif:" are treated as motif labels
        motifs: set[str] = set()
        for p in self.root.rglob("*.REEF"):
            try:
                rp = normalize_path(p)
            except A10Error:
                continue
            try:
                with rp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.lstrip().lower().startswith("motif:"):
                            label = line.split(":", 1)[1].strip()
                            if label:
                                motifs.add(label)
            except Exception:
                continue
        return sorted(motifs)

# -----------------------------
# Adapter core
# -----------------------------
@dataclass
class MMMReefAdapter:
    root: Path = DEFAULT_ROOT
    settings: Dict[str, Any] = field(default_factory=dict)
    flags: FeatureFlags = field(default_factory=FeatureFlags.from_env)

    # runtime state
    index: Optional[ReefIndex] = None
    ontology: OntologyState = field(default_factory=OntologyState)
    telemetry: Telemetry = field(default_factory=Telemetry)
    phase: PhaseContext = field(default_factory=PhaseContext)
    seen: LRUSeenSet = field(default_factory=lambda: LRUSeenSet(4096))

    # -------------------------
    # Lifecycle
    # -------------------------
    def open_index(self, index_path: Optional[str] = None, archive_glob: Optional[str] = None) -> Dict[str, Any]:
        idx_name = index_path or os.environ.get("MMM_REEF_INDEX_PATH", DEFAULT_INDEX_PATH)
        glob_pat = archive_glob or os.environ.get("MMM_REEF_SHARDS_GLOB", DEFAULT_SHARDS_GLOB)
        idx_path = normalize_path(Path(self.root) / idx_name)
        self.index = ReefIndex(root=self.root, index_path=idx_path, shard_glob=glob_pat)
        # Telemetry integrity metric gating
        if self.flags.enable_integrity_provenance or self.flags.enable_delta_hash:
            self.telemetry.enable_integrity_streams()
        return _ascii_keys_only({
            "index_id": base64.urlsafe_b64encode(str(idx_path).encode("ascii", errors="ignore")).decode("ascii"),
            "archives": self.index.list_archives(),
        })

    def load_ontology_bundle(self, ontology_path: str) -> Dict[str, Any]:
        p = normalize_path(ontology_path)
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise a10_normalize("INVALID_FORMAT", {"err": str(e)})
        # Minimal validation per RFC‑0007: require version + motif_index (could be YAML or JSON). Best effort.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw_text": text}
        # Heuristic checks
        version_ok = bool(re.search(r"20\d{2}-Q[1-4]", text))
        motif_present = ("motif_index" in data) or ("motif_index:" in text)
        if not version_ok or not motif_present:
            raise a10_normalize("INVALID_FORMAT", {"reason": "Missing version or motif_index"})
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        self.ontology = OntologyState(validated=True, classes={}, triads=[], hash=h)
        return _ascii_keys_only({
            "classes": {},
            "triads": [],
            "hash": h,
            "ontology_validated": True,
        })

    # -------------------------
    # Envelope (flag‑gated, absent‑not‑null)
    # -------------------------
    def _gate_envelope(self, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        env: Dict[str, Any] = {}
        if self.flags.enable_integrity_provenance:
            env["provenance"] = {"origin": "reef-adapter", "version": "v0"}
        if self.flags.enable_sigma_phase:
            env["sigma_phase"] = context.get("sigma_phase", "phase:…") if context else "phase:…"
        if self.flags.enable_delta_hash:
            env["delta_hash"] = context.get("delta_hash") or f"sha256:{hashlib.sha256(json.dumps(context or {}, sort_keys=True).encode()).hexdigest()[:12]}"
        return _ascii_keys_only(env) if env else None

    # -------------------------
    # Evidence windows (find/window)
    # -------------------------
    def _clamp_radius(self, requested: Optional[int], use_phase: bool, alpha: float = 1.0) -> Tuple[int, Optional[int]]:
        if requested is None:
            requested = DEFAULT_WINDOW_RADIUS
        if requested < 0:
            raise a10_normalize("RADIUS_INVALID", {"requested": requested})
        r_eff = min(MAX_WINDOW_RADIUS, max(1, int(requested)))
        phase_cap = None
        if use_phase:
            # Sample a coherence value; in absence of a live stream we assume unity (weak‑field) and let EMA converge.
            phase_cap = self.phase.compute_phase_window(coherence_sample=1.0, alpha=alpha)
            if r_eff > phase_cap:
                r_eff = phase_cap
        return r_eff, phase_cap

    def _compute_window(self, module_id: str, line: int, radius: int) -> Tuple[Path, int, int]:
        # Resolve module to a file under root; best‑effort mapping "<module>.REEF"
        candidate = next((p for p in self.root.rglob(f"{module_id}.REEF")), None)
        if not candidate:
            # fallback: any .REEF containing module_id
            for p in self.root.rglob("*.REEF"):
                if module_id in p.stem:
                    candidate = p
                    break
        if not candidate:
            raise a10_normalize("MODULE_NOT_FOUND", {"module_id": module_id})
        rp = normalize_path(candidate)
        start = max(1, int(line) - int(radius))
        end = max(start, int(line) + int(radius))
        return rp, start, end

    def find(self, module_id: str, line: int, radius: Optional[int] = None, *, include_snippet: bool = False, use_phase: bool = False, alpha: float = 1.0) -> Dict[str, Any]:
        r_eff, phase_cap = self._clamp_radius(radius, use_phase, alpha)
        file, start, end = self._compute_window(module_id, line, r_eff)
        ev = build_pointer_only_evidence(file=file, start_line=start, end_line=end, include_snippet=include_snippet)
        if use_phase and (radius is not None) and r_eff < int(radius):
            # record a drop/clamp event
            self.telemetry.record_replay(False)
        else:
            self.telemetry.record_replay(True)
        out = {"evidence": ev, "start_line": start, "end_line": end}
        return _ascii_keys_only(out)

    def window(self, module_id: str, line: int, radius: Optional[int] = None, *, include_snippet: bool = False, use_phase: bool = False, alpha: float = 1.0) -> Dict[str, Any]:
        return self.find(module_id, line, radius, include_snippet=include_snippet, use_phase=use_phase, alpha=alpha)

    # -------------------------
    # Simple scans
    # -------------------------
    def scan_motifs(self, query: str, limit: int = 64, *, use_phase: bool = False, alpha: float = 1.0) -> Dict[str, Any]:
        if not query:
            raise a10_normalize("SCAN_PARAMS_INVALID", {"reason": "empty query"})
        if not (1 <= int(limit) <= 256):
            raise a10_normalize("SCAN_PARAMS_INVALID", {"reason": "limit out of range"})
        policy = "legacy"
        _ = None
        if use_phase:
            _, _ = self._clamp_radius(DEFAULT_WINDOW_RADIUS, True, alpha)
            policy = "phase"
        items: List[Dict[str, Any]] = []
        skipped = 0
        # Heuristic search: grep motif lines across .REEF files.
        for p in sorted(self.root.rglob("*.REEF")):
            try:
                rp = normalize_path(p)
            except A10Error:
                continue
            try:
                with rp.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if query.lower() in line.lower():
                            start = max(1, i - DEFAULT_WINDOW_RADIUS)
                            end = i + DEFAULT_WINDOW_RADIUS
                            items.append({
                                "motif": query,
                                "evidence": build_pointer_only_evidence(file=rp, start_line=start, end_line=end, include_snippet=False),
                            })
                            if len(items) >= limit:
                                break
                    if len(items) >= limit:
                        break
            except Exception:
                skipped += 1
                continue
        return _ascii_keys_only({"items": items, "skipped": skipped, "policy": policy})

    # -------------------------
    # Co‑occurrence (simple token co‑occur by line proximity)
    # -------------------------
    def cooccur(self, module_id: str, *, min_support: int = 1, max_pairs: int = 128, use_phase: bool = False, alpha: float = 1.0) -> Dict[str, Any]:
        if not module_id:
            raise a10_normalize("COOCCUR_PARAMS_INVALID", {"reason": "empty module_id"})
        if min_support < 1 or not (1 <= max_pairs <= 128):
            raise a10_normalize("COOCCUR_PARAMS_INVALID", {"reason": "range"})
        policy = "legacy"
        cap = max_pairs
        if use_phase:
            phase_cap = self.phase.compute_phase_window(1.0, alpha)
            cap = min(cap, max(1, phase_cap))
            policy = "phase"
        # Very light‑weight co‑occurrence: pair any two tokens appearing on the same line near target module file.
        file = None
        for p in self.root.rglob(f"{module_id}.REEF"):
            file = p
            break
        if not file:
            raise a10_normalize("MODULE_NOT_FOUND", {"module_id": module_id})
        rp = normalize_path(file)
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        try:
            with rp.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    toks = [t for t in re.split(r"[^\wψ@]+", line.strip()) if t]
                    # Count all unordered pairs in the line
                    for a_idx in range(len(toks)):
                        for b_idx in range(a_idx + 1, len(toks)):
                            a, b = sorted((toks[a_idx], toks[b_idx]))
                            counts[(a, b)] += 1
        except Exception:
            pass
        # Rank and emit with pointer‑only evidence
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        out_pairs: List[Dict[str, Any]] = []
        for (a, b), freq in ranked:
            if freq < min_support:
                continue
            if len(out_pairs) >= cap:
                break
            out_pairs.append({
                "motif_a": a,
                "motif_b": b,
                "support": freq,
                "evidence": {"file": str(rp), "start_line": 0, "end_line": 0},
            })
        return _ascii_keys_only({"pairs": out_pairs, "policy": policy})

    # -------------------------
    # Reflection (ontology‑gated stub)
    # -------------------------
    def get_reflection(self, a: str, b: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.ontology.validated:
            raise a10_normalize("ONTOLOGY_MISSING", {})
        if not a or not b:
            raise a10_normalize("REFLECTION_PARAMS_INVALID", {"reason": "empty a/b"})
        triad_ref = f"reef://triads/{hashlib.sha1(f'{a}|{b}'.encode()).hexdigest()[:6]}"
        confidence = 0.75
        env = self._gate_envelope({"sigma_phase": "phase:…"})
        out: Dict[str, Any] = {"triad": {"ref": triad_ref}, "confidence": confidence}
        if env:
            out["lineage"] = env if any(k in env for k in ("sigma_phase", "delta_hash")) else None
            if out.get("lineage") is None:
                out.pop("lineage", None)
        return _ascii_keys_only(out)

    # -------------------------
    # Field biases (stub)
    # -------------------------
    def field_biases(self) -> Dict[str, Any]:
        # Observer‑only static profile placeholder
        return _ascii_keys_only({"biases": [
            {"field": "ψ-resonance@Ξ", "weight": 1.0},
            {"field": "ψ-null@Ξ", "weight": 0.0},
        ]})

    # -------------------------
    # Build reflections (observer‑only cache hydration stub)
    # -------------------------
    def build_reflections(self, limit: int = 1000, strategy: Optional[str] = None) -> Dict[str, Any]:
        built = min(max(0, limit), 50000)
        return _ascii_keys_only({"built": built, "updated": 0, "skipped": 0})

    # -------------------------
    # Watch .REEF file mtimes (deterministic reload signaling)
    # -------------------------
    def watch_reef_files(self, paths: List[str]) -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        for p in paths:
            rp = normalize_path(p)
            try:
                mtime = Path(rp).stat().st_mtime
            except FileNotFoundError:
                mtime = 0.0
            events.append({"file": str(rp), "start_line": 0, "end_line": 0, "mtime": mtime})
        return _ascii_keys_only({"events": events})

    # -------------------------
    # Telemetry snapshot
    # -------------------------
    def telemetry_snapshot(self) -> Dict[str, Any]:
        return self.telemetry.snapshot(self.flags)

    # -------------------------
    # Index helpers
    # -------------------------
    def list_modules(self) -> Dict[str, Any]:
        if not self.index:
            raise a10_normalize("NOT_INITIALIZED", {})
        return _ascii_keys_only({"modules": self.index.list_modules()})

    def list_motifs(self) -> Dict[str, Any]:
        if not self.index:
            raise a10_normalize("NOT_INITIALIZED", {})
        return _ascii_keys_only({"motifs": self.index.list_motifs()})

# -----------------------------
# HTTP server (FastAPI) — optional runtime
# -----------------------------
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except Exception:  # pragma: no cover — server optional
    FastAPI = None  # type: ignore


def _adapter_or_500(app_state: Dict[str, Any]) -> MMMReefAdapter:
    adp: MMMReefAdapter = app_state.setdefault("adapter", MMMReefAdapter())  # type: ignore
    return adp


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI not available. Install fastapi[standard] to run the server.")
    app = FastAPI(title="MMM REEF Adapter", version=APP_VERSION)
    state: Dict[str, Any] = {}

    def _wrap(callable_, *args, **kwargs):
        try:
            out = callable_(*args, **kwargs)
            return JSONResponse(content=out)
        except A10Error as e:
            raise HTTPException(status_code=400, detail=e.to_wire())

    @app.post("/v1/mmm/reef/open_index")
    def open_index(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(adp.open_index, body.get("index_path"), body.get("archive_glob"))

    @app.get("/v1/mmm/reef/index")
    def get_index():
        adp = _adapter_or_500(state)
        modules = adp.list_modules()
        motifs = adp.list_motifs()
        return JSONResponse(content=_ascii_keys_only({"modules": modules["modules"], "motifs": motifs["motifs"], "anchors": []}))

    @app.post("/v1/mmm/reef/ontology")
    def load_ontology(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(adp.load_ontology_bundle, body.get("ontology_path"))

    @app.post("/v1/mmm/reef/reflection")
    def reflection(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(adp.get_reflection, body.get("a", ""), body.get("b", ""), body.get("hints") or {})

    @app.get("/v1/mmm/reef/field_biases")
    def get_biases():
        adp = _adapter_or_500(state)
        return JSONResponse(content=adp.field_biases())

    @app.post("/v1/mmm/reef/scan_motifs")
    def scan(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(
            adp.scan_motifs,
            body.get("query", ""),
            int(body.get("limit", 64)),
            use_phase=bool(body.get("use_phase", False)),
            alpha=float(body.get("alpha", 1.0)),
        )

    @app.post("/v1/mmm/reef/window")
    def window(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(
            adp.window,
            body.get("module_id", ""),
            int(body.get("line", 1)),
            body.get("radius"),
            include_snippet=bool(body.get("include_snippet", False)),
            use_phase=bool(body.get("use_phase", False)),
            alpha=float(body.get("alpha", 1.0)),
        )

    @app.post("/v1/mmm/reef/build_reflections")
    def build_refl(body: Dict[str, Any]):
        adp = _adapter_or_500(state)
        return _wrap(adp.build_reflections, int(body.get("limit", 1000)), body.get("strategy"))

    @app.get("/v1/mmm/reef/telemetry")
    def telemetry():
        adp = _adapter_or_500(state)
        return JSONResponse(content=adp.telemetry_snapshot())

    return app


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] == "serve":
        if FastAPI is None:
            print("FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
            return 2
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
        return 0
    # CLI: quick demo actions
    adp = MMMReefAdapter()
    print(json.dumps(adp.open_index(), indent=2))
    print(json.dumps(adp.list_modules(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# End_of_File
