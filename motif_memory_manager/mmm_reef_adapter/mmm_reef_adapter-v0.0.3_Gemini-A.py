# mmm_reef_adapter.py
#
# MMM REEF Adapter — Observer-only bridge for Reef index/ontology access,
# bounded evidence windows, co-occurrence scans, reflection scaffolds, and mtime signals.
#
# Layering & provenance (PDP/RFC family):
#   - PDP-0001 (Protocol for RFC-Driven Symbolic Artifact Generation): generation & layering flow
#   - RFC-CORE-006 (Motif Memory Manager): STMM/LTMM, exchange/integrity hooks, replay window
#   - RFC-0005 (Motif Transmission Across Time): lawful decay & resurrection concepts
#   - RFC-0006 (Motif Field Coherence Geometry): triadic closure & coherence geometry
#   - RFC-0007 (Motif Ontology Format & Transfer): ontology bundle & triads
#   - RFC-0008 (Symbolic Resource Exchange): envelope/Σ_phase (structure-only at L2+)
#   - RFC-0009 (Coherence–Integrity Framework): integrity lineage & recovery bounds
#
# Policy (observer-only):
#   • Read-only access to files under /mnt/data
#   • ASCII keys on-wire; ψ-* symbols allowed in values/examples
#   • Any Reef-derived claim MUST include evidence:{file,start_line,end_line}
#
# API (project_local): binder-style class exposing adapter_entrypoints in App-Spec §4.1–4.8.
# No HTTP server here; embedders can wrap this class.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any
import fnmatch
import json
import os
import re
import time
import hashlib

# -------------------------------
# Constants & Config Defaults
# -------------------------------

WHITELIST_ROOT = Path("/mnt/data").resolve()
DEFAULT_INDEX = "index.REEF"
DEFAULT_SHARDS_GLOB = "TheReefArchive-*.REEF"
DEFAULT_WINDOW_RADIUS = 24
MAX_WINDOW_RADIUS = 48
REFLECTIONS_LIMIT = 50_000

# Error model (App-Spec §7)
class AdapterErrorCode:
    NOT_INITIALIZED = "NOT_INITIALIZED"
    REEF_INDEX_NOT_FOUND = "REEF_INDEX_NOT_FOUND"
    INVALID_FORMAT = "INVALID_FORMAT"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    ANCHOR_NOT_FOUND = "ANCHOR_NOT_FOUND"
    RADIUS_INVALID = "RADIUS_INVALID"
    SCAN_LIMIT_REACHED = "SCAN_LIMIT_REACHED"
    ONTOLOGY_MISSING = "ONTOLOGY_MISSING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class Telemetry:
    reef_index_modules_count: int = 0
    reef_shard_count: int = 0
    reef_reflections_cache_size: int = 0
    reef_cooccur_candidate_rate: float = 0.0
    ascii_key_normalization_events: int = 0
    needs_reload: int = 0


@dataclass
class AdapterConfig:
    index_path: str = DEFAULT_INDEX
    shards_glob: str = DEFAULT_SHARDS_GLOB
    mtime_watch: bool = True
    window_radius_lines: int = DEFAULT_WINDOW_RADIUS
    max_window_radius_lines: int = MAX_WINDOW_RADIUS
    reflections_strategy: List[str] = field(default_factory=lambda: ["ontology", "file_reflections", "index_cooccur"]) 
    reflections_limit: int = REFLECTIONS_LIMIT
    env_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "MMM_REEF_INDEX_PATH": DEFAULT_INDEX,
        "MMM_REEF_SHARDS_GLOB": DEFAULT_SHARDS_GLOB,
        "MMM_WINDOW_RADIUS": DEFAULT_WINDOW_RADIUS,
        "MMM_FEATURE_FLAGS": "exchange,integrity,provenance,gliders",
    })

    def clamp_radius(self, requested: Optional[int]) -> Tuple[int, bool]:
        # clamp_rule: effective_radius := min(max_window_radius_lines, max(1, requested || window_radius_lines))
        if requested is None:
            r = self.window_radius_lines
        else:
            r = max(1, int(requested))
        clamped = min(self.max_window_radius_lines, r)
        return clamped, clamped != (requested if requested is not None else self.window_radius_lines)


@dataclass
class Evidence:
    file: str
    start_line: int
    end_line: int


@dataclass
class ReflectionResult:
    triad: Optional[Tuple[str, str, str]]
    confidence: float
    lineage: Dict[str, Any]


class ReefFormat:
    """Utilities to parse .REEF inputs as either text-ledger or JSON.

    We do not prescribe a single on-disk schema at Layer_2; instead we provide
    resilient readers that can operate on:
      • Plain text ledgers (line-indexed) — default
      • JSON documents — we flatten to text lines for searching

    All line numbers are 1-based.
    """

    MOTIF_LINE_RE = re.compile(r"\b(?:motif|Motif)[\s:_-]+([A-Za-z0-9_.\-]+)")

    @staticmethod
    def read_lines(path: Path) -> List[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise FileNotFoundError(f"Cannot read {path}: {e}")
        # try JSON → pretty-print → lines for consistent search
        try:
            obj = json.loads(text)
            text = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # Normalize newlines; enforce ASCII keys downstream only
        return text.splitlines()

    @staticmethod
    def list_motifs_from_lines(lines: List[str]) -> List[str]:
        motifs = []
        for ln in lines:
            m = ReefFormat.MOTIF_LINE_RE.search(ln)
            if m:
                motifs.append(m.group(1))
        # de-dup preserving order
        seen = set()
        out = []
        for m in motifs:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out


# -------------------------------
# Adapter Core
# -------------------------------

class MMMReefAdapter:
    """Observer-only adapter for the Motif Memory Manager (RFC-CORE-006).

    States (§6):
      cold → indexed → ready

    Public entrypoints (App-Spec §5):
      open_index, load_ontology_bundle, get_reflection, field_biases, scan_motifs,
      list_modules, list_motifs, find, window, cooccur, build_reflections, watch_reef_files

    Policy: read-only; ASCII keys on-wire; Reef claims carry Evidence.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.cfg = config or AdapterConfig()
        self.telemetry = Telemetry()
        self.state = "cold"  # cold/indexed/ready
        self.root = WHITELIST_ROOT
        self.index_path: Optional[Path] = None
        self.archives: List[Path] = []
        self.index_mtime: Optional[float] = None
        self.arch_mtimes: Dict[str, float] = {}
        self.ontology: Optional[Dict[str, Any]] = None
        self.reflections_cache: List[Tuple[str, str, str]] = []

    # --------------
    # Helpers
    # --------------

    def _guard_readonly_path(self, p: Path) -> Path:
        rp = p.resolve()
        if not str(rp).startswith(str(self.root)):
            raise PermissionError(f"Path outside whitelist root: {rp}")
        return rp

    def _ascii_keys(self, obj: Any) -> Any:
        # Enforce ASCII keys on-wire; values may include ψ-symbols.
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                try:
                    ak = str(k).encode("ascii", "ignore").decode("ascii")
                except Exception:
                    ak = str(k)
                if ak != k:
                    self.telemetry.ascii_key_normalization_events += 1
                new[ak] = self._ascii_keys(v)
            return new
        elif isinstance(obj, list):
            return [self._ascii_keys(x) for x in obj]
        else:
            return obj

    def _sha256_file(self, p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _list_archives(self, glob_pat: str) -> List[Path]:
        paths = sorted(self.root.glob(glob_pat))
        return [self._guard_readonly_path(p) for p in paths if p.is_file()]

    def _load_lines_cache(self, p: Path) -> List[str]:
        return ReefFormat.read_lines(p)

    # --------------
    # State & Index
    # --------------

    def open_index(self, index_path: Optional[str] = None, archive_glob: Optional[str] = None, checksum: bool = False) -> Dict[str, Any]:
        """Discover the index and shard archives (App-Spec §4.1, §6.2).

        Returns: {index_id, archives[]}
        Errors: REEF_INDEX_NOT_FOUND when missing/unreadable.
        """
        idx = Path(index_path or os.getenv("MMM_REEF_INDEX_PATH", self.cfg.index_path))
        glb = archive_glob or os.getenv("MMM_REEF_SHARDS_GLOB", self.cfg.shards_glob)
        idx = self._guard_readonly_path(self.root / idx)
        if not idx.exists():
            return self._ascii_keys({
                "error": AdapterErrorCode.REEF_INDEX_NOT_FOUND,
                "evidence": None,
            })
        self.index_path = idx
        self.index_mtime = idx.stat().st_mtime
        archives = self._list_archives(glb)
        self.archives = archives
        self.arch_mtimes = {str(p): p.stat().st_mtime for p in archives}
        self.state = "indexed"
        self.telemetry.reef_shard_count = len(archives)
        self.telemetry.reef_index_modules_count = len(archives)
        index_id = self._sha256_file(idx) if checksum else f"reef://{idx.name}:{int(self.index_mtime)}"
        return self._ascii_keys({
            "index_id": index_id,
            "archives": [str(p) for p in archives],
        })

    # --------------
    # Listings
    # --------------

    def list_modules(self) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        modules = []
        for p in self.archives:
            try:
                lines = self._load_lines_cache(p)
                modules.append({
                    "module_id": str(p),
                    "line_count": len(lines),
                })
            except Exception:
                modules.append({"module_id": str(p), "line_count": 0})
        return self._ascii_keys({"modules": modules})

    def list_motifs(self) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        motifs: List[str] = []
        for p in self.archives:
            try:
                lines = self._load_lines_cache(p)
                motifs.extend(ReefFormat.list_motifs_from_lines(lines))
            except Exception:
                continue
        # de-dup preserving order
        seen = set()
        ordered = []
        for m in motifs:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return self._ascii_keys({"motifs": ordered})

    # --------------
    # Ontology / Reflections
    # --------------

    def load_ontology_bundle(self, ontology_path: str) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        p = self._guard_readonly_path(self.root / ontology_path)
        try:
            text = p.read_text(encoding="utf-8")
            try:
                onto = json.loads(text)
            except json.JSONDecodeError:
                # minimal YAML support (naive) — interpret as key: value / lists
                import yaml  # type: ignore
                onto = yaml.safe_load(text)
        except Exception as e:
            return self._ascii_keys({"error": AdapterErrorCode.INVALID_FORMAT, "detail": str(e)})
        # basic validation per RFC-0007 (§8): require classes and/or triads
        if not isinstance(onto, dict) or not any(k in onto for k in ("classes", "triads")):
            return self._ascii_keys({"error": AdapterErrorCode.INVALID_FORMAT})
        self.ontology = onto
        self.state = "ready"
        return self._ascii_keys({
            "classes": len(onto.get("classes", {})),
            "triads": len(onto.get("triads", [])),
            "hash": hashlib.sha256(json.dumps(onto, sort_keys=True).encode("utf-8")).hexdigest(),
        })

    def get_reflection(self, a: str, b: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        if not self.ontology:
            return self._ascii_keys({"error": AdapterErrorCode.ONTOLOGY_MISSING})
        triads = self.ontology.get("triads", [])
        candidates = [t for t in triads if isinstance(t, (list, tuple)) and {a, b}.issubset(set(t))]
        if not candidates:
            return self._ascii_keys({"triad": None, "confidence": 0.0, "lineage": {"source": "ontology", "hints_used": False}})
        if len(candidates) > 1 and not hints:
            return self._ascii_keys({"error": AdapterErrorCode.AMBIGUOUS, "candidates": len(candidates)})
        chosen = candidates[0]
        conf = 0.5 + 0.5 * (1.0 / len(candidates))
        if hints and isinstance(hints, dict):
            # trivial bump if hint matches any element
            hv = set(map(str, hints.values()))
            if hv & set(map(str, chosen)):
                conf = min(0.99, conf + 0.1)
        result = ReflectionResult(triad=tuple(chosen[:3]), confidence=conf, lineage={"source": "ontology", "hints_used": bool(hints)})
        return self._ascii_keys(asdict(result))

    # --------------
    # Scans & Windows
    # --------------

    def find(self, query: str, module_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        patt = re.compile(re.escape(query), re.IGNORECASE)
        hits = []
        files = [Path(module_id)] if module_id else self.archives
        for p in files:
            if not Path(p).exists():
                continue
            lines = self._load_lines_cache(Path(p))
            for i, ln in enumerate(lines, start=1):
                if patt.search(ln):
                    hits.append({
                        "module_id": str(p),
                        "line": i,
                        "context": ln.strip(),
                        "evidence": asdict(Evidence(file=str(p), start_line=i, end_line=i)),
                    })
                    if len(hits) >= limit:
                        break
        return self._ascii_keys({"items": hits})

    def window(self, module_id: str, line: int, radius: Optional[int] = None) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        p = Path(module_id)
        if not p.exists():
            return self._ascii_keys({"error": AdapterErrorCode.MODULE_NOT_FOUND, "module_id": module_id})
        lines = self._load_lines_cache(p)
        eff_r, clamped = self.cfg.clamp_radius(radius)
        if clamped:
            err = AdapterErrorCode.RADIUS_INVALID
        else:
            err = None
        start = max(1, int(line) - eff_r)
        end = min(len(lines), int(line) + eff_r)
        snippet = "\n".join(lines[start - 1 : end])
        out = {
            "snippet": snippet,
            "start_line": start,
            "end_line": end,
            "evidence": asdict(Evidence(file=str(p), start_line=start, end_line=end)),
        }
        if err:
            out["warning"] = err
        return self._ascii_keys(out)

    def scan_motifs(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        lim = int(limit) if limit is not None else 2000
        patt = re.compile(query, re.IGNORECASE)
        items = []
        scanned = 0
        for p in self.archives:
            lines = self._load_lines_cache(p)
            for i, ln in enumerate(lines, start=1):
                scanned += 1
                if patt.search(ln):
                    items.append({
                        "module_id": str(p),
                        "line": i,
                        "context": ln.strip(),
                        "evidence": asdict(Evidence(file=str(p), start_line=max(1, i-1), end_line=min(len(lines), i+1))),
                    })
                if len(items) >= lim:
                    return self._ascii_keys({"items": items, "error": AdapterErrorCode.SCAN_LIMIT_REACHED})
        # Update simple rate metric (items per second baseline ~ scans / now)
        self.telemetry.reef_cooccur_candidate_rate = float(scanned)
        return self._ascii_keys({"items": items})

    # --------------
    # Co-Occurrence & Biases
    # --------------

    def field_biases(self) -> Dict[str, Any]:
        """Compute a simple bias profile from motif token frequencies across archives.
        This is an observer statistic; no mutation.
        """
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        counts: Dict[str, int] = {}
        total = 0
        for p in self.archives:
            lines = self._load_lines_cache(p)
            for ln in lines:
                m = ReefFormat.MOTIF_LINE_RE.search(ln)
                if m:
                    k = m.group(1)
                    counts[k] = counts.get(k, 0) + 1
                    total += 1
        biases = [{"motif": k, "p": (v / total if total else 0.0), "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        return self._ascii_keys({"biases": biases})

    def cooccur(self, motifs: Optional[List[str]] = None, radius: Optional[int] = None, alpha: float = 1.0) -> Dict[str, Any]:
        """Scan for co-occurrence of motifs within a bounded evidence window.
        Parameters reflect the math in App-Spec §3: Δτ_phase = α · EMA32(ℂ), α∈[0.5,2.0].
        Here we treat radius as the controlling bound over text lines.
        """
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        if not (0.5 <= float(alpha) <= 2.0):
            alpha = 1.0
        eff_r, _ = self.cfg.clamp_radius(radius)
        target = set(motifs or [])
        pair_counts: Dict[Tuple[str, str], int] = {}
        # naive sliding window over motif indices per file
        for p in self.archives:
            lines = self._load_lines_cache(p)
            motif_index: List[Tuple[int, str]] = []  # (line, motif)
            for i, ln in enumerate(lines, start=1):
                for m in ReefFormat.MOTIF_LINE_RE.findall(ln):
                    if (not target) or (m in target):
                        motif_index.append((i, m))
            # co-occur if |Δline| ≤ eff_r
            for i in range(len(motif_index)):
                li, mi = motif_index[i]
                j = i + 1
                while j < len(motif_index) and (motif_index[j][0] - li) <= eff_r:
                    lj, mj = motif_index[j]
                    a, b = sorted((mi, mj))
                    if a != b:
                        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
                    j += 1
        pairs = [{"a": a, "b": b, "count": c} for (a, b), c in sorted(pair_counts.items(), key=lambda kv: -kv[1])]
        return self._ascii_keys({"pairs": pairs, "effective_radius": eff_r, "alpha": float(alpha)})

    # --------------
    # Reflection Cache
    # --------------

    def build_reflections(self, limit: Optional[int] = None, strategy: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        lim = int(limit) if limit is not None else self.cfg.reflections_limit
        strat = strategy or self.cfg.reflections_strategy
        built = 0
        skipped = 0
        updated = 0
        cache = set(map(tuple, self.reflections_cache))
        # Strategy: from ontology triads first, then cooccur-inferred dyads
        if self.ontology and "ontology" in strat:
            for t in self.ontology.get("triads", [])[:lim]:
                key = tuple(t[:3])
                if key in cache:
                    skipped += 1
                else:
                    self.reflections_cache.append(key)
                    cache.add(key)
                    built += 1
                if (built + skipped) >= lim:
                    break
        # Lightweight inference via co-occur if space remains
        if (built + skipped) < lim and "index_cooccur" in strat:
            pairs = self.cooccur().get("pairs", [])
            # forge triads heuristically: chain top co-occurring pairs (a,b) with any c that co-occurs with either
            for pr in pairs:
                if (built + skipped) >= lim:
                    break
                a, b = pr["a"], pr["b"]
                c = None
                for pr2 in pairs:
                    if pr2["a"] in (a, b):
                        c = pr2["b"]
                        break
                    if pr2["b"] in (a, b):
                        c = pr2["a"]
                        break
                if c and a != c and b != c:
                    key = tuple(sorted((a, b, c)))
                    if key not in cache:
                        self.reflections_cache.append(key)  # observer heuristic only
                        cache.add(key)
                        built += 1
                    else:
                        skipped += 1
        self.telemetry.reef_reflections_cache_size = len(self.reflections_cache)
        if (built + skipped) > self.cfg.reflections_limit:
            return self._ascii_keys({"built": built, "updated": updated, "skipped": skipped, "error": AdapterErrorCode.SCAN_LIMIT_REACHED})
        return self._ascii_keys({"built": built, "updated": updated, "skipped": skipped})

    # --------------
    # MTime Watch
    # --------------

    def watch_reef_files(self) -> Dict[str, Any]:
        if self.state == "cold":
            return self._ascii_keys({"error": AdapterErrorCode.NOT_INITIALIZED})
        changed = []
        # index
        if self.index_path and self.index_path.exists():
            mt = self.index_path.stat().st_mtime
            if self.index_mtime and mt != self.index_mtime:
                changed.append(str(self.index_path))
                self.index_mtime = mt
        # archives
        for p in self.archives:
            mt = p.stat().st_mtime
            sp = str(p)
            if sp not in self.arch_mtimes:
                self.arch_mtimes[sp] = mt
                continue
            if self.arch_mtimes[sp] != mt:
                changed.append(sp)
                self.arch_mtimes[sp] = mt
        if changed:
            self.telemetry.needs_reload += 1
        return self._ascii_keys({"changed": changed})

    # --------------
    # Telemetry dump
    # --------------

    def get_telemetry(self) -> Dict[str, Any]:
        return self._ascii_keys(asdict(self.telemetry))


# -------------------------------
# Minimal CLI for local testing (Appx A — Examples E.1–E.4)
# -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMM REEF Adapter (observer-only)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open_index")
    p_open.add_argument("--index", default=os.getenv("MMM_REEF_INDEX_PATH", DEFAULT_INDEX))
    p_open.add_argument("--glob", default=os.getenv("MMM_REEF_SHARDS_GLOB", DEFAULT_SHARDS_GLOB))
    p_open.add_argument("--checksum", action="store_true")

    sub.add_parser("list_modules")
    sub.add_parser("list_motifs")

    p_find = sub.add_parser("find")
    p_find.add_argument("query")
    p_find.add_argument("--module")
    p_find.add_argument("--limit", type=int, default=50)

    p_win = sub.add_parser("window")
    p_win.add_argument("module_id")
    p_win.add_argument("line", type=int)
    p_win.add_argument("--radius", type=int)

    p_onto = sub.add_parser("load_ontology_bundle")
    p_onto.add_argument("ontology_path")

    p_refl = sub.add_parser("get_reflection")
    p_refl.add_argument("a")
    p_refl.add_argument("b")

    p_bias = sub.add_parser("field_biases")

    p_scan = sub.add_parser("scan_motifs")
    p_scan.add_argument("query")
    p_scan.add_argument("--limit", type=int)

    p_co = sub.add_parser("cooccur")
    p_co.add_argument("--motif", action="append")
    p_co.add_argument("--radius", type=int)
    p_co.add_argument("--alpha", type=float, default=1.0)

    p_build = sub.add_parser("build_reflections")
    p_build.add_argument("--limit", type=int)

    sub.add_parser("watch_reef_files")
    sub.add_parser("telemetry")

    args = parser.parse_args()

    adapter = MMMReefAdapter()

    def jprint(d: Dict[str, Any]):
        print(json.dumps(adapter._ascii_keys(d), ensure_ascii=True, indent=2))

    if args.cmd == "open_index":
        jprint(adapter.open_index(index_path=args.index, archive_glob=args.glob, checksum=args.checksum))
    elif args.cmd == "list_modules":
        jprint(adapter.list_modules())
    elif args.cmd == "list_motifs":
        jprint(adapter.list_motifs())
    elif args.cmd == "find":
        jprint(adapter.find(args.query, module_id=args.module, limit=args.limit))
    elif args.cmd == "window":
        jprint(adapter.window(args.module_id, args.line, radius=args.radius))
    elif args.cmd == "load_ontology_bundle":
        jprint(adapter.load_ontology_bundle(args.ontology_path))
    elif args.cmd == "get_reflection":
        jprint(adapter.get_reflection(args.a, args.b))
    elif args.cmd == "field_biases":
        jprint(adapter.field_biases())
    elif args.cmd == "scan_motifs":
        jprint(adapter.scan_motifs(args.query, limit=args.limit))
    elif args.cmd == "cooccur":
        jprint(adapter.cooccur(motifs=args.motif, radius=args.radius, alpha=args.alpha))
    elif args.cmd == "build_reflections":
        jprint(adapter.build_reflections(limit=args.limit))
    elif args.cmd == "watch_reef_files":
        jprint(adapter.watch_reef_files())
    elif args.cmd == "telemetry":
        jprint(adapter.get_telemetry())

# End_of_File
