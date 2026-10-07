"""Score mouse-to-human optimal-transport (OT) plans by GO correspondence.

A plan is split into blocks of co-transported mouse and human genes, and each
block is scored by how much more Gene Ontology (GO) biological-process
annotation its two gene sets share than annotation-matched random gene sets
would. The scores are signed excess-support measures, not probabilities of
correctness.

Public entry points
-------------------
``get_go_files``
    Locate the three GO input files and fingerprint them for caching.
``score_ot_plan``
    Score one plan and return ``(Q, S, block_scores)``.
``GOBlockEvaluator``
    Load GO once and score or compare many plans on one gene universe.

Dependencies: numpy, pandas, scipy (and scikit-learn and threadpoolctl for
block detection). No GO package is required.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import os
import pickle
import tempfile
import warnings
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .extract_dense_block import (
    _expand_brim_cores,
    _validate_overlap_eta,
    extract_dense_blocks,
    extract_dense_blocks_adaptive,
    extract_dense_blocks_coclustered,
    extract_dense_blocks_hard_brim,
    extract_dense_blocks_lpawb,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
GO_FILE_NAMES = ("go-basic.obo", "MOUSE-mod.gaf.gz", "HUMAN-uniprot.gaf.gz")
MOUSE_TAXON = 10090
HUMAN_TAXON = 9606

EXPERIMENTAL = frozenset(
    {
        "EXP",
        "IDA",
        "IPI",
        "IMP",
        "IGI",
        "IEP",
        "HTP",
        "HDA",
        "HMP",
        "HGI",
        "HEP",
    }
)
ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
ASPECTS = {
    "BP": ("biological_process", "P"),
    "MF": ("molecular_function", "F"),
    "CC": ("cellular_component", "C"),
}

AGGREGATIONS = ("mass", "uniform")
DETECTORS = (
    "coclustered",
    "adaptive",
    "hard_brim",
    "soft_brim",
    "brim",
    "lpawb",
    "modularity",
    "kl_nmf",
    "bayesian_nmf",
)
DETECTOR_ALIASES = {"brim": "hard_brim", "modularity": "hard_brim"}
NMF_DETECTORS = frozenset({"kl_nmf", "bayesian_nmf"})
ETA_DETECTORS = frozenset({"soft_brim"}) | NMF_DETECTORS
CORE_DETECTORS = frozenset({"hard_brim", "soft_brim"}) | NMF_DETECTORS
SEED_DETECTORS = frozenset({"lpawb"}) | NMF_DETECTORS
NMF_BACKENDS = ("auto", "numpy", "native", "torch")
DEFAULT_NMF_OPTIONS = ("auto", "cpu", "float64", 500, 1e-5)

# Block-detector failures that mean "no blocks", not "bad input".
_NO_BLOCK_MESSAGES = (
    "No dense blocks found.",
    "All candidate components were smaller",
    "No component passed `min_density_ratio`.",
)

# Bump when GO parsing or the pickled layout of the sources changes.
_GO_CACHE_FORMAT = 1
_NULL_BATCH_PERMUTATIONS = 8
_NULL_BATCH_PAIRS = 262144

# OBO term fields read by this module; everything else is skipped.
_KEPT_FIELDS = frozenset(
    {
        "id",
        "name",
        "namespace",
        "alt_id",
        "is_a",
        "relationship",
        "is_obsolete",
        "replaced_by",
    }
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _dataframe_array(frame):
    """Convert a DataFrame to a float array once, handling sparse storage."""
    sparse_columns = [
        isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes
    ]
    if any(sparse_columns):
        zero_filled = all(sparse_columns) and all(
            dtype.fill_value is not pd.NA and dtype.fill_value == 0
            for dtype in frame.dtypes
        )
        if zero_filled:
            return frame.sparse.to_coo().toarray().astype(float, copy=False)
        frame = frame.apply(
            lambda column: (
                column.sparse.to_dense()
                if isinstance(column.dtype, pd.SparseDtype)
                else column
            )
        )
    return frame.to_numpy(dtype=float)


def _is_integer(value):
    """True for real integers (numpy included) but not booleans."""
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_permutations(n_permutations, seed):
    if not _is_integer(n_permutations) or n_permutations < 2:
        raise ValueError("R must be an integer >= 2")
    if not _is_integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _scoring_file(name):
    """Find a GO file in the current folder or beside this module."""
    for folder in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (folder, folder / "upload"):
            path = directory / name
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(
        f"Put {name} beside go_block_quality.py or in the current folder"
    )


def get_go_files(go_folder=None):
    """Return the fingerprinted GO input files used by ``score_ot_plan``.

    Parameters
    ----------
    go_folder : str or Path, optional
        Folder holding ``go-basic.obo``, ``MOUSE-mod.gaf.gz`` and
        ``HUMAN-uniprot.gaf.gz``. When omitted, the current folder and the
        folder of this module (and an ``upload`` subfolder of each) are
        searched.

    Returns
    -------
    tuple of (str, int, int)
        ``(path, size, mtime_ns)`` for the ontology, the mouse annotations
        and the human annotations, in that order. The tuple is hashable and
        changes whenever a file changes, so it doubles as a cache key.

    Raises
    ------
    FileNotFoundError
        If any of the three files cannot be found.
    """
    paths = []
    for name in GO_FILE_NAMES:
        if go_folder is None:
            path = _scoring_file(name)
        else:
            path = (Path(go_folder).expanduser() / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file does not exist: {path}")
        paths.append(path)
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    )


# --------------------------------------------------------------------------
# GO parsing
# --------------------------------------------------------------------------
class _Ontology:
    """Minimal OBO reader: term lookup, obsolescence and ancestors."""

    def __init__(self, path, aspect):
        self.namespace, self.gaf_aspect = ASPECTS[aspect]
        self.terms, self.aliases = {}, {}
        self.version = None
        self._read(path)

    def _read(self, path):
        stanza, fields = None, defaultdict(list)

        def finish():
            if stanza == "Term" and fields.get("id"):
                term = fields["id"][0]
                self.terms[term] = dict(fields)
                for alias in fields.get("alt_id", []):
                    self.aliases[alias] = term

        with _open_text(path) as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("["):
                    finish()
                    stanza, fields = line.strip("[]"), defaultdict(list)
                elif ": " in line:
                    key, value = line.split(": ", 1)
                    if stanza is None and key == "data-version":
                        self.version = value
                    elif stanza == "Term" and key in _KEPT_FIELDS:
                        if key != "relationship" or value.startswith(
                            "part_of "
                        ):
                            fields[key].append(value)
            finish()

    def resolve(self, term):
        """Map a GO id to a current term in this namespace, else None."""
        term = self.aliases.get(term, term)
        seen = set()
        while term in self.terms and term not in seen:
            seen.add(term)
            record = self.terms[term]
            if record.get("is_obsolete", ["false"])[0] != "true":
                in_namespace = (
                    record.get("namespace", [None])[0] == self.namespace
                )
                return term if in_namespace else None
            replacements = record.get("replaced_by", [])
            if len(replacements) != 1:
                return None
            term = self.aliases.get(replacements[0], replacements[0])
        return None

    @lru_cache(maxsize=None)
    def ancestors(self, term):
        """Return ``term`` and its ``is_a``/``part_of`` ancestors, no roots.

        Only annotation-safe relations are followed; ``regulates`` and
        ``has_part`` are never treated as ``is_a``.
        """
        result = {term}
        record = self.terms[term]
        parents = [x.split()[0] for x in record.get("is_a", [])]
        parents += [
            x.split()[1]
            for x in record.get("relationship", [])
            if x.startswith("part_of ")
        ]
        for parent in parents:
            parent = self.resolve(parent)
            if parent is not None:
                result.update(self.ancestors(parent))
        return frozenset(result - ROOTS)


def _allowed_evidence(evidence):
    """Evidence codes to keep, or None to keep every code."""
    if isinstance(evidence, str):
        if evidence not in {"experimental", "reviewed", "all"}:
            raise ValueError(
                "evidence must be experimental, reviewed, all, or a set of "
                "codes"
            )
        return EXPERIMENTAL if evidence == "experimental" else None
    return set(evidence)


class _Annotations:
    """Gene identities and direct GO annotations from one species' GAF."""

    def __init__(self, path, ontology, evidence, taxon):
        self.direct = defaultdict(set)
        self.aliases = defaultdict(set)
        self.symbols = set()
        self.ambiguous_aliases = {}
        self.mass_based_alias_choices = {}
        self.taxon = int(taxon)
        self.stats = Counter()
        self.headers = {}
        self._read(path, ontology, evidence, _allowed_evidence(evidence))

    def _read(self, path, ontology, evidence, allowed):
        taxon_text = str(self.taxon)
        seen_identity, resolved_terms = set(), {}
        with _open_text(path) as handle:
            for raw in handle:
                if raw.startswith("!"):
                    if ": " in raw:
                        key, value = raw[1:].strip().split(": ", 1)
                        self.headers[key] = value
                    continue
                if raw.isspace():
                    continue
                fields = raw.rstrip("\n").split("\t")
                self.stats["records"] += 1
                if len(fields) < 15:
                    self.stats["malformed"] += 1
                    continue
                if not self._matches_taxon(fields[12], taxon_text):
                    self.stats["wrong_taxon"] += 1
                    continue
                self._record_identity(fields, seen_identity)
                self._record_annotation(
                    fields, ontology, evidence, allowed, resolved_terms
                )

    @staticmethod
    def _matches_taxon(field, taxon_text):
        # The substring test is a cheap necessary condition for the exact
        # token match.
        return taxon_text in field and any(
            token.rsplit(":", 1)[-1] == taxon_text
            for token in field.split("|")
        )

    def _record_identity(self, fields, seen_identity):
        """Register symbol and aliases, independent of annotation evidence.

        A gene lacking experimental BP evidence must still prevent its
        symbol or alias from being assigned to a different gene.
        """
        symbol = fields[2]
        identity = (fields[0], fields[1], symbol, fields[10])
        if not symbol or identity in seen_identity:  # records repeat per gene
            return
        seen_identity.add(identity)
        self.symbols.add(symbol)
        aliases = {symbol, fields[1], fields[0] + ":" + fields[1]}
        aliases.update(fields[10].split("|"))
        for alias in aliases - {""}:
            self.aliases[alias].add(symbol)

    def _record_annotation(
        self, fields, ontology, evidence, allowed, resolved_terms
    ):
        if fields[8] != ontology.gaf_aspect:
            return
        if "NOT" in fields[3].split("|"):
            self.stats["negative_excluded"] += 1
            return
        code = fields[6]
        rejected = code == "ND" or (
            allowed is not None and code not in allowed
        )
        if rejected or (evidence == "reviewed" and code == "IEA"):
            self.stats["evidence_excluded"] += 1
            return
        go_id = fields[4]
        if go_id not in resolved_terms:
            resolved_terms[go_id] = ontology.resolve(go_id)
        term = resolved_terms[go_id]
        if term is None:
            self.stats["unresolved_GO"] += 1
            return
        if term in ROOTS:
            return
        self.direct[fields[2]].add(term)
        self.stats["accepted"] += 1

    def resolve_gene(self, label, choices=None):
        """Resolve a species-specific label to a gene symbol, or None.

        Exact GAF symbols from any valid record take precedence. A unique
        synonym or database identifier resolves to its gene even when that
        gene lacks the selected annotation evidence. For an ambiguous alias
        the candidate in ``choices`` is used (chosen from marginal plan
        mass); without a choice the alphabetically first candidate wins.
        GO annotations of different candidates are never combined.
        """
        label = str(label)
        if label in self.symbols:
            return label
        candidates = self.aliases.get(label, set())
        if len(candidates) > 1:
            candidates = sorted(candidates)
            self.ambiguous_aliases[label] = candidates
            selected = (choices or {}).get(label, candidates[0])
            if selected not in candidates:
                raise ValueError(
                    f"Invalid candidate {selected!r} for ambiguous alias "
                    f"{label!r}"
                )
            self.mass_based_alias_choices[label] = selected
            return selected
        return next(iter(candidates)) if candidates else None


# --------------------------------------------------------------------------
# Cached GO sources and evaluators
# --------------------------------------------------------------------------
def _go_cache_file(files):
    """Pickle path for parsed GO sources, or None when caching is off.

    ``GO_BLOCK_QUALITY_CACHE`` names the folder (empty, ``off``, ``0`` or
    ``none`` disables caching; the default is ``~/.cache/go_block_quality``).
    The key covers the format version and each input file's path, size and
    modification time, so changed files are parsed again.
    """
    folder = os.environ.get("GO_BLOCK_QUALITY_CACHE")
    if folder is not None and folder.strip().lower() in {
        "",
        "off",
        "0",
        "none",
    }:
        return None
    if folder:
        folder = Path(folder).expanduser()
    else:
        folder = Path.home() / ".cache" / "go_block_quality"
    key = hashlib.sha256(
        repr((_GO_CACHE_FORMAT, tuple(files))).encode()
    ).hexdigest()[:32]
    return folder / f"go_sources_{key}.pkl"


def _load_cached_sources(path):
    """Return the pickled sources, or None if missing or unusable."""
    if path is None or not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
    except Exception:  # corrupt or stale cache: caller parses again
        return None
    valid = (
        isinstance(loaded, tuple)
        and len(loaded) == 3
        and isinstance(loaded[0], _Ontology)
        and all(isinstance(x, _Annotations) for x in loaded[1:])
    )
    return loaded if valid else None


def _store_cached_sources(path, sources):
    """Best-effort atomic write; read-only or full disks are ignored."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        )
        try:
            with handle:
                pickle.dump(sources, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
    except Exception:
        pass


@lru_cache(maxsize=2)
def _cached_go_sources(files):
    """Parse the ontology and both annotation files (disk- and RAM-cached)."""
    path = _go_cache_file(files)
    loaded = _load_cached_sources(path)
    if loaded is not None:
        return loaded
    ontology = _Ontology(files[0][0], "BP")
    sources = (
        ontology,
        _Annotations(files[1][0], ontology, "experimental", MOUSE_TAXON),
        _Annotations(files[2][0], ontology, "experimental", HUMAN_TAXON),
    )
    _store_cached_sources(path, sources)
    return sources


def _axis_mass_choices(genes, masses, annotations):
    """Choose each ambiguous alias by the mass of OTHER resolved genes.

    Mass carried by ambiguous labels is excluded to avoid circular
    assignments. Ties, including candidates with zero mass, are broken by
    symbol order.
    """
    totals = defaultdict(float)
    for label, value in zip(genes, masses):
        if label in annotations.symbols:
            totals[label] += float(value)
        else:
            candidates = annotations.aliases.get(label, set())
            if len(candidates) == 1:
                totals[next(iter(candidates))] += float(value)
    return tuple(
        sorted(
            (
                label,
                min(annotations.aliases[label], key=lambda g: (-totals[g], g)),
            )
            for label in set(genes)
            if label not in annotations.symbols
            and len(annotations.aliases.get(label, ())) > 1
        )
    )


@lru_cache(maxsize=2)
def _cached_evaluator(
    mouse_genes, human_genes, files, mouse_choices, human_choices
):
    return GOBlockEvaluator(
        files[0][0],
        files[1][0],
        files[2][0],
        mouse_genes=mouse_genes,
        human_genes=human_genes,
        _sources=_cached_go_sources(files),
        _alias_choices=(dict(mouse_choices), dict(human_choices)),
    )


# --------------------------------------------------------------------------
# Score aggregation
# --------------------------------------------------------------------------
_TABLE_COLUMNS = [
    "block",
    "mouse_genes",
    "human_genes",
    "block_mass",
    "allocated_mass",
    "block_score",
    "corrected_block_score",
    "null_mean_block_score",
    "observed_similarity",
    "baseline_similarity",
    "shared_go_terms",
    "aggregation_weight",
    "plan_score_contribution",
    "corrected_plan_score_contribution",
]
_DERIVED_COLUMNS = frozenset(
    {
        "aggregation_weight",
        "plan_score_contribution",
        "corrected_plan_score_contribution",
    }
)


def _aggregate_block_scores(blocks, aggregation, coverage=1.0):
    """Return Q and a block table with weights and Q/S contributions."""
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation must be 'mass' or 'uniform'")
    if blocks.empty:
        return 0.0, pd.DataFrame(columns=_TABLE_COLUMNS)
    table = blocks[
        [c for c in _TABLE_COLUMNS if c not in _DERIVED_COLUMNS]
    ].copy()
    masses = table["allocated_mass"].to_numpy(dtype=float)
    qualities = table["block_score"].to_numpy(dtype=float)
    valid = (masses > 0) & np.isfinite(qualities)
    weights = np.zeros(len(table))
    if valid.any():
        if aggregation == "uniform":
            weights[valid] = 1 / valid.sum()
        else:
            scaled = masses[valid] / masses[valid].max()
            weights[valid] = scaled / scaled.sum()
    corrected = table["corrected_block_score"].to_numpy()
    table["aggregation_weight"] = weights
    table["plan_score_contribution"] = (
        coverage * weights * np.nan_to_num(qualities, nan=0.0)
    )
    table["corrected_plan_score_contribution"] = (
        coverage * weights * np.nan_to_num(corrected, nan=0.0)
    )
    score = float(np.clip(table["plan_score_contribution"].sum(), 0, 1))
    return score, table[_TABLE_COLUMNS]


# --------------------------------------------------------------------------
# Block detection for score_ot_plan
# --------------------------------------------------------------------------
class _BlockDetector:
    """Detect blocks in a plan and cache results on the evaluator.

    Three levels of reuse keep repeated scoring cheap: the final block list,
    the hard BRIM cores shared by ``hard_brim``/``soft_brim``/NMF, and the
    NMF fits (at most two per plan). Keys include the plan *content*, so a
    mutated DataFrame never reuses a stale result.
    """

    def __init__(self, evaluator, block_detection, eta, seed, nmf_options):
        self.evaluator = evaluator
        self.method = DETECTOR_ALIASES.get(block_detection, block_detection)
        self.block_detection = block_detection
        self.eta = eta
        self.seed = seed
        self.nmf_options = nmf_options

    def __call__(self, array):
        plan_key = (
            array.shape,
            hashlib.sha256(
                np.ascontiguousarray(array).view(np.uint8)
            ).digest(),
        )
        key = self._result_key(plan_key)
        cached = getattr(self.evaluator, "_last_detected", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if self.method in CORE_DETECTORS:
            found = self._core_based_blocks(array, plan_key)
        elif self.method == "lpawb":
            found = extract_dense_blocks_lpawb(array, seed=self.seed)
        else:
            found = self._thresholded_blocks(array)
            if found is None:  # nothing transported; result is not cached
                return []
        if isinstance(found, tuple):
            found = found[0]
        unique = self._unique_blocks(found, array.shape)
        self.evaluator._last_detected = (key, unique)
        return unique

    def _result_key(self, plan_key):
        key = (
            self.method,
            self.eta if self.method in ETA_DETECTORS else None,
            int(self.seed) if self.method in SEED_DETECTORS else None,
            plan_key,
        )
        if self.method in NMF_DETECTORS:
            key += self.nmf_options
        return key

    def _core_based_blocks(self, array, plan_key):
        cores = self._hard_cores(array, plan_key)
        if self.method in NMF_DETECTORS:
            return self._nmf_blocks(array, cores, plan_key)
        if self.method == "hard_brim":
            return cores
        return _expand_brim_cores(array, cores, self.eta, include_blocks=False)

    def _hard_cores(self, array, plan_key):
        cached = getattr(self.evaluator, "_last_hard_brim_cores", None)
        if cached is not None and cached[0] == plan_key:
            return cached[1]
        cores = [
            {
                "row_indices": block["row_indices"].copy(),
                "col_indices": block["col_indices"].copy(),
            }
            for block in extract_dense_blocks_hard_brim(array)
        ]
        self.evaluator._last_hard_brim_cores = (plan_key, cores)
        return cores

    def _fit_key(self):
        key = (self.method, int(self.seed))
        if self.nmf_options != DEFAULT_NMF_OPTIONS:
            key += self.nmf_options
        return key

    def _nmf_blocks(self, array, cores, plan_key):
        try:
            from .nmf_blocks import blocks_from_fit, fit_nmf
        except ImportError:
            from nmf_blocks import blocks_from_fit, fit_nmf
        evaluator = self.evaluator
        cache = getattr(evaluator, "_nmf_fits", None)
        if cache is None or cache[0] != plan_key:
            cache = (plan_key, {})
            evaluator._nmf_fits = cache
        fits, key = cache[1], self._fit_key()
        if key not in fits:
            if len(fits) >= 2:  # bound memory even if many seeds are tried
                fits.pop(next(iter(fits)))
            backend, device, dtype, max_iter, tol = self.nmf_options
            fits[key] = fit_nmf(
                array,
                method=self.method,
                cores=cores,
                seed=self.seed,
                backend=backend,
                device=device,
                dtype=dtype,
                max_iter=max_iter,
                tol=tol,
            )
        return blocks_from_fit(array, fits[key], self.eta)

    def _thresholded_blocks(self, array):
        """Coclustered/adaptive detection; None when nothing is transported.

        Entries tied at the median are kept active, and the input is
        normalised only for detection so tiny, unbalanced masses stay
        stable.
        """
        nonzero = array[array > 0]
        if not nonzero.size:
            return None
        scaled = array / nonzero.max()
        threshold = np.nextafter(float(np.median(scaled[scaled > 0])), -np.inf)
        active = scaled > threshold
        groups = min(
            10, int(active.any(axis=1).sum()), int(active.any(axis=0).sum())
        )
        if groups < 2:
            return extract_dense_blocks(scaled, activity_threshold=threshold)
        if self.block_detection == "adaptive":
            method = extract_dense_blocks_adaptive
        else:
            method = extract_dense_blocks_coclustered
        return method(scaled, n_clusters=groups, activity_threshold=threshold)

    def _unique_blocks(self, found, shape):
        unique, seen = [], set()
        for block in found:
            rows, cols = self.evaluator._indices(block, shape)
            key = (tuple(sorted(rows)), tuple(sorted(cols)))
            if key not in seen:
                unique.append(
                    {"row_indices": rows.copy(), "col_indices": cols.copy()}
                )
                seen.add(key)
        return unique


# --------------------------------------------------------------------------
# Public scoring function
# --------------------------------------------------------------------------
def _validate_score_arguments(
    aggregation, block_detection, eta, nmf_options, R, seed
):
    """Validate ``score_ot_plan`` arguments; return the checked ``eta``."""
    backend, device, dtype, max_iter, tol = nmf_options
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation must be 'mass' or 'uniform'")
    if backend not in NMF_BACKENDS:
        raise ValueError(
            "nmf_backend must be 'auto', 'numpy', 'native', or 'torch'"
        )
    if not _is_integer(max_iter) or max_iter < 1:
        raise ValueError("nmf_max_iter must be a positive integer")
    numeric = isinstance(tol, (int, float, np.integer, np.floating))
    if (
        isinstance(tol, (bool, np.bool_))
        or not numeric
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("nmf_tol must be a finite positive number")
    custom_nmf = (
        backend,
        device,
        dtype,
        int(max_iter),
        float(tol),
    ) != DEFAULT_NMF_OPTIONS
    if custom_nmf and block_detection not in NMF_DETECTORS:
        raise ValueError(
            "nmf_backend/device/dtype/max_iter/tol only apply to NMF detectors"
        )
    if block_detection not in DETECTORS:
        raise ValueError(
            "block_detection must be 'adaptive', 'hard_brim', 'soft_brim', "
            "'lpawb', 'coclustered', 'kl_nmf', or 'bayesian_nmf'; 'brim' and "
            "'modularity' alias 'hard_brim'"
        )
    if block_detection in ETA_DETECTORS:
        eta = _validate_overlap_eta(eta)
    elif eta is not None:
        raise ValueError(
            "eta is only accepted with block_detection='soft_brim', "
            "'kl_nmf', or 'bayesian_nmf'; omit it for other detectors"
        )
    _validate_permutations(R, seed)
    return eta


def score_ot_plan(
    plan,
    go_files,
    aggregation="mass",
    *,
    block_detection="coclustered",
    eta=None,
    R=20,
    seed=0,
    nmf_backend="auto",
    nmf_device="cpu",
    nmf_dtype="float64",
    nmf_max_iter=500,
    nmf_tol=1e-5,
):
    """Score an OT plan by GO functional correspondence of its blocks.

    The plan is the only data input. Blocks of co-transported genes are
    detected without using GO, then each block is scored by the GO overlap
    of its mouse and human genes above an annotation-matched background.

    Parameters
    ----------
    plan : pandas.DataFrame
        Nonnegative, finite transport mass. Rows are mouse gene names and
        columns are human gene names. It is not modified.
    go_files : tuple
        Output of :func:`get_go_files`: the OBO file and the mouse and human
        GAF files. Parsed GO data is cached in memory and on disk.
    aggregation : {"mass", "uniform"}, default "mass"
        How block scores are combined: weighted by allocated mass, or equal
        weight per block.
    block_detection : str, default "coclustered"
        Block detector; see *Block detectors* below. Changing it changes the
        blocks and therefore the scores, so use the same detector for every
        plan in a comparison.
    eta : float, optional
        Overlap threshold in (0, 1]. Required for ``soft_brim``,
        ``kl_nmf`` and ``bayesian_nmf``; passing it with any other detector
        raises ``ValueError``.
    R : int, default 20
        Number of shuffled-profile draws (>= 2). It sets Monte Carlo
        precision and runtime only and never changes detection. Increase it
        for close comparisons; the first draws are unchanged.
    seed : int, default 0
        Nonnegative seed. It drives the GO shuffles and, for ``lpawb`` and
        the NMF detectors, detection too (through independent streams).
        The same seed gives common draws across plans on the same genes.
    nmf_backend : {"auto", "numpy", "native", "torch"}, default "auto"
        NMF engine. ``auto`` uses compiled CPU updates when a C compiler is
        available and NumPy otherwise (a non-CPU device selects PyTorch);
        ``numpy`` is the reference path, ``native`` requires the compiled
        updates and ``torch`` forces PyTorch (also valid on CPU).
    nmf_device : {"cpu", "cuda", "cuda:N", "mps"}, default "cpu"
        Device for PyTorch fitting. Apple MPS needs
        ``nmf_dtype="float32"``.
    nmf_dtype : {"float64", "float32"}, default "float64"
        PyTorch precision. Reduced precision can change memberships near
        ``eta``; validate it on your hardware.
    nmf_max_iter : int, default 500
        Iteration cap for the NMF fit.
    nmf_tol : float, default 1e-5
        The fit stops when the relative objective decrease over five updates
        falls below this value. Smaller is stricter.

    All ``nmf_*`` options apply only to ``kl_nmf`` and ``bayesian_nmf``;
    non-default values with another detector raise ``ValueError``.

    Returns
    -------
    Q : float
        Raw GO score in [0, 1].
    S : float
        Shuffle-corrected score ``Q - mean(Q_shuffled)`` in [-1, 1].
    block_scores : pandas.DataFrame
        One row per block. ``block_score`` is ``q_b`` and
        ``corrected_block_score`` is ``s_b``; ``plan_score_contribution``
        sums to ``Q`` and ``corrected_plan_score_contribution`` to ``S``.
        ``block_scores.attrs`` records ``Q``, ``S``, ``raw_biological_score``
        (= ``Q``), ``null_mean_plan_score``, ``score_mc_standard_error``,
        ``n_permutations``/``R``, ``seed``, ``block_mass_coverage``,
        ``annotation_mass_coverage``, ``block_detection`` and ``eta``, plus
        the gene-label diagnostics ``merged_{mouse,human}_labels``,
        ``ambiguous_{mouse,human}_aliases`` and
        ``mass_based_{mouse,human}_alias_choices``. For NMF detectors,
        ``attrs["nmf_fit"]`` holds the fit report (``converged``,
        ``iterations``, ``backend``, objective history). Check
        ``converged``: reaching ``nmf_max_iter`` leaves it ``False``.

    Raises
    ------
    TypeError
        If ``plan`` is not a DataFrame.
    ValueError
        For invalid arguments, or negative or non-finite transport (values
        are never silently clipped).
    FileNotFoundError
        If a GO file is missing.

    Notes
    -----
    **Scores.** For block *b*, each cell's mass is split equally among the
    blocks covering it, giving normalised weights *w* over the block's
    cells. With ``sim`` the information-weighted GO similarity of a mouse
    and a human gene (below) and ``base`` its annotation-matched baseline::

        observed = sum(w * sim)
        expected = sum(w * base)
        q_b = clip(max(0, observed - expected) / (1 - expected), 0, 1)
        s_b = q_b - mean(q_b under R shuffled GO profiles)

    The plan scores are ``Q = coverage * sum(a_b * q_b)`` and
    ``S = coverage * sum(a_b * s_b)``, where *coverage* is the fraction of
    whole-plan mass inside the union of blocks, ``a_b`` is proportional to
    block allocated mass (``aggregation="mass"``) or ``1 / n_blocks``
    (``"uniform"``). Zero-mass blocks are undefined and get weight 0. If the
    total mass is zero, or no transported pair has usable annotations,
    ``Q = S = NaN``. Negative ``s_b`` are kept.

    ``sim`` is the Jaccard overlap of the two genes' propagated
    biological-process term sets, weighting each term by
    ``-log(frequency)`` (frequency averaged over the two species so neither
    dominates). Only experimental-evidence annotations are used, to limit
    orthology-inferred circularity. ``base`` is the mean ``sim`` over all
    gene pairs in the same pair of log2 annotation-burden bins. Shuffles
    permute whole GO profiles independently within these bins in each
    species, with the plan, blocks, mass allocation, frequencies and
    baseline fixed.

    Q and S are related views of one signal, not independent evidence, and
    are not significance tests. ``score_mc_standard_error`` measures Monte
    Carlo precision only.

    **Block detectors.** None of them uses GO.

    ``coclustered`` (default)
        Spectral coclustering with a median activity threshold.
    ``adaptive``
        First finds separated dense components of the bipartite activity
        graph, which can recover small scattered blocks, otherwise falls
        back to coclustering. No tuning parameters.
    ``hard_brim`` (aliases ``brim``, ``modularity``)
        Weighted BRIM modularity with module merging against a
        marginal-preserving reference (Barber 2007; Beckett 2016). Needs no
        cluster count or activity threshold, detects excess mass rather
        than filled rectangles, and assigns each gene to at most one
        module. Two deterministic starts are used. Modularity has a
        resolution limit.
    ``soft_brim``
        Expands the frozen hard cores once using transport above the
        marginal-preserving reference, so genes may join several blocks.
        With full-plan row and column marginals *r*, *c* and mass *T*, the
        excess of mouse gene *i* for human core *C_k* is
        ``a_ik = sum_{j in C_k} P_ij - r_i * sum_{j in C_k} c_j / T``. Gene
        *i* is proposed for core *k* if ``a_ik > 0`` and
        ``a_ik >= eta * max_l a_il`` (symmetric for human genes), using the
        ORIGINAL cores. Proposals are taken in descending normalised
        excess and accepted only when the currently uncovered cells they
        add have positive excess over ``r_i * c_j / T``. One pass; rejected
        proposals are not revisited. Smaller ``eta`` proposes more
        memberships. It cannot split modules that ``hard_brim`` merged.
    ``lpawb``
        Beckett's LPAwb+ from singleton starts on the smaller species:
        label updates with seeded random ties, positive-gain merging, and
        refinement until the objective stops improving.
    ``kl_nmf``, ``bayesian_nmf``
        Experimental overlapping fits. ``kl_nmf`` fits normalised transport
        with KL-NMF; ``bayesian_nmf`` adds half-normal ARD priors
        (a=5, b=2) with the rectangular MAP updates of Psorakis et al.
        (2011) after rescaling positive weights to average 1. Both start
        from the hard BRIM cores, use the core count as the initial rank,
        and place a gene in each component whose mass is at least ``eta``
        times its strongest. They are local fits, so a fixed rank can limit
        the modules recovered. Fits are shared across ``eta``, aggregation
        and ``R``.

    **Gene labels** (automatic; the input is not modified):

    * Labels resolving to one gene, and exact duplicates, are merged by
      summing their rows or columns before detection. Total mass is
      preserved and the gene counts once in the GO background.
    * An ambiguous alias (e.g. human ``TAZ`` -> ``TAFAZZIN`` or
      ``WWTR1``) maps to the candidate with the largest total mass among
      the species' OTHER unambiguously resolved labels (never the alias's
      own mass); ties go to the alphabetically first. Its transport is
      summed into the chosen gene and only that gene's annotations are
      used. This is a mass-based heuristic, not proof of identity, and can
      differ between plans.
    * Exact GAF symbols take precedence over synonyms; unique aliases
      resolve directly. Distinct unknown labels stay separate and are
      unannotated.

    **Caching.** The evaluator is cached per gene axes. Repeated identical
    inputs reuse detected blocks, NMF fits and null samples, and a larger
    ``R`` extends existing samples. Keys include plan content, so mutating
    a plan never returns stale results.

    Examples
    --------
    >>> go_files = get_go_files("/path/to/go")          # doctest: +SKIP
    >>> q, s, blocks = score_ot_plan(                    # doctest: +SKIP
    ...     plan, go_files, block_detection="kl_nmf", eta=0.5,
    ...     nmf_device="cuda", nmf_max_iter=1000)
    >>> blocks.attrs["nmf_fit"]["converged"]             # doctest: +SKIP
    True
    """
    nmf_options = (nmf_backend, nmf_device, nmf_dtype, nmf_max_iter, nmf_tol)
    eta = _validate_score_arguments(
        aggregation, block_detection, eta, nmf_options, R, seed
    )
    nmf_options = (
        nmf_backend,
        nmf_device,
        nmf_dtype,
        int(nmf_max_iter),
        float(nmf_tol),
    )
    if not isinstance(plan, pd.DataFrame):
        raise TypeError(
            "plan must be a DataFrame with mouse row names and human column "
            "names"
        )
    mouse_genes = tuple(map(str, plan.index))
    human_genes = tuple(map(str, plan.columns))
    array = _dataframe_array(plan)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(
            "OT plans must be finite and nonnegative; negative values are "
            "not silently clipped"
        )
    # Marginal mass is the evidence for choosing an alias; totals are taken
    # before any alias merging, then ambiguous labels are excluded.
    mouse_mass = array.sum(axis=1)
    human_mass = array.sum(axis=0)
    if not np.isfinite(mouse_mass).all() or not np.isfinite(human_mass).all():
        raise ValueError("Plan marginals must be finite")
    sources = _cached_go_sources(go_files)
    evaluator = _cached_evaluator(
        mouse_genes,
        human_genes,
        go_files,
        _axis_mass_choices(mouse_genes, mouse_mass, sources[1]),
        _axis_mass_choices(human_genes, human_mass, sources[2]),
    )
    detector = _BlockDetector(
        evaluator, block_detection, eta, seed, nmf_options
    )
    result = evaluator.evaluate(
        array, detector=detector, aggregation=aggregation, R=R, seed=seed
    )
    blocks = result["blocks"]
    blocks.attrs["block_detection"] = block_detection
    blocks.attrs["eta"] = eta
    if block_detection in NMF_DETECTORS:
        fitted = getattr(evaluator, "_nmf_fits", None)
        fit_key = detector._fit_key()
        if fitted is not None and fit_key in fitted[1]:
            blocks.attrs["nmf_fit"] = dict(fitted[1][fit_key]["details"])
    summary = result["summary"]
    return (
        float(summary["raw_biological_score"]),
        float(summary["biological_score"]),
        blocks,
    )


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------
def _burden_bins(matrix):
    """Fixed log2 bins of propagated annotation counts; 0 has its own bin."""
    counts = np.diff(matrix.indptr)
    groups = np.zeros(len(counts), dtype=int)
    positive = counts > 0
    groups[positive] = 1 + np.floor(np.log2(counts[positive])).astype(int)
    return groups


class GOBlockEvaluator:
    """Load GO once and score many plans on one ordered gene universe.

    Default evidence is experimental, to limit reuse of orthology-inferred
    annotation. The fixed gene axes define the GO term frequencies and the
    annotation-matched baseline, so every plan is scored against the same
    definition and adding plans cannot change an existing score.

    Duplicate labels and aliases of one gene are merged (transport summed);
    ambiguous aliases use the candidate with the highest reference-plan
    mass. Axis resolution and merge/ambiguity records are kept in
    ``annotation_report``.

    Parameters
    ----------
    obo_path, mouse_gaf_path, human_gaf_path : str or Path
        Ontology and per-species annotation files.
    mouse_genes, human_genes : sequence of str
        Ordered, nonempty input axis labels. Duplicates and aliases are
        allowed; the original axes are kept in ``input_mouse_genes`` and
        ``input_human_genes`` and the merged ones in ``mouse_genes`` and
        ``human_genes``.
    aspect : {"BP", "MF", "CC"}, default "BP"
    evidence : str or collection, default "experimental"
        ``"experimental"``, ``"reviewed"``, ``"all"`` or a set of evidence
        codes.
    reference_plan : DataFrame, array or sparse matrix, optional
        Plan whose marginals decide ambiguous aliases (the highest
        marginal mass among other unambiguous genes; ties alphabetical).
        Without it every candidate has zero mass and ties resolve
        alphabetically. Use a common reference plan in :meth:`compare` to
        keep the choices fixed across candidates.
    """

    def __init__(
        self,
        obo_path,
        mouse_gaf_path,
        human_gaf_path,
        mouse_genes,
        human_genes,
        *,
        aspect="BP",
        evidence="experimental",
        reference_plan=None,
        _sources=None,
        _alias_choices=None,
    ):
        if aspect not in ASPECTS:
            raise ValueError("aspect must be BP, MF, or CC")
        self.input_mouse_genes = tuple(map(str, mouse_genes))
        self.input_human_genes = tuple(map(str, human_genes))
        for genes in (self.input_mouse_genes, self.input_human_genes):
            if not genes:
                raise ValueError("Each species must have nonempty gene labels")
        self.aspect, self.evidence = aspect, evidence
        mouse, human = self._load_sources(
            obo_path,
            mouse_gaf_path,
            human_gaf_path,
            aspect,
            evidence,
            _sources,
        )
        choices = self._alias_choices(
            _alias_choices, reference_plan, mouse, human
        )
        axes = [
            self._resolve_axis(genes, annotations, species_choices)
            for genes, annotations, species_choices in (
                (self.input_mouse_genes, mouse, choices[0]),
                (self.input_human_genes, human, choices[1]),
            )
        ]
        self._build_matrices(axes)
        self._build_report(axes, mouse, human)

    # -- construction helpers ---------------------------------------------
    def _load_sources(
        self, obo_path, mouse_gaf, human_gaf, aspect, evidence, sources
    ):
        if sources is None:
            self.ontology = _Ontology(obo_path, aspect)
            return (
                _Annotations(mouse_gaf, self.ontology, evidence, MOUSE_TAXON),
                _Annotations(human_gaf, self.ontology, evidence, HUMAN_TAXON),
            )
        self.ontology, mouse, human = sources
        if aspect != "BP" or evidence != "experimental":
            raise ValueError(
                "Cached GO sources require BP and experimental evidence"
            )
        # Each evaluator records only ambiguities seen on its own axes.
        mouse, human = copy.copy(mouse), copy.copy(human)
        for species in (mouse, human):
            species.ambiguous_aliases = {}
            species.mass_based_alias_choices = {}
        return mouse, human

    def _alias_choices(self, alias_choices, reference_plan, mouse, human):
        if alias_choices is not None:
            return alias_choices
        if reference_plan is None:
            return {}, {}
        shape = (len(self.input_mouse_genes), len(self.input_human_genes))
        if isinstance(reference_plan, pd.DataFrame):
            same_axes = (
                tuple(map(str, reference_plan.index)) == self.input_mouse_genes
                and tuple(map(str, reference_plan.columns))
                == self.input_human_genes
            )
            if not same_axes:
                raise ValueError(
                    "reference_plan axes must match the evaluator's input axes"
                )
            row_mass = np.asarray(reference_plan.sum(axis=1), dtype=float)
            col_mass = np.asarray(reference_plan.sum(axis=0), dtype=float)
            row_mass, col_mass = row_mass.ravel(), col_mass.ravel()
        else:
            if sparse.issparse(reference_plan):
                reference = reference_plan.toarray()
            else:
                reference = np.asarray(reference_plan)
            if reference.shape != shape:
                raise ValueError(
                    "reference_plan shape must match the evaluator's input "
                    "axes"
                )
            row_mass, col_mass = reference.sum(axis=1), reference.sum(axis=0)
        return (
            dict(_axis_mass_choices(self.input_mouse_genes, row_mass, mouse)),
            dict(_axis_mass_choices(self.input_human_genes, col_mass, human)),
        )

    def _resolve_axis(self, genes, annotations, choices):
        """Resolve one species' labels, merging labels that name one gene.

        Identities are assigned (including mass-based alias choices) and
        identical unknown labels are merged, preserving first-seen order.
        """
        names = [annotations.resolve_gene(g, choices) for g in genes]
        lookup, canonical, canonical_names, groups = {}, [], [], []
        members = defaultdict(list)
        for label, name in zip(genes, names):
            key = ("known", name) if name is not None else ("unknown", label)
            if key not in lookup:
                lookup[key] = len(canonical)
                canonical.append(name if name is not None else label)
                canonical_names.append(name)
            groups.append(lookup[key])
            members[canonical[lookup[key]]].append(label)
        propagated = []
        for name in canonical_names:
            direct = annotations.direct.get(name, set())
            propagated.append(
                set().union(*(self.ontology.ancestors(t) for t in direct))
                if direct
                else set()
            )
        return {
            "resolved": names,
            "canonical": tuple(canonical),
            "groups": np.asarray(groups, dtype=int),
            "merged": {g: m for g, m in members.items() if len(m) > 1},
            "terms": propagated,
            "missing": [
                g for g, terms in zip(canonical, propagated) if not terms
            ],
        }

    def _build_matrices(self, axes):
        self.mouse_genes, self.human_genes = (a["canonical"] for a in axes)
        self._row_groups, self._col_groups = (a["groups"] for a in axes)
        self.terms = sorted(set().union(*axes[0]["terms"], *axes[1]["terms"]))
        lookup = {term: k for k, term in enumerate(self.terms)}
        matrices = []
        for axis in axes:
            rows, cols = [], []
            for i, terms in enumerate(axis["terms"]):
                rows.extend([i] * len(terms))
                cols.extend(lookup[t] for t in terms)
            matrices.append(
                sparse.csr_matrix(
                    (np.ones(len(rows)), (rows, cols)),
                    shape=(len(axis["terms"]), len(self.terms)),
                )
            )
        self.Xm, self.Xh = matrices
        self.am = np.diff(self.Xm.indptr) > 0
        self.ah = np.diff(self.Xh.indptr) > 0

    def _build_report(self, axes, mouse, human):
        mouse_axis, human_axis = axes
        self.annotation_report = {
            "aspect": self.aspect,
            "evidence": str(self.evidence),
            "obo_version": self.ontology.version,
            "mouse_gaf_headers": mouse.headers,
            "human_gaf_headers": human.headers,
            "mouse_parse_counts": dict(mouse.stats),
            "human_parse_counts": dict(human.stats),
            "mouse_annotated_genes": int(self.am.sum()),
            "human_annotated_genes": int(self.ah.sum()),
            "mouse_unannotated_labels": mouse_axis["missing"],
            "human_unannotated_labels": human_axis["missing"],
            "mouse_resolved_symbols": mouse_axis["resolved"],
            "human_resolved_symbols": human_axis["resolved"],
            "mouse_merged_labels": mouse_axis["merged"],
            "human_merged_labels": human_axis["merged"],
            "mouse_ambiguous_aliases": dict(mouse.ambiguous_aliases),
            "human_ambiguous_aliases": dict(human.ambiguous_aliases),
            "mouse_mass_based_alias_choices": dict(
                mouse.mass_based_alias_choices
            ),
            "human_mass_based_alias_choices": dict(
                human.mass_based_alias_choices
            ),
            "terms": len(self.terms),
        }
        if not self.am.any() or not self.ah.any():
            warnings.warn(
                "One species has no usable annotations: biological scores "
                "will be NaN",
                stacklevel=3,
            )
        if mouse.stats["unresolved_GO"] or human.stats["unresolved_GO"]:
            warnings.warn(
                "Some GAF GO IDs could not be resolved in this OBO; inspect "
                "annotation_report",
                stacklevel=3,
            )

    # -- plan and block input ---------------------------------------------
    def _array(self, plan):
        """Validate a plan on the original axes and merge duplicate genes.

        DataFrame axes must equal the ORIGINAL ordered input labels.
        Negative or non-finite values are rejected. Rows and columns that
        resolve to one gene are summed, preserving total mass without
        mutating the input.
        """
        if isinstance(plan, pd.DataFrame):
            same_axes = (
                tuple(map(str, plan.index)) == self.input_mouse_genes
                and tuple(map(str, plan.columns)) == self.input_human_genes
            )
            if not same_axes:
                raise ValueError(
                    "Plan axes must exactly match the evaluator's ordered "
                    "gene universe. Reindex explicitly BEFORE detecting "
                    "blocks."
                )
            array = _dataframe_array(plan)
        elif sparse.issparse(plan):
            array = plan.toarray().astype(float, copy=False)
        elif hasattr(plan, "detach"):
            array = plan.detach().cpu().numpy().astype(float, copy=False)
        else:
            array = np.asarray(plan, dtype=float)
        shape = (len(self.input_mouse_genes), len(self.input_human_genes))
        if array.shape != shape:
            raise ValueError(
                "Plan shape differs from the evaluator gene universe"
            )
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(
                "OT plans must be finite and nonnegative; negative values "
                "are not silently clipped"
            )
        if len(self.mouse_genes) != len(self.input_mouse_genes):
            merged = np.zeros((len(self.mouse_genes), array.shape[1]))
            np.add.at(merged, self._row_groups, array)
            array = merged
        if len(self.human_genes) != len(self.input_human_genes):
            merged_transpose = np.zeros(
                (len(self.human_genes), array.shape[0])
            )
            np.add.at(merged_transpose, self._col_groups, array.T)
            array = merged_transpose.T
        if not np.isfinite(array).all():
            raise ValueError(
                "Merged transport mass overflows floating-point precision"
            )
        return array

    def _remap_blocks(self, blocks):
        """Project original-axis gene memberships onto the merged axes."""
        shape = (len(self.input_mouse_genes), len(self.input_human_genes))
        result = []
        for block in blocks:
            rows, cols = self._indices(block, shape)
            result.append(
                {
                    "row_indices": np.unique(self._row_groups[rows]),
                    "col_indices": np.unique(self._col_groups[cols]),
                }
            )
        return result

    @staticmethod
    def _indices(block, shape):
        """Return validated ``(rows, cols)`` index arrays for a block."""
        if "row_indices" in block and "col_indices" in block:
            rows, cols = block["row_indices"], block["col_indices"]
        elif "bbox" in block:
            r0, r1, c0, c1 = block["bbox"]
            ints = all(
                isinstance(v, (int, np.integer)) for v in (r0, r1, c0, c1)
            )
            if not ints or not (
                0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]
            ):
                raise ValueError(
                    "bbox must contain valid exclusive-end integer bounds in "
                    "ORIGINAL plan order"
                )
            rows, cols = np.arange(r0, r1), np.arange(c0, c1)
        else:
            raise ValueError(
                "Block needs original row_indices/col_indices or bbox; "
                "bbox_reordered alone is unsafe"
            )
        output = []
        for values, limit in ((rows, shape[0]), (cols, shape[1])):
            values = np.asarray(values)
            if (
                values.ndim != 1
                or values.size == 0
                or values.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "Block indices must be nonempty one-dimensional integer "
                    "arrays"
                )
            if (
                (values < 0).any()
                or (values >= limit).any()
                or len(np.unique(values)) != len(values)
            ):
                raise ValueError(
                    "Block indices are duplicated or out of bounds"
                )
            output.append(values.astype(int))
        return output

    def _find_blocks(self, array, blocks, detector, detector_kwargs):
        if blocks is not None:
            if detector is not None:
                raise ValueError("Supply blocks OR detector, not both")
            return list(blocks), None
        if detector is None:
            raise ValueError(
                "Supply existing blocks or an explicit detector callable"
            )
        try:
            found = detector(array, **(detector_kwargs or {}))
            found = found[0] if isinstance(found, tuple) else found
            return list(found), None
        except ValueError as error:
            if str(error).startswith(_NO_BLOCK_MESSAGES):
                return [], str(error)
            raise

    # -- GO similarity ------------------------------------------------------
    @staticmethod
    def _burden_groups(matrix):
        return _burden_bins(matrix)

    def _functional_similarity(self):
        """Return the cached ``(similarity, baseline)`` matrices.

        ``similarity[i, j]`` is the information-weighted Jaccard overlap of
        mouse gene *i* and human gene *j*. ``baseline`` is its exact mean
        under independent within-bin permutations of whole annotation
        profiles, i.e. the mean over each (mouse bin, human bin) block.
        """
        if hasattr(self, "_similarity"):
            return self._similarity, self._baseline
        n_mouse, n_human = int(self.am.sum()), int(self.ah.sum())
        if not n_mouse or not n_human:
            self._similarity = np.zeros((len(self.am), len(self.ah)))
            self._baseline = self._similarity.copy()
            self._information = np.zeros(len(self.terms))
            return self._similarity, self._baseline
        # Equal species weighting stops a larger universe dominating IC.
        freq_mouse = np.asarray(self.Xm.sum(axis=0)).ravel() / n_mouse
        freq_human = np.asarray(self.Xh.sum(axis=0)).ravel() / n_human
        frequency = (freq_mouse + freq_human) / 2
        information = -np.log(np.maximum(frequency, np.finfo(float).tiny))
        weighted_mouse = self.Xm.multiply(information).tocsr()
        intersection = (weighted_mouse @ self.Xh.T).toarray()
        total_mouse = np.asarray(weighted_mouse.sum(axis=1)).ravel()
        total_human = np.asarray(
            self.Xh.multiply(information).sum(axis=1)
        ).ravel()
        union = total_mouse[:, None] + total_human[None, :] - intersection
        valid_union = union > np.finfo(float).eps
        # Reuse the intersection storage rather than allocating a second
        # full mouse x human matrix.
        similarity = np.divide(
            intersection, union, out=intersection, where=valid_union
        )
        similarity[~valid_union] = 0
        np.clip(similarity, 0, 1, out=similarity)
        self._similarity = similarity
        self._baseline = self._matched_baseline(similarity)
        self._information = information
        return self._similarity, self._baseline

    def _matched_baseline(self, similarity):
        mouse_bins = _burden_bins(self.Xm)
        human_bins = _burden_bins(self.Xh)
        baseline = np.zeros_like(similarity)
        for mouse_bin in np.unique(mouse_bins):
            rows = np.flatnonzero(mouse_bins == mouse_bin)
            for human_bin in np.unique(human_bins):
                cols = np.flatnonzero(human_bins == human_bin)
                selection = np.ix_(rows, cols)
                baseline[selection] = similarity[selection].mean()
        return baseline

    # -- shuffled null ------------------------------------------------------
    def _annotation_permutations(self, n_permutations, seed):
        """Whole-profile shuffle maps within each species' burden bins.

        The first R maps do not change when more are requested. Genes
        without annotations stay in their zero-count bin. IC and the exact
        within-bin baseline are invariant under these permutations.
        """
        key = (int(n_permutations), int(seed))
        if getattr(self, "_permutation_key", None) == key:
            return self._permutation_maps
        rng = np.random.default_rng(seed)
        maps, groups = [], []
        for matrix in (self.Xm, self.Xh):
            bins = _burden_bins(matrix)
            identity = np.arange(matrix.shape[0], dtype=np.int32)
            maps.append(np.tile(identity, (n_permutations, 1)))
            groups.append(
                [np.flatnonzero(bins == b) for b in np.unique(bins) if b > 0]
            )
        for k in range(n_permutations):
            for mapping, memberships in zip(maps, groups):
                for members in memberships:
                    if len(members) > 1:
                        mapping[k, members] = rng.permutation(members)
        self._permutation_key, self._permutation_maps = key, tuple(maps)
        total = self.Xm.shape[0] * self.Xh.shape[0]
        dtype = np.int32 if total <= np.iinfo(np.int32).max else np.intp
        self._permuted_row_offsets = maps[0].astype(dtype) * self.Xh.shape[0]
        return self._permutation_maps

    def _null_block_scores(
        self, rows, cols, allocated, mass, expected, n_permutations, seed
    ):
        """Return ``q_b`` under each cached profile shuffle.

        Only positive-mass annotated pairs are gathered, in batches of
        eight permutations and at most 262144 pairs, which bounds temporary
        storage to about 50 MiB whatever the plan size. Detection and GO
        loading are not repeated.
        """
        observed = np.zeros(n_permutations)
        if expected >= 1 - 1e-12:
            return observed
        key = (
            int(seed),
            float(mass),
            float(expected),
            rows.tobytes(),
            cols.tobytes(),
            hashlib.sha256(
                np.ascontiguousarray(allocated).view(np.uint8)
            ).digest(),
        )
        if not hasattr(self, "_null_score_cache"):
            self._null_score_cache = OrderedDict()
        cached = self._null_score_cache.get(key)
        if cached is not None and len(cached) >= n_permutations:
            self._null_score_cache.move_to_end(key)
            return cached[:n_permutations].copy()
        completed = 0 if cached is None else len(cached)
        edge_rows, edge_cols = np.nonzero(allocated)
        edges_m, edges_h = rows[edge_rows], cols[edge_cols]
        annotated = self.am[edges_m] & self.ah[edges_h]
        edges_m, edges_h = edges_m[annotated], edges_h[annotated]
        weights = allocated[edge_rows[annotated], edge_cols[annotated]] / mass
        del edge_rows, edge_cols, annotated
        if not len(weights):
            return observed
        perm_m, perm_h = self._annotation_permutations(n_permutations, seed)
        flat = self._similarity.ravel()
        offsets = self._permuted_row_offsets
        for first in range(
            completed, n_permutations, _NULL_BATCH_PERMUTATIONS
        ):
            batch = slice(
                first, min(first + _NULL_BATCH_PERMUTATIONS, n_permutations)
            )
            for start in range(0, len(weights), _NULL_BATCH_PAIRS):
                part = slice(start, start + _NULL_BATCH_PAIRS)
                addresses = perm_h[batch, edges_h[part]].astype(
                    offsets.dtype, copy=False
                )
                addresses += offsets[batch, edges_m[part]]
                values = flat[addresses]
                observed[batch] += np.einsum(
                    "rk,k->r", values, weights[part], optimize=False
                )
        np.clip(observed, 0, 1, out=observed)
        scores = np.maximum(0.0, observed - expected) / (1 - expected)
        np.clip(scores, 0, 1, out=scores)
        if cached is not None:
            scores[:completed] = cached
        self._null_score_cache[key] = scores
        self._null_score_cache.move_to_end(key)
        capacity = getattr(self, "_null_score_cache_capacity", 16)
        while len(self._null_score_cache) > capacity:
            self._null_score_cache.popitem(last=False)
        return scores.copy()

    def _shared_terms(self, rows, cols):
        """Top shared GO terms as a label string (descriptive only)."""
        freq_m = np.asarray(self.Xm[rows].mean(axis=0)).ravel()
        freq_h = np.asarray(self.Xh[cols].mean(axis=0)).ravel()
        importance = np.minimum(freq_m, freq_h) * self._information
        order = np.argsort(-importance, kind="stable")
        labels = []
        for k in order[:5]:
            if importance[k] <= 0:
                break
            term = self.terms[k]
            name = self.ontology.terms[term].get("name", [""])[0]
            labels.append(f"{term}: {name}")
        return "; ".join(labels)

    # -- scoring ------------------------------------------------------------
    @staticmethod
    def _unique_indices(blocks, shape):
        indices, seen = [], set()
        for block in blocks:
            rows, cols = GOBlockEvaluator._indices(block, shape)
            key = (tuple(sorted(rows)), tuple(sorted(cols)))
            if key not in seen:
                indices.append((rows, cols))
                seen.add(key)
        return indices

    @staticmethod
    def _coverage(array, indices, total):
        """Return ``(coverage, overlap_counts, single_block_cells)``."""
        counts = single_local = None
        if len(indices) == 1:
            single_local = array[np.ix_(*indices[0])]
            covered = float(single_local.sum())
        elif indices:
            # A cell's count never exceeds the number of distinct blocks.
            counts = np.zeros(
                array.shape, dtype=np.min_scalar_type(len(indices))
            )
            for rows, cols in indices:
                counts[np.ix_(rows, cols)] += 1
            covered = float(array[counts > 0].sum())
        else:
            covered = 0.0
        coverage = covered / total if total > 0 else 0.0
        return coverage, counts, single_local

    def _score_block(
        self,
        number,
        rows,
        cols,
        array,
        counts,
        single_local,
        n_permutations,
        seed,
    ):
        """Score one block; return its table record and null samples."""
        similarity, baseline = self._similarity, self._baseline
        ix = np.ix_(rows, cols)
        local = single_local if single_local is not None else array[ix]
        allocated = local if counts is None else local / counts[ix]
        mass = float(allocated.sum())
        observed = expected = quality = raw_quality = null_mean = np.nan
        null_scores = np.zeros(n_permutations)
        if mass > 0:
            # Normalise before multiplying to avoid underflow with very
            # small retained masses in unbalanced OT.
            weights = allocated / mass
            observed = float(np.sum(weights * similarity[ix]))
            expected = float(np.sum(weights * baseline[ix]))
            if expected < 1 - 1e-12:
                raw_quality = max(0.0, observed - expected) / (1 - expected)
            else:
                raw_quality = 0.0
            raw_quality = float(np.clip(raw_quality, 0, 1))
            null_scores = self._null_block_scores(
                rows, cols, allocated, mass, expected, n_permutations, seed
            )
            null_mean = float(null_scores.mean())
            quality = float(np.clip(raw_quality - null_mean, -1, 1))
        active_rows = rows[local.sum(axis=1) > 0]
        active_cols = cols[local.sum(axis=0) > 0]
        themes = ""
        if len(active_rows) and len(active_cols):
            themes = self._shared_terms(active_rows, active_cols)
        record = {
            "block": number,
            "mouse_genes": len(rows),
            "human_genes": len(cols),
            "block_mass": float(local.sum()),
            "allocated_mass": mass,
            "block_score": raw_quality,
            "corrected_block_score": quality,
            "null_mean_block_score": null_mean,
            "observed_similarity": observed,
            "baseline_similarity": expected,
            "shared_go_terms": themes,
        }
        return record, null_scores

    def _score_plan(
        self,
        plan,
        supplied,
        detector,
        detector_kwargs,
        aggregation,
        n_permutations,
        seed,
    ):
        """Score one plan; return its summary dict and details dict."""
        array = self._array(plan)
        total = float(array.sum())
        if not np.isfinite(total):
            raise ValueError(
                "Total plan mass overflows floating-point precision"
            )
        if supplied is not None:
            supplied = self._remap_blocks(supplied)
        blocks, message = self._find_blocks(
            array, supplied, detector, detector_kwargs
        )
        indices = self._unique_indices(blocks, array.shape)
        coverage, counts, single_local = self._coverage(array, indices, total)
        # Keep every block of this plan in the null cache: a fixed small LRU
        # would evict and recompute all draws when switching aggregation on
        # a plan with many blocks.
        self._null_score_cache_capacity = max(16, len(indices))
        cache = getattr(self, "_null_score_cache", {})
        while len(cache) > self._null_score_cache_capacity:
            cache.popitem(last=False)
        records, null_samples = [], []
        for number, (rows, cols) in enumerate(indices):
            record, null_scores = self._score_block(
                number,
                rows,
                cols,
                array,
                counts,
                single_local,
                n_permutations,
                seed,
            )
            records.append(record)
            null_samples.append(null_scores)
        table = pd.DataFrame.from_records(records)
        raw_score, table = _aggregate_block_scores(
            table, aggregation, coverage
        )
        score = float(
            np.clip(table["corrected_plan_score_contribution"].sum(), -1, 1)
        )
        null_plan_mean = mc_error = 0.0
        if len(table):
            weights = table["aggregation_weight"].to_numpy()
            null_plan_scores = np.clip(
                coverage * weights @ np.stack(null_samples), 0, 1
            )
            null_plan_mean = float(null_plan_scores.mean())
            mc_error = float(
                null_plan_scores.std(ddof=1) / np.sqrt(n_permutations)
            )
        if self.am.all() and self.ah.all():
            annotated = total
        else:
            annotated = float(array[np.ix_(self.am, self.ah)].sum())
        if total <= 0 or annotated <= 0:
            score = raw_score = null_plan_mean = mc_error = np.nan
            if not table.empty:
                table["plan_score_contribution"] = np.nan
                table["corrected_plan_score_contribution"] = np.nan
        summary = {
            "biological_score": score,
            "transported_mass": total,
            "Q": raw_score,
            "S": score,
            "raw_biological_score": raw_score,
            "null_mean_plan_score": null_plan_mean,
            "score_mc_standard_error": mc_error,
            "n_permutations": int(n_permutations),
            "seed": int(seed),
            "R": int(n_permutations),
            "block_mass_coverage": coverage,
            "annotation_mass_coverage": (
                annotated / total if total > 0 else np.nan
            ),
        }
        report = self.annotation_report
        table.attrs.update(
            summary,
            aggregation=aggregation,
            merged_mouse_labels=report["mouse_merged_labels"],
            merged_human_labels=report["human_merged_labels"],
            ambiguous_mouse_aliases=report["mouse_ambiguous_aliases"],
            ambiguous_human_aliases=report["human_ambiguous_aliases"],
            mass_based_mouse_alias_choices=report[
                "mouse_mass_based_alias_choices"
            ],
            mass_based_human_alias_choices=report[
                "human_mass_based_alias_choices"
            ],
            scoring_method="S: GO block support above a shuffled-profile "
            "reference",
        )
        details = {
            "summary": summary,
            "blocks": table,
            "diagnostics": {
                "n_blocks": len(table),
                "detector_message": message,
                "baseline": "independent annotation-profile permutations in "
                "log2 burden bins",
                "significance_test": False,
            },
        }
        return summary, details

    def compare(
        self,
        plans,
        blocks_by_plan=None,
        *,
        detector=None,
        detector_kwargs=None,
        aggregation="mass",
        R=20,
        seed=0,
    ):
        """Rank plans with one GO definition on a fixed gene universe.

        Parameters
        ----------
        plans : mapping of str to plan
            Candidate plans, all on the evaluator's original input axes.
        blocks_by_plan : mapping, optional
            Existing blocks per plan, in original-axis indices (supply
            this or ``detector``, not both). Duplicate identities are
            remapped to the merged gene memberships.
        detector : callable, optional
            ``detector(array, **detector_kwargs)`` returning blocks, run on
            each plan after duplicate genes are summed.
        aggregation : {"mass", "uniform"}, default "mass"
        R, seed : int
            Shuffle count (>= 2) and seed; the same seed gives common
            shuffles across candidates.

        Returns
        -------
        ranking : pandas.DataFrame
            Columns ``Q`` and ``S``, sorted by ``S``.
        details : dict
            Per plan: ``summary``, the block table and diagnostics. The
            block-table ``attrs`` carry the gene-label records using the
            same keys as :func:`score_ot_plan`.

        Notes
        -----
        ``S`` is the signed ``Q_real - mean(Q_shuffled)`` in [-1, 1].
        Ambiguous aliases follow the ``reference_plan`` given at
        construction; their mass is retained. The Monte Carlo standard
        error measures estimation precision, not biological uncertainty.
        """
        if aggregation not in AGGREGATIONS:
            raise ValueError("aggregation must be 'mass' or 'uniform'")
        _validate_permutations(R, seed)
        n_permutations = int(R)
        if blocks_by_plan is not None and detector is not None:
            raise ValueError("Supply blocks_by_plan OR detector, not both")
        if not plans:
            raise ValueError("plans must be a nonempty mapping")
        self._functional_similarity()
        summaries, details = {}, {}
        for name, plan in plans.items():
            supplied = None if blocks_by_plan is None else blocks_by_plan[name]
            summaries[name], details[name] = self._score_plan(
                plan,
                supplied,
                detector,
                detector_kwargs,
                aggregation,
                n_permutations,
                seed,
            )
        ranking = pd.DataFrame.from_dict(summaries, orient="index")
        ranking.index.name = "plan"
        ranking = ranking.sort_values(
            "biological_score",
            ascending=False,
            kind="stable",
            na_position="last",
        )
        self._record_selection(ranking)
        return (
            ranking[["raw_biological_score", "biological_score"]].rename(
                columns={"raw_biological_score": "Q", "biological_score": "S"}
            ),
            details,
        )

    def _record_selection(self, ranking):
        """Store the best plan(s) and per-plan diagnostics."""
        self.best_plan_name, self.tied_best_plans = None, []
        scores = ranking["biological_score"]
        if scores.notna().any() and scores.max() > 0:
            best = scores.max()
            self.tied_best_plans = ranking.index[scores == best].tolist()
            self.best_plan_name = self.tied_best_plans[0]
        self.plan_diagnostics = ranking.drop(columns="biological_score").copy()
        self.selection_diagnostics = {
            "best_plan_name": self.best_plan_name,
            "tied_best_plans": self.tied_best_plans,
            "status": (
                "supported_candidate"
                if self.best_plan_name
                else "no_supported_candidate"
            ),
        }

    def evaluate(
        self,
        plan,
        blocks=None,
        *,
        detector=None,
        detector_kwargs=None,
        aggregation="mass",
        R=20,
        seed=0,
    ):
        """Score one plan; same semantics as :meth:`compare`.

        Returns the plan's details dict with ``summary``, ``blocks`` and
        ``diagnostics``.
        """
        _, details = self.compare(
            {"plan": plan},
            None if blocks is None else {"plan": blocks},
            detector=detector,
            detector_kwargs=detector_kwargs,
            aggregation=aggregation,
            R=R,
            seed=seed,
        )
        return details["plan"]
