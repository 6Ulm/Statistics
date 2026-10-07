"""Score mouse-to-human OT blocks by continuous GO functional correspondence.

Dependencies: numpy, pandas, scipy, scikit-learn, threadpoolctl. No GO package required.
Scores use information-weighted GO overlap, a matched annotation baseline,
whole-plan block coverage, and a shuffled-profile correction. They are signed
excess-support scores, not correctness probabilities.
See GO_block_quality_README.md for formulas, assumptions, and interpretation.
"""
from __future__ import annotations

from collections import Counter, defaultdict, OrderedDict
import copy
from functools import lru_cache
import gzip
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from .extract_dense_block import (extract_dense_blocks, extract_dense_blocks_coclustered,
                                 extract_dense_blocks_adaptive, extract_dense_blocks_hard_brim,
                                 extract_dense_blocks_lpawb, _expand_brim_cores,
                                 _validate_overlap_eta)


def _scoring_file(name):
    for folder in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (folder, folder / "upload"):
            path = directory / name
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"Put {name} beside go_block_quality.py or in the current folder")


def _dataframe_array(frame):
    """Convert numerical transport once, including pandas sparse storage."""
    sparse_columns = [isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes]
    if any(sparse_columns):
        if all(sparse_columns) and all(dtype.fill_value is not pd.NA and dtype.fill_value == 0
                                       for dtype in frame.dtypes):
            return frame.sparse.to_coo().toarray().astype(float, copy=False)
        frame = frame.apply(lambda column: column.sparse.to_dense()
                            if isinstance(column.dtype, pd.SparseDtype) else column)
    return frame.to_numpy(dtype=float)


_GO_CACHE_FORMAT = 1  # bump when _Ontology/_Annotations parsing or layout changes


def _go_cache_file(files):
    """Pickle path for parsed GO sources, or None when disk caching is off.

    GO_BLOCK_QUALITY_CACHE sets the folder ('' or 'off' disables caching;
    default ~/.cache/go_block_quality). The key covers the format version and
    every input file's path, size and mtime, so edited files are re-parsed.
    """
    folder = os.environ.get("GO_BLOCK_QUALITY_CACHE")
    if folder is not None and folder.strip().lower() in {"", "off", "0", "none"}:
        return None
    folder = Path(folder).expanduser() if folder else Path.home() / ".cache" / "go_block_quality"
    key = hashlib.sha256(repr((_GO_CACHE_FORMAT, tuple(files))).encode()).hexdigest()[:32]
    return folder / f"go_sources_{key}.pkl"


@lru_cache(maxsize=2)
def _cached_go_sources(files):
    path = _go_cache_file(files)
    if path is not None and path.is_file():
        try:
            with open(path, "rb") as handle:
                loaded = pickle.load(handle)
            if (isinstance(loaded, tuple) and len(loaded) == 3 and isinstance(loaded[0], _Ontology)
                    and all(isinstance(x, _Annotations) for x in loaded[1:])):
                return loaded
        except Exception:  # corrupt/stale cache: fall through and re-parse
            pass
    ontology = _Ontology(files[0][0], "BP")
    sources = (ontology, _Annotations(files[1][0], ontology, "experimental", 10090),
               _Annotations(files[2][0], ontology, "experimental", 9606))
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False)
            try:
                with handle:
                    pickle.dump(sources, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(handle.name, path)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except Exception:  # read-only or full disk: caching is best-effort
            pass
    return sources


def _axis_mass_choices(genes, masses, annotations):
    """Choose each ambiguous alias by OTHER unambiguously resolved mass.

    Exclude mass carried by ambiguous labels to avoid circular assignments.
    Ties (including absent candidates with zero mass) use symbol order.
    """
    totals = defaultdict(float)
    for label, value in zip(genes, masses):
        if label in annotations.symbols:
            totals[label] += float(value)
        else:
            candidates = annotations.aliases.get(label, set())
            if len(candidates) == 1:
                totals[next(iter(candidates))] += float(value)
    return tuple(sorted((label, min(annotations.aliases[label],
                                    key=lambda g: (-totals[g], g)))
                        for label in set(genes)
                        if label not in annotations.symbols
                        and len(annotations.aliases.get(label, ())) > 1))


@lru_cache(maxsize=2)
def _cached_evaluator(mouse_genes, human_genes, files, mouse_choices, human_choices):
    return GOBlockEvaluator(files[0][0], files[1][0], files[2][0],
                            mouse_genes=mouse_genes, human_genes=human_genes,
                            _sources=_cached_go_sources(files),
                            _alias_choices=(dict(mouse_choices), dict(human_choices)))


def _aggregate_block_scores(blocks, aggregation, coverage=1.):
    """Return Q and a table with shared weights and Q/S contributions."""
    if aggregation not in {"mass", "uniform"}:
        raise ValueError("aggregation must be 'mass' or 'uniform'")
    columns = ["block", "mouse_genes", "human_genes", "block_mass", "allocated_mass", "block_score",
               "corrected_block_score", "null_mean_block_score",
               "observed_similarity", "baseline_similarity", "shared_go_terms",
               "aggregation_weight", "plan_score_contribution", "corrected_plan_score_contribution"]
    if blocks.empty:
        return 0., pd.DataFrame(columns=columns)
    table = blocks[[c for c in columns if c not in {"aggregation_weight", "plan_score_contribution", "corrected_plan_score_contribution"}]].copy()
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
    table["aggregation_weight"] = weights
    table["plan_score_contribution"] = coverage * weights * np.nan_to_num(qualities, nan=0.)
    table["corrected_plan_score_contribution"] = coverage * weights * np.nan_to_num(table["corrected_block_score"].to_numpy(), nan=0.)
    score = float(np.clip(table["plan_score_contribution"].sum(), 0, 1))
    return score, table[columns]


def _validate_permutations(n_permutations, seed):
    if (not isinstance(n_permutations, (int, np.integer))
            or isinstance(n_permutations, (bool, np.bool_)) or n_permutations < 2):
        raise ValueError("R must be an integer >= 2")
    if (not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")


def get_go_files(go_folder):
    paths = []
    for name in ("go-basic.obo", "MOUSE-mod.gaf.gz", "HUMAN-uniprot.gaf.gz"):
        path = _scoring_file(name) if go_folder is None else (Path(go_folder).expanduser() / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file does not exist: {path}")
        paths.append(path)
    go_files = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)

    return go_files


def score_ot_plan(plan, go_files, aggregation="mass", *, block_detection="coclustered",
                  eta=None, R=20, seed=0, nmf_backend='auto', nmf_device='cpu', nmf_dtype='float64'):
    """Return (Q, S, block_scores). The plan is the only data input.

    Q is the original GO score; S = Q - mean(Q_shuffled).
    aggregation='mass': Q=coverage*sum(A_b*q_b)/sum(A_b),
    S=coverage*sum(A_b*s_b)/sum(A_b).
    aggregation='uniform': Q=coverage*mean(q_b), S=coverage*mean(s_b).
    A_b allocates overlapping mass equally among blocks; coverage is the
    fraction of whole-plan mass inside the block union. q_b is continuous GO
    similarity above an annotation-burden-matched background, rescaled to [0,1].
    s_b = q_b - mean(q_b_shuffled). Whole GO profiles are independently
    permuted within log2 annotation-burden bins in each species. Plan, blocks,
    mass allocation, GO frequencies, and matched baseline remain fixed.
    Q and block_score=q_b are in [0,1]; S and corrected_block_score=s_b
    are in [-1,1]. Negative corrected scores are retained. Zero-mass blocks
    have undefined scores; zero total mass or no annotated-to-annotated
    transport gives Q=S=NaN. These two criteria are related, not independent
    evidence, and are not combined into another metric.

    Rows are mouse gene names; columns are human gene names. Uses the supplied
    go-basic.obo, MOUSE-mod.gaf.gz, HUMAN-uniprot.gaf.gz
    beside this module or in the current folder (also searches upload/).
    go_folder is an optional string/Path containing all three GO files.
    extract_dense_block.py is imported normally at module load time.
    block_detection='coclustered' keeps the existing block definition, with
    faster but equivalent mask processing. Optional 'adaptive' first detects
    separated dense components in the bipartite activity graph; it can recover
    small scattered blocks without spatial smoothing. Otherwise it falls back
    to coclustering. It uses no GO information and adds no tuning parameters.
    Optional 'hard_brim' ('brim' and 'modularity' are compatibility aliases) uses all
    positive transport weights and a marginal-
    preserving independent reference, with weighted BRIM updates and module
    merging (Barber 2007; Beckett 2016). It requires no fixed cluster count or
    activity threshold. It can separate modules linked by weak bridges; it
    detects excess mass, not necessarily filled rectangles. Two deterministic
    starts are fixed. 'soft_brim' expands those same frozen hard cores once
    using excess transport above the marginal-preserving reference. eta is
    REQUIRED for soft_brim, kl_nmf, and bayesian_nmf, and must be a finite
    number in (0,1]; omit it (or pass None) for other detectors. Supplying eta for another
    detector raises ValueError instead of silently ignoring it. For mouse
    gene i and original human core C_k, a_ik=sum_{j in C_k} P_ij
    -r_i*sum_{j in C_k} c_j/T, with full-plan marginals r/c and mass T.
    Keep hard memberships and propose k if a_ik>0 and
    a_ik>=eta*max_l a_il; apply the symmetric human rule. Eligibility uses
    ORIGINAL cores. Process candidates in descending normalized original
    excess (rounded to 14 decimal places for stable ordering; then axis,
    gene index, core). Accept only if CURRENTLY uncovered cells added to the
    CURRENT rectangle have positive excess over r_i*c_j/T. This accounts for
    new mouse-human corners and avoids rewarding already-covered cells.
    Each accepted move increases M=sum_covered(P_ij/T-r_i*c_j/T**2), within
    roundoff tolerance. This is an internal structural objective, not Q/S.
    One pass; rejected candidates are not revisited. Smaller eta proposes
    more memberships but final covers need not be nested. eta=1 can add ties.
    This is a binary overlapping core-expansion heuristic, not continuous
    fuzzy memberships or a published fuzzy-modularity optimizer. Hard cores
    are reused when only eta changes; its value is part of the expanded-
    membership cache key. Shared-cell mass allocation already avoids double
    counting overlaps. Optional 'lpawb' implements Beckett's LPAwb+ singleton
    start on the smaller species, alternating label updates with seeded
    random tie choices, positive-gain module merging, then label refinement
    repeatedly until no objective improvement. It adds no search repetitions
    or biological weight. seed controls both LPAwb+ tie choices and GO
    shuffles through independent random streams; R never changes detection.
    Experimental 'kl_nmf' jointly fits normalized transport with KL-NMF;
    'bayesian_nmf' adds half-normal ARD priors, a=5,b=2, using the rectangular
    Psorakis et al. (2011) MAP updates. The Bayesian fit rescales input so
    positive weights average 1; this is a continuous-weight adaptation.
    Both start from hard BRIM cores, use that core count as initial rank,
    and threshold per-gene component mass at eta times its strongest mass.
    seed also controls their small positive initialization; fits are shared
    across eta/aggregation/R, capped at 500 iterations with tolerance 1e-5.
    The table's nmf_fit attribute reports convergence and objective history.
    nmf_backend='auto' uses compiled sparse CPU updates when a C compiler is
    available, otherwise NumPy. Use 'numpy' for the original reference or
    'native' to require compiled updates. nmf_device='cuda' selects optional
    PyTorch GPU fitting; Apple MPS uses nmf_device='mps', nmf_dtype='float32'.
    nmf_backend='torch' also permits CPU execution for backend validation.
    Reduced precision can change memberships near eta; validate on your GPU.
    The native CPU path changes computation only, not
    rank, precision, objective, or stopping settings. See nmf_fit['backend'].
    These are local fits; fixed rank capacity can limit recovered modules.
    With remaining detectors seed controls GO shuffles only. Modularity has a
    resolution limit; hard_brim and lpawb assign each gene to at most one
    module. soft_brim cannot recover modules already merged by hard_brim. Existing
    coclustered/adaptive behavior is unchanged. No method is selected by GO.
    Changing this option can change detected blocks and therefore scores;
    use the same block_detection for all plans in a comparison.
    BP and experimental evidence are fixed for this convenience function.
    The GO evaluator is cached for repeated plans with the same gene axes.
    R (default 20, integer >=2) controls Monte Carlo precision and runtime,
    not a biological weighting. seed=0 gives reproducible common
    draws across plans with identical canonical axes. Increase R for
    close comparisons; the first draws are unchanged. Detection and GO loading
    are not rerun during shuffling. A cached similarity matrix is reindexed in
    bounded batches, using only transported annotated pairs. Repeated identical
    inputs reuse detected memberships and null samples; increasing R extends
    existing samples. The null cache holds all blocks of the current plan
    (minimum capacity 16), so changing aggregation does not recompute draws.
    Cache keys include plan/allocated content, so mutation
    does not reuse stale results. Small R estimates need more care near ties.
    Gene-label handling (automatic; input DataFrame is not modified):
    * Multiple labels resolving to one gene, or exact duplicate labels, are
      merged by SUMMING their rows/columns before block detection. Total
      transport mass is preserved; the gene counts once in the GO background.
      This replaces the former "Multiple input labels resolve to the same
      gene; deduplicate the plan axes" error.
    * An ambiguous alias, e.g. human TAZ -> TAFAZZIN or WWTR1, maps to the
      candidate with the largest total transport mass in the corresponding
      species' OTHER unambiguously resolved rows/columns of this plan. Do not
      use the alias's own mass to decide. Choose the alphabetically first
      candidate on a tie, including when both candidate masses are zero.
      Sum the alias's transport into the chosen gene if already present. Use
      only that gene's GO annotations. The choice is a mass-based heuristic,
      not proof of biological identity, and may differ between OT plans.
      A selected gene with no usable annotations adds no positive GO evidence.
    * Exact GAF symbols take precedence over synonyms. Unique aliases resolve
      directly; ambiguous ones use the mass rule above. Identity is species-
      specific and read before annotation-aspect/evidence filtering. Distinct
      unknown labels remain separate.

    Returned block_scores.attrs contains merged_mouse_labels,
    merged_human_labels, ambiguous_mouse_aliases, ambiguous_human_aliases,
    mass_based_mouse_alias_choices, mass_based_human_alias_choices,
    and annotation_mass_coverage for inspection. block_score is q_b;
    corrected_block_score is s_b; null_mean_block_score is the shuffled mean
    of q_b. plan_score_contribution sums to Q; corrected_plan_score_contribution
    sums to S. attrs also records Q, S, raw_biological_score
    (Q), null_mean_plan_score, score_mc_standard_error, n_permutations, and seed;
    R is also recorded. These are diagnostics, not extra ranking metrics or significance tests.
    If no transported pair has
    usable annotations, or total transport is zero, both plan scores are NaN.
    Missing files, negative/nonfinite transport, and invalid arguments still
    raise errors; they are not silently repaired.
    """
    if aggregation not in {"mass", "uniform"}:
        raise ValueError("aggregation must be 'mass' or 'uniform'")
    if nmf_backend not in {'auto','numpy','native','torch'}:
        raise ValueError("nmf_backend must be 'auto', 'numpy', 'native', or 'torch'")
    if (nmf_backend,nmf_device,nmf_dtype)!=('auto','cpu','float64') and block_detection not in {'kl_nmf','bayesian_nmf'}:
        raise ValueError('nmf_backend/device/dtype only apply to NMF detectors')
    if block_detection not in {"coclustered", "adaptive", "hard_brim", "soft_brim", "brim", "lpawb", "modularity", "kl_nmf", "bayesian_nmf"}:
        raise ValueError("block_detection must be 'adaptive', 'hard_brim', 'soft_brim', 'lpawb', 'coclustered', 'kl_nmf', or 'bayesian_nmf'; 'brim' and 'modularity' alias 'hard_brim'")
    if block_detection in {'soft_brim', 'kl_nmf', 'bayesian_nmf'}:
        eta = _validate_overlap_eta(eta)
    elif eta is not None:
        raise ValueError("eta is only accepted with block_detection='soft_brim', 'kl_nmf', or 'bayesian_nmf'; omit it for other detectors")
    _validate_permutations(R, seed)
    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a DataFrame with mouse row names and human column names")

    mouse_genes, human_genes = tuple(map(str, plan.index)), tuple(map(str, plan.columns))
    array = _dataframe_array(plan)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("OT plans must be finite and nonnegative; negative values are not silently clipped")
    # Marginal mass is the evidence for choosing an alias; label totals are
    # computed before any alias merging, then excluded for ambiguous labels.
    mouse_mass = array.sum(axis=1)
    human_mass = array.sum(axis=0)
    if not np.isfinite(mouse_mass).all() or not np.isfinite(human_mass).all():
        raise ValueError("Plan marginals must be finite")
    sources = _cached_go_sources(go_files)
    mouse_choices = _axis_mass_choices(mouse_genes, mouse_mass, sources[1])
    human_choices = _axis_mass_choices(human_genes, human_mass, sources[2])
    evaluator = _cached_evaluator(mouse_genes, human_genes, go_files,
                                  mouse_choices, human_choices)

    def detect(array):
        # Content-based cache is safe even when callers mutate their DataFrame.
        # Retain just one plan's gene memberships, not its large submatrices.
        method_name = "hard_brim" if block_detection in {"brim", "modularity"} else block_detection
        plan_key = (array.shape, hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).digest())
        cache_key = (method_name, eta if method_name in {"soft_brim", "kl_nmf", "bayesian_nmf"} else None,
                     int(seed) if method_name in {"lpawb", "kl_nmf", "bayesian_nmf"} else None, plan_key)
        if method_name in {'kl_nmf','bayesian_nmf'}:
            cache_key += (nmf_backend,nmf_device,nmf_dtype)
        cached = getattr(evaluator, "_last_detected", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        # Keep the supplied detector, include entries tied at its median,
        # and normalize only its input so tiny unbalanced masses are stable.
        if method_name in {"hard_brim", "soft_brim", "kl_nmf", "bayesian_nmf"}:
            cached_cores = getattr(evaluator, '_last_hard_brim_cores', None)
            if cached_cores is not None and cached_cores[0] == plan_key:
                cores = cached_cores[1]
            else:
                cores = extract_dense_blocks_hard_brim(array)
                cores = [dict(row_indices=b['row_indices'].copy(), col_indices=b['col_indices'].copy()) for b in cores]
                evaluator._last_hard_brim_cores = (plan_key, cores)
            if method_name in {'kl_nmf', 'bayesian_nmf'}:
                try:
                    from .nmf_blocks import fit_nmf, blocks_from_fit
                except ImportError:
                    from nmf_blocks import fit_nmf, blocks_from_fit
                cache = getattr(evaluator, '_nmf_fits', None)
                if cache is None or cache[0] != plan_key:
                    cache = (plan_key, {})
                    evaluator._nmf_fits = cache
                fit_key = (method_name, int(seed))
                if (nmf_backend,nmf_device,nmf_dtype)!=('auto','cpu','float64'):
                    fit_key += (nmf_backend,nmf_device,nmf_dtype)
                if fit_key not in cache[1]:
                    # Bound memory even if a caller tries many seeds.
                    if len(cache[1]) >= 2:
                        cache[1].pop(next(iter(cache[1])))
                    cache[1][fit_key] = fit_nmf(array, method=method_name, cores=cores, seed=seed,
                        backend=nmf_backend,device=nmf_device,dtype=nmf_dtype)
                found = blocks_from_fit(array, cache[1][fit_key], eta)
            else:
                found = cores if method_name == 'hard_brim' else _expand_brim_cores(array, cores, eta, include_blocks=False)
        elif method_name == "lpawb":
            found = extract_dense_blocks_lpawb(array, seed=seed)
        else:
            nonzero = array[array > 0]
            if not nonzero.size:
                return []
            scaled = array / nonzero.max()
            threshold = np.nextafter(float(np.median(scaled[scaled > 0])), -np.inf)
            active = scaled > threshold
            groups = min(10, int(active.any(axis=1).sum()), int(active.any(axis=0).sum()))
            if groups < 2:
                found = extract_dense_blocks(scaled, activity_threshold=threshold)
            else:
                method = extract_dense_blocks_adaptive if block_detection == "adaptive" else extract_dense_blocks_coclustered
                found = method(
                    scaled, n_clusters=groups, activity_threshold=threshold)
        if isinstance(found, tuple):
            found = found[0]
        unique, seen = [], set()
        for block in found:
            r, c = evaluator._indices(block, array.shape)
            key = (tuple(sorted(r)), tuple(sorted(c)))
            if key not in seen:
                unique.append({"row_indices": r.copy(), "col_indices": c.copy()})
                seen.add(key)
        evaluator._last_detected = (cache_key, unique)
        return unique

    result = evaluator.evaluate(array, detector=detect, aggregation=aggregation,
                                R=R, seed=seed)
    result["blocks"].attrs["block_detection"] = block_detection
    result["blocks"].attrs["eta"] = eta
    if block_detection in {'kl_nmf', 'bayesian_nmf'}:
        fitted = getattr(evaluator, '_nmf_fits', None)
        options=(nmf_backend,nmf_device,nmf_dtype)
        fit_key = (block_detection, int(seed)) + (() if options==('auto','cpu','float64') else options)
        if fitted is not None and fit_key in fitted[1]:
            result['blocks'].attrs['nmf_fit'] = dict(fitted[1][fit_key]['details'])
    return (float(result["summary"]["raw_biological_score"]),
            float(result["summary"]["biological_score"]), result["blocks"])

EXPERIMENTAL = frozenset({
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"
})
ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
ASPECTS = {"BP": ("biological_process", "P"),
           "MF": ("molecular_function", "F"),
           "CC": ("cellular_component", "C")}


# Term fields read by this module; everything else in the OBO is skipped.
_KEPT_FIELDS = frozenset({"id", "name", "namespace", "alt_id", "is_a", "relationship",
                          "is_obsolete", "replaced_by"})


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


class _Ontology:
    def __init__(self, path, aspect):
        self.namespace, self.gaf_aspect = ASPECTS[aspect]
        self.terms, self.aliases = {}, {}
        self.version = None
        stanza, fields = None, defaultdict(list)

        def finish():
            if stanza == "Term" and fields.get("id"):
                term = fields["id"][0]
                self.terms[term] = dict(fields)  # only _KEPT_FIELDS are collected
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
                        if key != "relationship" or value.startswith("part_of "):
                            fields[key].append(value)
            finish()

    def resolve(self, term):
        term = self.aliases.get(term, term)
        seen = set()
        while term in self.terms and term not in seen:
            seen.add(term)
            record = self.terms[term]
            if record.get("is_obsolete", ["false"])[0] != "true":
                return term if record.get("namespace", [None])[0] == self.namespace else None
            replacements = record.get("replaced_by", [])
            if len(replacements) != 1:
                return None
            term = self.aliases.get(replacements[0], replacements[0])
        return None

    @lru_cache(maxsize=None)
    def ancestors(self, term):
        # Only annotation-safe relations. Never treat regulates/has_part as is_a.
        result = {term}
        record = self.terms[term]
        parents = [x.split()[0] for x in record.get("is_a", [])]
        parents += [x.split()[1] for x in record.get("relationship", [])
                    if x.startswith("part_of ")]
        for parent in parents:
            parent = self.resolve(parent)
            if parent is not None:
                result.update(self.ancestors(parent))
        return frozenset(result - ROOTS)


class _Annotations:
    def __init__(self, path, ontology, evidence, taxon):
        if isinstance(evidence, str):
            if evidence not in {"experimental", "reviewed", "all"}:
                raise ValueError("evidence must be experimental, reviewed, all, or a set of codes")
            allowed = EXPERIMENTAL if evidence == "experimental" else None
        else:
            allowed = set(evidence)
        self.direct = defaultdict(set)
        self.aliases = defaultdict(set)
        self.symbols = set()
        self.ambiguous_aliases = {}
        self.mass_based_alias_choices = {}
        self.taxon = int(taxon)
        self.stats = Counter()
        self.headers = {}
        taxon_text = str(taxon)
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
                # Substring test is a cheap necessary condition for the exact
                # taxon-token match below.
                if taxon_text not in fields[12] or not any(
                        x.rsplit(":", 1)[-1] == taxon_text for x in fields[12].split("|")):
                    self.stats["wrong_taxon"] += 1
                    continue
                # Identity metadata is independent of annotation evidence.
                # A gene lacking experimental BP evidence must still prevent
                # its symbol/alias from being assigned to a different gene.
                symbol = fields[2]
                identity = (fields[0], fields[1], symbol, fields[10])
                if symbol and identity not in seen_identity:  # records repeat per gene
                    seen_identity.add(identity)
                    self.symbols.add(symbol)
                    aliases = {symbol, fields[1], fields[0] + ":" + fields[1]}
                    aliases.update(fields[10].split("|"))
                    for alias in aliases - {""}:
                        self.aliases[alias].add(symbol)
                if fields[8] != ontology.gaf_aspect:
                    continue
                if "NOT" in fields[3].split("|"):
                    self.stats["negative_excluded"] += 1
                    continue
                code = fields[6]
                if code == "ND" or (allowed is not None and code not in allowed) or (evidence == "reviewed" and code == "IEA"):
                    self.stats["evidence_excluded"] += 1
                    continue
                if fields[4] not in resolved_terms:
                    resolved_terms[fields[4]] = ontology.resolve(fields[4])
                term = resolved_terms[fields[4]]
                if term is None:
                    self.stats["unresolved_GO"] += 1
                    continue
                if term in ROOTS:
                    continue
                symbol = fields[2]
                self.direct[symbol].add(term)
                self.stats["accepted"] += 1

    def resolve_gene(self, label, choices=None):
        """Resolve a species-specific label using precomputed marginal choices.

        Exact symbols from ANY valid species-matching GAF record take
        precedence. A unique synonym/database identifier resolves to its
        gene, even when that gene lacks the selected annotation evidence.
        Unknown labels return None. For an ambiguous alias, use the selected
        candidate from `choices`, obtained by comparing total marginal masses
        of other unambiguously resolved labels. If no plan was supplied for
        that comparison, the alphabetically first candidate wins the zero-mass
        tie. Record candidates and chosen symbol separately. Never combine GO
        annotations of different candidate genes.
        """
        # Exact GAF symbols take precedence over aliases.
        label = str(label)
        if label in self.symbols:
            return label
        candidates = self.aliases.get(label, set())
        if len(candidates) > 1:
            candidates = sorted(candidates)
            self.ambiguous_aliases[label] = candidates
            selected = (choices or {}).get(label, candidates[0])
            if selected not in candidates:
                raise ValueError(f"Invalid candidate {selected!r} for ambiguous alias {label!r}")
            self.mass_based_alias_choices[label] = selected
            return selected
        return next(iter(candidates)) if candidates else None


class GOBlockEvaluator:
    """Load once and reuse for all plans on the same ordered gene universe.

    Default evidence is experimental, to reduce reuse of orthology evidence.
    Fixed gene axes define GO term frequencies and the annotation-matched
    baseline. Blocks receive continuous scores; enrichment is descriptive.
    Repeated identities are merged with transport summed before detection;
    ambiguous aliases use the candidate with the highest reference-plan mass.
    Input-axis resolution and merge/ambiguity records are in annotation_report.
    """
    def __init__(self, obo_path, mouse_gaf_path, human_gaf_path,
                 mouse_genes, human_genes, *, aspect="BP", evidence="experimental",
                 reference_plan=None, _sources=None, _alias_choices=None):
        """Load annotations and establish original-to-unique-gene mappings.

        Nonempty input axes may contain duplicate labels or different aliases
        of one gene; these no longer raise duplicate-identity errors. Each
        resolved or mass-assigned identity becomes one first-seen-order
        gene entry.
        Keep original axes in input_mouse_genes/input_human_genes for plan
        validation and canonical axes in mouse_genes/human_genes.

        If reference_plan is supplied, ambiguous aliases use the candidate
        with highest marginal mass among other unambiguously identified genes
        on that species axis. Ties use alphabetical order. For `.compare`, use
        a common reference_plan to keep choices fixed across candidates; the
        simple score_ot_plan function makes the choice separately for each
        supplied plan. Without reference_plan, all candidate masses are zero
        and ties resolve alphabetically. Candidate and choice diagnostics are
        in annotation_report; mass-based identity is only a heuristic.
        All labels assigned to the same identity are summed. Merges are
        stored under mouse_merged_labels/human_merged_labels. Actual transport
        is summed when a plan is supplied; no rows/columns are discarded.
        """
        if aspect not in ASPECTS:
            raise ValueError("aspect must be BP, MF, or CC")
        self.input_mouse_genes = tuple(map(str, mouse_genes))
        self.input_human_genes = tuple(map(str, human_genes))
        for genes in (self.input_mouse_genes, self.input_human_genes):
            if not genes:
                raise ValueError("Each species must have nonempty gene labels")
        self.aspect, self.evidence = aspect, evidence
        if _sources is None:
            self.ontology = _Ontology(obo_path, aspect)
            mouse = _Annotations(mouse_gaf_path, self.ontology, evidence, 10090)
            human = _Annotations(human_gaf_path, self.ontology, evidence, 9606)
        else:
            self.ontology, mouse, human = _sources
            if aspect != "BP" or evidence != "experimental":
                raise ValueError("Cached GO sources require BP and experimental evidence")
            # Each evaluator records only ambiguities observed on its own axes.
            mouse, human = copy.copy(mouse), copy.copy(human)
            mouse.ambiguous_aliases, human.ambiguous_aliases = {}, {}
            mouse.mass_based_alias_choices, human.mass_based_alias_choices = {}, {}
        if _alias_choices is not None:
            choices_by_species = _alias_choices
        elif reference_plan is not None:
            if isinstance(reference_plan, pd.DataFrame):
                if (tuple(map(str, reference_plan.index)) != self.input_mouse_genes or
                        tuple(map(str, reference_plan.columns)) != self.input_human_genes):
                    raise ValueError("reference_plan axes must match the evaluator's input axes")
                row_mass = np.asarray(reference_plan.sum(axis=1), dtype=float).ravel()
                col_mass = np.asarray(reference_plan.sum(axis=0), dtype=float).ravel()
            else:
                reference = reference_plan.toarray() if sparse.issparse(reference_plan) else np.asarray(reference_plan)
                if reference.shape != (len(self.input_mouse_genes), len(self.input_human_genes)):
                    raise ValueError("reference_plan shape must match the evaluator's input axes")
                row_mass, col_mass = reference.sum(axis=1), reference.sum(axis=0)
            choices_by_species = (dict(_axis_mass_choices(self.input_mouse_genes, row_mass, mouse)),
                                  dict(_axis_mass_choices(self.input_human_genes, col_mass, human)))
        else:
            choices_by_species = ({}, {})
        sets, degrees, missing, resolved = [], [], [], []
        canonical_axes, group_indices, merged_labels = [], [], []
        for genes, annotations, choices in ((self.input_mouse_genes, mouse, choices_by_species[0]),
                                            (self.input_human_genes, human, choices_by_species[1])):
            names = [annotations.resolve_gene(g, choices) for g in genes]
            # Merge assigned identities (including mass-based alias choices) or
            # identical unknown labels. Preserve first-seen gene order.
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
            canonical_axes.append(tuple(canonical))
            group_indices.append(np.asarray(groups, dtype=int))
            merged_labels.append({gene: labels for gene, labels in members.items() if len(labels) > 1})
            propagated, counts = [], []
            for name in canonical_names:
                direct = annotations.direct.get(name, set())
                propagated.append(set().union(*(self.ontology.ancestors(t) for t in direct)) if direct else set())
                counts.append(len(direct))
            sets.append(propagated)
            degrees.append(np.asarray(counts))
            missing.append([g for g, terms in zip(canonical, propagated) if not terms])
            resolved.append(names)
        self.mouse_genes, self.human_genes = canonical_axes
        self._row_groups, self._col_groups = group_indices
        self.terms = sorted(set().union(*sets[0], *sets[1]))
        lookup = {t: k for k, t in enumerate(self.terms)}
        matrices = []
        for gene_sets in sets:
            rows, cols = [], []
            for i, terms in enumerate(gene_sets):
                rows.extend([i] * len(terms))
                cols.extend(lookup[t] for t in terms)
            matrices.append(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                             shape=(len(gene_sets), len(self.terms))))
        self.Xm, self.Xh = matrices
        self.am = np.diff(self.Xm.indptr) > 0
        self.ah = np.diff(self.Xh.indptr) > 0
        self.annotation_report = {
            "aspect": aspect, "evidence": str(evidence), "obo_version": self.ontology.version,
            "mouse_gaf_headers": mouse.headers, "human_gaf_headers": human.headers,
            "mouse_parse_counts": dict(mouse.stats), "human_parse_counts": dict(human.stats),
            "mouse_annotated_genes": int(self.am.sum()), "human_annotated_genes": int(self.ah.sum()),
            "mouse_unannotated_labels": missing[0], "human_unannotated_labels": missing[1],
            "mouse_resolved_symbols": resolved[0], "human_resolved_symbols": resolved[1],
            "mouse_merged_labels": merged_labels[0], "human_merged_labels": merged_labels[1],
            "mouse_ambiguous_aliases": dict(mouse.ambiguous_aliases),
            "human_ambiguous_aliases": dict(human.ambiguous_aliases),
            "mouse_mass_based_alias_choices": dict(mouse.mass_based_alias_choices),
            "human_mass_based_alias_choices": dict(human.mass_based_alias_choices),
            "terms": len(self.terms),
        }
        if not self.am.any() or not self.ah.any():
            warnings.warn("One species has no usable annotations: biological scores will be NaN", stacklevel=2)
        if mouse.stats["unresolved_GO"] or human.stats["unresolved_GO"]:
            warnings.warn("Some GAF GO IDs could not be resolved in this OBO; inspect annotation_report", stacklevel=2)

    def _array(self, plan):
        """Validate the original plan, then sum assigned duplicate identities.

        DataFrame axes must match the ORIGINAL ordered input labels; duplicate
        labels are allowed. Reject wrong shape, negative or nonfinite values.
        Sum rows/columns using the mappings built in __init__, preserving total
        mass without mutating the input. Ambiguous aliases follow their
        reference-plan mass assignments; unknown labels remain unannotated, with identical
        repeated unknown labels summed.
        """
        if isinstance(plan, pd.DataFrame):
            if tuple(map(str, plan.index)) != self.input_mouse_genes or tuple(map(str, plan.columns)) != self.input_human_genes:
                raise ValueError("Plan axes must exactly match the evaluator's ordered gene universe. Reindex explicitly BEFORE detecting blocks.")
            array = _dataframe_array(plan)
        elif sparse.issparse(plan):
            array = plan.toarray().astype(float, copy=False)
        elif hasattr(plan, "detach"):
            array = plan.detach().cpu().numpy().astype(float, copy=False)
        else:
            array = np.asarray(plan, dtype=float)
        if array.shape != (len(self.input_mouse_genes), len(self.input_human_genes)):
            raise ValueError("Plan shape differs from the evaluator gene universe")
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("OT plans must be finite and nonnegative; negative values are not silently clipped")
        if len(self.mouse_genes) != len(self.input_mouse_genes):
            merged = np.zeros((len(self.mouse_genes), array.shape[1]), dtype=float)
            np.add.at(merged, self._row_groups, array)
            array = merged
        if len(self.human_genes) != len(self.input_human_genes):
            merged_transpose = np.zeros((len(self.human_genes), array.shape[0]), dtype=float)
            np.add.at(merged_transpose, self._col_groups, array.T)
            array = merged_transpose.T
        if not np.isfinite(array).all():
            raise ValueError("Merged transport mass overflows floating-point precision")
        return array

    def _remap_blocks(self, blocks):
        """Project supplied original-axis gene memberships to merged axes."""
        shape = (len(self.input_mouse_genes), len(self.input_human_genes))
        result = []
        for block in blocks:
            r, c = self._indices(block, shape)
            result.append({"row_indices": np.unique(self._row_groups[r]),
                           "col_indices": np.unique(self._col_groups[c])})
        return result

    @staticmethod
    def _indices(block, shape):
        if "row_indices" in block and "col_indices" in block:
            r, c = block["row_indices"], block["col_indices"]
        elif "bbox" in block:
            r0, r1, c0, c1 = block["bbox"]
            if any(not isinstance(v, (int, np.integer)) for v in (r0, r1, c0, c1)) or not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
                raise ValueError("bbox must contain valid exclusive-end integer bounds in ORIGINAL plan order")
            r, c = np.arange(r0, r1), np.arange(c0, c1)
        else:
            raise ValueError("Block needs original row_indices/col_indices or bbox; bbox_reordered alone is unsafe")
        output = []
        for values, limit in ((r, shape[0]), (c, shape[1])):
            values = np.asarray(values)
            if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
                raise ValueError("Block indices must be nonempty one-dimensional integer arrays")
            if (values < 0).any() or (values >= limit).any() or len(np.unique(values)) != len(values):
                raise ValueError("Block indices are duplicated or out of bounds")
            output.append(values.astype(int))
        return output

    def _find_blocks(self, array, blocks, detector, detector_kwargs):
        if blocks is not None:
            if detector is not None:
                raise ValueError("Supply blocks OR detector, not both")
            return list(blocks), None
        if detector is None:
            raise ValueError("Supply existing blocks or an explicit detector callable")
        try:
            found = detector(array, **(detector_kwargs or {}))
            found = found[0] if isinstance(found, tuple) else found
            return list(found), None
        except ValueError as error:
            if str(error).startswith(("No dense blocks found.", "All candidate components were smaller", "No component passed `min_density_ratio`.")):
                return [], str(error)
            raise

    @staticmethod
    def _burden_groups(matrix):
        """Fixed log2 bins of propagated annotation counts; zero has its own bin."""
        counts = np.diff(matrix.indptr)
        groups = np.zeros(len(counts), dtype=int)
        positive = counts > 0
        groups[positive] = 1 + np.floor(np.log2(counts[positive])).astype(int)
        return groups

    def _functional_similarity(self):
        if hasattr(self, "_similarity"):
            return self._similarity, self._baseline
        nm, nh = int(self.am.sum()), int(self.ah.sum())
        if not nm or not nh:
            self._similarity = np.zeros((len(self.am), len(self.ah)))
            self._baseline = self._similarity.copy()
            self._information = np.zeros(len(self.terms))
            return self._similarity, self._baseline
        # Equal species weighting prevents a larger universe dominating IC.
        fm = np.asarray(self.Xm.sum(axis=0)).ravel() / nm
        fh = np.asarray(self.Xh.sum(axis=0)).ravel() / nh
        frequency = (fm + fh) / 2
        information = -np.log(np.maximum(frequency, np.finfo(float).tiny))
        weighted_m = self.Xm.multiply(information).tocsr()
        intersection = (weighted_m @ self.Xh.T).toarray()
        wm = np.asarray(weighted_m.sum(axis=1)).ravel()
        wh = np.asarray(self.Xh.multiply(information).sum(axis=1)).ravel()
        union = wm[:, None] + wh[None, :] - intersection
        valid_union = union > np.finfo(float).eps
        # Reuse the dense intersection storage instead of allocating another
        # complete mouse x human matrix during evaluator construction.
        similarity = np.divide(intersection, union, out=intersection, where=valid_union)
        similarity[~valid_union] = 0
        np.clip(similarity, 0, 1, out=similarity)
        # Exact expected similarity under independent within-bin permutations
        # of whole annotation profiles, with the plan and blocks held fixed.
        gm, gh = self._burden_groups(self.Xm), self._burden_groups(self.Xh)
        baseline = np.zeros_like(similarity)
        for a in np.unique(gm):
            rows = np.flatnonzero(gm == a)
            for b in np.unique(gh):
                cols = np.flatnonzero(gh == b)
                selection = np.ix_(rows, cols)
                baseline[selection] = similarity[selection].mean()
        self._similarity, self._baseline = similarity, baseline
        self._information = information
        return similarity, baseline

    def _annotation_permutations(self, n_permutations, seed):
        """Common whole-profile shuffle maps within each species' burden bins.

        The first R maps do not change when requesting more permutations.
        Missing annotations stay in their zero-count bin. IC and the exact
        within-bin baseline are invariant under these profile permutations.
        """
        key = (int(n_permutations), int(seed))
        if getattr(self, "_permutation_key", None) == key:
            return self._permutation_maps
        rng = np.random.default_rng(seed)
        maps, groups = [], []
        for matrix in (self.Xm, self.Xh):
            bins = self._burden_groups(matrix)
            maps.append(np.tile(np.arange(matrix.shape[0], dtype=np.int32), (n_permutations, 1)))
            groups.append([np.flatnonzero(bins == b) for b in np.unique(bins) if b > 0])
        for k in range(n_permutations):
            for mapping, memberships in zip(maps, groups):
                for members in memberships:
                    if len(members) > 1:
                        mapping[k, members] = rng.permutation(members)
        self._permutation_key, self._permutation_maps = key, tuple(maps)
        dtype = np.int32 if self.Xm.shape[0] * self.Xh.shape[0] <= np.iinfo(np.int32).max else np.intp
        self._permuted_row_offsets = maps[0].astype(dtype) * self.Xh.shape[0]
        return self._permutation_maps

    def _null_block_scores(self, rows, cols, allocated, mass, expected,
                           n_permutations, seed):
        """Evaluate q_b under cached profile maps; no detector/GO recomputation.

        Gather only positive-mass annotated pairs. Eight permutations and at
        most 262144 pairs per batch bound temporary gather storage to about
        50 MiB regardless of full plan size or number of permutations.
        """
        observed = np.zeros(n_permutations)
        if expected >= 1 - 1e-12:
            return observed
        key = (int(seed), float(mass), float(expected),
               rows.tobytes(), cols.tobytes(),
               hashlib.sha256(np.ascontiguousarray(allocated).view(np.uint8)).digest())
        if not hasattr(self, "_null_score_cache"):
            self._null_score_cache = OrderedDict()
        cached = self._null_score_cache.get(key)
        if cached is not None and len(cached) >= n_permutations:
            self._null_score_cache.move_to_end(key)
            return cached[:n_permutations].copy()
        completed = 0 if cached is None else len(cached)
        rr, cc = np.nonzero(allocated)
        edges_m, edges_h = rows[rr], cols[cc]
        annotated = self.am[edges_m] & self.ah[edges_h]
        edges_m, edges_h = edges_m[annotated], edges_h[annotated]
        weights = allocated[rr[annotated], cc[annotated]] / mass
        del rr, cc, annotated
        if not len(weights):
            return observed
        pm, ph = self._annotation_permutations(n_permutations, seed)
        flat = self._similarity.ravel()
        for first in range(completed, n_permutations, 8):
            batch = slice(first, min(first + 8, n_permutations))
            for start in range(0, len(weights), 262144):
                part = slice(start, start + 262144)
                addresses = ph[batch, edges_h[part]].astype(self._permuted_row_offsets.dtype, copy=False)
                addresses += self._permuted_row_offsets[batch, edges_m[part]]
                values = flat[addresses]
                observed[batch] += np.einsum("rk,k->r", values, weights[part], optimize=False)
        np.clip(observed, 0, 1, out=observed)
        scores = np.maximum(0., observed - expected) / (1 - expected)
        np.clip(scores, 0, 1, out=scores)
        if cached is not None:
            scores[:completed] = cached
        self._null_score_cache[key] = scores
        self._null_score_cache.move_to_end(key)
        while len(self._null_score_cache) > getattr(self, '_null_score_cache_capacity', 16):
            self._null_score_cache.popitem(last=False)
        return scores.copy()

    def _shared_terms(self, rows, cols):
        # Descriptive labels only; no significance claim or scoring threshold.
        fm = np.asarray(self.Xm[rows].mean(axis=0)).ravel()
        fh = np.asarray(self.Xh[cols].mean(axis=0)).ravel()
        importance = np.minimum(fm, fh) * self._information
        order = np.argsort(-importance, kind="stable")
        labels = []
        for k in order[:5]:
            if importance[k] <= 0:
                break
            t = self.terms[k]
            labels.append(f"{t}: {self.ontology.terms[t].get('name', [''])[0]}")
        return "; ".join(labels)

    def compare(self, plans, blocks_by_plan=None, *, detector=None,
                detector_kwargs=None, aggregation="mass", R=20,
                seed=0):
        """Rank plans with one common GO definition and fixed gene universe.

        Returns Q and S columns (sorted by S) and per-plan details. All candidates use
        the same similarity and matched background; adding plans cannot alter
        an existing score. Enrichment significance does not determine scores.
        All plans must match the original input axes. Duplicate identities are
        summed before detection; supplied original-axis blocks are remapped to
        merged gene memberships. Ambiguous aliases use the reference_plan
        choices set at construction; their mass is retained. Merge,
        ambiguity, and mass-based choice records are also returned in each
        block DataFrame's attrs, using the public score_ot_plan key names.
        biological_score is signed S=Q_real-mean(Q_shuffled), in [-1,1].
        Same seed and axes give common shuffles across candidates.
        corrected_block_score values aggregate to S; block_score values
        aggregate to Q. Both are returned as separate criteria.
        The Monte Carlo standard error measures estimation precision, not
        biological uncertainty or a significance test.
        """
        if aggregation not in {"mass", "uniform"}:
            raise ValueError("aggregation must be 'mass' or 'uniform'")
        _validate_permutations(R, seed)
        n_permutations = int(R)
        if blocks_by_plan is not None and detector is not None:
            raise ValueError("Supply blocks_by_plan OR detector, not both")
        if not plans:
            raise ValueError("plans must be a nonempty mapping")
        similarity, baseline = self._functional_similarity()
        summaries, details = {}, {}
        for name, plan in plans.items():
            array = self._array(plan)
            total = float(array.sum())
            if not np.isfinite(total):
                raise ValueError("Total plan mass overflows floating-point precision")
            supplied = None if blocks_by_plan is None else blocks_by_plan[name]
            if supplied is not None:
                supplied = self._remap_blocks(supplied)
            blocks, message = self._find_blocks(array, supplied, detector, detector_kwargs)
            indices, seen = [], set()
            for block in blocks:
                r, c = self._indices(block, array.shape)
                key = (tuple(sorted(r)), tuple(sorted(c)))
                if key not in seen:
                    indices.append((r, c))
                    seen.add(key)
            counts, single_local = None, None
            if len(indices) == 1:
                single_local = array[np.ix_(*indices[0])]
                covered = float(single_local.sum())
            elif indices:
                # The count never exceeds the number of distinct blocks.
                counts = np.zeros(array.shape, dtype=np.min_scalar_type(len(indices)))
                for r, c in indices:
                    counts[np.ix_(r, c)] += 1
                covered = float(array[counts > 0].sum())
            else:
                covered = 0.
            coverage = covered / total if total > 0 else 0.
            # Keep every block of this plan: a fixed 16-entry LRU caused
            # sequential mass -> uniform calls to evict/recompute all draws
            # whenever a plan had more than 16 blocks. Still bounded by one
            # current plan's block count (or the small standalone default).
            self._null_score_cache_capacity = max(16, len(indices))
            cache = getattr(self, '_null_score_cache', {})
            while len(cache) > self._null_score_cache_capacity:
                cache.popitem(last=False)
            records, null_samples = [], []
            for number, (r, c) in enumerate(indices):
                ix = np.ix_(r, c)
                local = single_local if single_local is not None else array[ix]
                allocated = local if counts is None else local / counts[ix]
                mass = float(allocated.sum())
                observed = expected = quality = raw_quality = null_mean = np.nan
                null_scores = np.zeros(n_permutations)
                if mass > 0:
                    # Normalize before multiplication to avoid underflow with
                    # very small retained masses in unbalanced OT.
                    weights = allocated / mass
                    observed = float(np.sum(weights * similarity[ix]))
                    expected = float(np.sum(weights * baseline[ix]))
                    raw_quality = (max(0., observed - expected) / (1 - expected)
                                   if expected < 1 - 1e-12 else 0.)
                    raw_quality = float(np.clip(raw_quality, 0, 1))
                    null_scores = self._null_block_scores(r, c, allocated, mass, expected,
                                                         n_permutations, seed)
                    null_mean = float(null_scores.mean())
                    quality = float(np.clip(raw_quality - null_mean, -1, 1))
                null_samples.append(null_scores)
                active_r = r[local.sum(axis=1) > 0]
                active_c = c[local.sum(axis=0) > 0]
                themes = self._shared_terms(active_r, active_c) if len(active_r) and len(active_c) else ""
                records.append({"block": number, "mouse_genes": len(r), "human_genes": len(c),
                                "block_mass": float(local.sum()), "allocated_mass": mass,
                                "block_score": raw_quality, "corrected_block_score": quality,
                                "null_mean_block_score": null_mean, "observed_similarity": observed,
                                "baseline_similarity": expected, "shared_go_terms": themes})
            table = pd.DataFrame.from_records(records)
            raw_score, table = _aggregate_block_scores(table, aggregation, coverage)
            score = float(np.clip(table["corrected_plan_score_contribution"].sum(), -1, 1))
            null_plan_mean = mc_error = 0.
            if len(table):
                aggregation_weights = table["aggregation_weight"].to_numpy()
                null_plan_scores = np.clip(coverage * aggregation_weights @ np.stack(null_samples), 0, 1)
                null_plan_mean = float(null_plan_scores.mean())
                mc_error = float(null_plan_scores.std(ddof=1) / np.sqrt(n_permutations))
            annotated = total if self.am.all() and self.ah.all() else float(array[np.ix_(self.am, self.ah)].sum())
            if total <= 0 or annotated <= 0:
                score = raw_score = null_plan_mean = mc_error = np.nan
                if not table.empty:
                    table["plan_score_contribution"] = np.nan
                    table["corrected_plan_score_contribution"] = np.nan
            summary = {"biological_score": score, "transported_mass": total,
                       "Q": raw_score, "S": score,
                       "raw_biological_score": raw_score, "null_mean_plan_score": null_plan_mean,
                       "score_mc_standard_error": mc_error,
                       "n_permutations": int(n_permutations), "seed": int(seed),
                       "R": int(n_permutations),
                       "block_mass_coverage": coverage,
                       "annotation_mass_coverage": annotated / total if total > 0 else np.nan}
            table.attrs.update(summary, aggregation=aggregation,
                               merged_mouse_labels=self.annotation_report["mouse_merged_labels"],
                               merged_human_labels=self.annotation_report["human_merged_labels"],
                               ambiguous_mouse_aliases=self.annotation_report["mouse_ambiguous_aliases"],
                               ambiguous_human_aliases=self.annotation_report["human_ambiguous_aliases"],
                               mass_based_mouse_alias_choices=self.annotation_report["mouse_mass_based_alias_choices"],
                               mass_based_human_alias_choices=self.annotation_report["human_mass_based_alias_choices"],
                               scoring_method="S: GO block support above a shuffled-profile reference")
            summaries[name] = summary
            details[name] = {"summary": summary, "blocks": table,
                             "diagnostics": {"n_blocks": len(table), "detector_message": message,
                                             "baseline": "independent annotation-profile permutations in log2 burden bins",
                                             "significance_test": False}}
        ranking = pd.DataFrame.from_dict(summaries, orient="index")
        ranking.index.name = "plan"
        ranking = ranking.sort_values("biological_score", ascending=False, kind="stable", na_position="last")
        self.best_plan_name, self.tied_best_plans = None, []
        if ranking["biological_score"].notna().any() and ranking["biological_score"].max() > 0:
            best = ranking["biological_score"].max()
            self.tied_best_plans = ranking.index[ranking["biological_score"] == best].tolist()
            self.best_plan_name = self.tied_best_plans[0]
        self.plan_diagnostics = ranking.drop(columns="biological_score").copy()
        self.selection_diagnostics = {"best_plan_name": self.best_plan_name,
                                      "tied_best_plans": self.tied_best_plans,
                                      "status": "supported_candidate" if self.best_plan_name else "no_supported_candidate"}
        return ranking[["raw_biological_score", "biological_score"]].rename(
            columns={"raw_biological_score": "Q", "biological_score": "S"}), details

    def evaluate(self, plan, blocks=None, *, detector=None, detector_kwargs=None,
                 aggregation="mass", R=20, seed=0):
        """Score one plan; same score semantics as compare and score_ot_plan."""
        _, details = self.compare({"plan": plan},
                                  None if blocks is None else {"plan": blocks},
                                  detector=detector, detector_kwargs=detector_kwargs,
                                  aggregation=aggregation, R=R,
                                  seed=seed)
        return details["plan"]
