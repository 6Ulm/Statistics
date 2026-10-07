"""
extract_dense_block.py

Detects and extracts one or more spatially-localized dense/high-magnitude
blocks from an otherwise sparse, noise-like 2D numpy array (e.g. the pattern
seen in a heatmap where most cells are near-zero/random speckle and one or
more regions stand out as coherent square/rectangular blocks of elevated
values).

Algorithm
---------
1. Convert the array to an "activity" signal (binary above-threshold mask,
   or raw magnitude).
2. Compute a local density map with a box filter.
3. Threshold the density map using a robust (median/MAD) statistic.
4. Take ALL connected components of the thresholded mask (not just the
   strongest), filter out ones that are too small or not dense enough.
5. Rank the surviving components by strength and return a bounding box +
   extracted sub-array for each one.
"""

import numpy as np
import heapq
from scipy import ndimage
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import SpectralCoclustering
from threadpoolctl import threadpool_limits


def _close_square(mask, size):
    """Exact binary closing with a square footprint, using linear filters.

    Even footprints need the dilation origin shifted by one, matching
    scipy.ndimage.binary_closing's reflected footprint convention.
    """
    origin = -1 if size % 2 == 0 else 0
    closed = ndimage.maximum_filter1d(
        mask, size, axis=0, mode="constant", cval=0, origin=origin
    )
    closed = ndimage.maximum_filter1d(
        closed, size, axis=1, mode="constant", cval=0, origin=origin
    )
    closed = ndimage.minimum_filter1d(
        closed, size, axis=0, mode="constant", cval=0
    )
    return ndimage.minimum_filter1d(
        closed, size, axis=1, mode="constant", cval=0
    )


def _fill_holes(mask):
    """Fill holes by labeling background connected to any image boundary.

    Uses the same four-neighbour connectivity as binary_fill_holes with its
    default structure, without propagating a flood across the full image.
    """
    if not mask.size:
        return mask.copy()
    background, count = ndimage.label(~mask)
    exterior = np.zeros(count + 1, dtype=bool)
    for border in (
        background[0],
        background[-1],
        background[:, 0],
        background[:, -1],
    ):
        exterior[border] = True
    exterior[0] = False  # Label zero denotes original foreground.
    return ~exterior[background]


def extract_dense_blocks(
    A,
    activity_threshold=None,
    window_size=15,
    mad_multiplier=6.0,
    min_component_size=4,
    use_binary_activity=True,
    closing_size=5,
    fill_holes=True,
    max_blocks=None,
    min_density_ratio=None,
    return_details=False,
):
    """
    Detect and extract all densest contiguous rectangular blocks in a 2D array.

    Generalization of a single-block detector: instead of keeping only the
    strongest connected component after thresholding the local density map,
    this keeps every component that survives filtering, ranks them by
    strength, and returns one bounding box + sub-array per detected block.

    Parameters
    ----------
    A : np.ndarray, shape (n_rows, n_cols)
        The raw 2D data matrix underlying the heatmap (real numeric values,
        NOT the rendered RGB image). Sign is ignored (magnitude is what
        matters). Must be 2-dimensional.

    activity_threshold : float or None, default None
        Value above which a cell in `A` is considered "active". Only used
        when `use_binary_activity=True`.
        - None (default): auto-set to the 50th percentile of the nonzero
          absolute values of `A`.
        - Pass an explicit value if your data has no exact zeros, or if the
          auto threshold is picking up too much/too little background.

    window_size : int, default 15
        Side length (in cells) of the square box filter used to build the
        local density map (`scipy.ndimage.uniform_filter`).
        - Should be smaller than the smallest block you expect to detect,
          and larger than the typical gap between background noise
          speckles.
        - If your true regions differ a lot in size, a single `window_size`
          may under- or over-smooth some of them; consider running this
          function multiple times with different values and merging /
          deduplicating the results (see Notes).

    mad_multiplier : float, default 6.0
        Threshold on the local density map, expressed in robust
        median-absolute-deviation units above the median:
            density > median(density) + mad_multiplier * MAD(density)
        - Lower it to catch more/weaker regions; raise it to keep only the
          most pronounced ones. With several true regions of different
          intensity, a lower value is usually needed to catch the weakest
          one, at the cost of possibly fragmenting noise into extra
          spurious components (use `min_component_size` and
          `min_density_ratio` to filter those out).

    min_component_size : int, default 4
        Minimum number of cells (in the thresholded density mask) for a
        connected component to be considered a candidate block at all.
        Components smaller than this are dropped before ranking.

    use_binary_activity : bool, default True
        - True: binarize `A` (`|A| > activity_threshold`) before computing
          density -- density becomes "fraction of active cells nearby".
          Best for sparse matrices.
        - False: use `|A|` directly -- density becomes "local mean
          magnitude". Best when regions are dense everywhere but differ in
          value size rather than in how many nonzero cells they have.

    closing_size : int, default 5
        Side length of the structuring element used to morphologically
        *close* the thresholded mask (`scipy.ndimage.binary_closing`) before
        labeling connected components. This addresses the case where noise
        exists *inside* a true region: a local dip in density (e.g. a patch
        of noise that happens to look like background) can locally drop
        below the density threshold and cut a hole or a full-width gap
        through an otherwise dense block, splitting one true region into
        several smaller detected components (or making its bounding box too
        small/wrong). Closing dilates the mask slightly and then erodes it
        back, which bridges gaps up to roughly `closing_size` cells wide
        without growing the mask's overall footprint.
        - Set higher if your true regions can have wide internal noise
          dropouts; set lower (or 0 to disable) if separate true regions can
          sit close together, since too much closing can incorrectly fuse
          two distinct nearby regions into one.
        - This only reshapes the *mask* used to decide block boundaries. It
          never modifies `A` itself -- the returned sub-arrays always
          contain the original, untouched values (including whatever noise
          was inside the region to begin with).

    fill_holes : bool, default True
        If True, fill any fully-enclosed holes in the mask
        (`scipy.ndimage.binary_fill_holes`) after closing, so that a small
        island of noise sitting inside an otherwise solid block doesn't
        leave a gap in the detected region. Like `closing_size`, this only
        affects the boundary-detection mask, never the values returned in
        `block`.

    max_blocks : int or None, default None
        Maximum number of blocks to return.
        - None (default): return every component that survives
          `min_component_size` (and `min_density_ratio`, if given), most
          significant first.
        - An integer keeps only the top-`max_blocks` strongest components
          (ranked by total density, see Returns). Use this if you know
          roughly how many true regions to expect and want to ignore
          weaker false positives.

    min_density_ratio : float or None, default None
        Optional extra filter applied after bounding boxes are computed.
        For each candidate block, its density ratio is
        (mean |A| inside its bounding box) / (mean |A| outside it). If
        `min_density_ratio` is set, any candidate whose ratio falls below
        it is discarded, even if it passed the earlier density-map
        threshold. Use this to prune components that are technically local
        maxima but aren't meaningfully denser than the background -- e.g.
        `min_density_ratio=2.0` keeps only blocks at least twice as dense
        as the rest of the matrix.

    return_details : bool, default False
        If True, also return a dict of full intermediate arrays
        (`density_map`, `mask`, `labels`) shared across all detected
        blocks, useful for visualizing/debugging.

    Returns
    -------
    blocks : list of dict, sorted by strength (strongest first), each with:
        - "bbox"          : (r0, r1, c0, c1) -- row/col bounding box,
                             usable as `A[r0:r1, c0:c1]` (r1/c1 exclusive).
        - "block"         : np.ndarray, the extracted sub-array.
        - "size"          : int, number of mask cells in the underlying
                             connected component (not the bbox area).
        - "total_density" : float, sum of the density map over the
                             component -- the ranking criterion.
        - "density_ratio" : float, mean(|A| inside bbox) / mean(|A| outside
                             bbox). Values well above 1 (e.g. 3+) confirm a
                             genuinely denser region.

    details : dict, only if `return_details=True`, with keys "density_map",
        "mask", "labels" (shared across all blocks; cross-reference each
        block's cells in "labels" to see which component is which).

    Raises
    ------
    ValueError
        If `A` is not 2-dimensional, or if no component survives all
        filtering (try lowering `mad_multiplier`, `min_component_size`, or
        `min_density_ratio`).

    Notes
    -----
    - Each true region becomes its own connected component as long as they
      are spatially separated by lower-density background in the density
      map; regions that are close together or touching may merge into a
      single detected component. If that happens, lower `window_size` to
      reduce blurring between them.
    - Bounding boxes are the tightest axis-aligned rectangle around each
      component's cells. If a component is non-rectangular (e.g. L-shaped,
      or two nearby blobs joined by a thin bridge), its bounding box may
      include some background cells or, rarely, slightly overlap a
      neighboring block's bounding box -- compare `size` to the bbox area
      (`(r1-r0)*(c1-c0)`) as a quick check of how "rectangular" a detection
      really is.
    - Each block's `density_ratio` is computed against everything else in
      `A`, *including other detected blocks*, so it slightly underestimates
      the true contrast against pure background when several strong blocks
      are present. This only affects the reported number, not detection.
    - Noise *inside* a true region (a patch that locally looks like
      background) is handled by `closing_size`/`fill_holes` at the mask
      level only, so that internal noise doesn't fragment or shrink the
      detected boundary. The returned `block` array is always an untouched
      slice of `A` -- no smoothing, thresholding, or denoising is ever
      applied to the values themselves, so any noise genuinely present
      inside the true region is preserved exactly as-is in the output.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = rng.random((300, 1000)) * 0.05
    >>> A[20:140, 10:230] += rng.random((120, 220)) * 0.8     # block 1
    >>> A[180:260, 700:950] += rng.random((80, 250)) * 0.6    # block 2
    >>> blocks = extract_dense_blocks(A, mad_multiplier=4.0)
    >>> [b["bbox"] for b in blocks]
    [(20, 140, 10, 230), (180, 260, 700, 950)]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"`A` must be 2-dimensional, got shape {A.shape}")

    magnitude = np.abs(A)

    if use_binary_activity:
        if activity_threshold is None:
            nonzero = magnitude[magnitude > 0]
            activity_threshold = (
                np.percentile(nonzero, 50) if nonzero.size else 0.0
            )
        activity = (magnitude > activity_threshold).astype(float)
    else:
        activity = magnitude

    density_map = ndimage.uniform_filter(activity, size=window_size)

    med = np.median(density_map)
    mad = np.median(np.abs(density_map - med)) + 1e-12
    mask = density_map > (med + mad_multiplier * mad)

    if closing_size and closing_size > 1:
        mask = _close_square(mask, closing_size)

    if fill_holes:
        mask = _fill_holes(mask)

    labels, n_components = ndimage.label(mask)
    if n_components == 0:
        raise ValueError(
            "No dense blocks found. Try lowering `mad_multiplier` and/or "
            "`window_size`."
        )

    component_ids = np.arange(1, n_components + 1)
    sizes = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
    totals = np.bincount(
        labels.ravel(), weights=density_map.ravel(), minlength=n_components + 1
    )[1:]
    bounds = ndimage.find_objects(labels)

    candidates = [
        (int(cid), float(total))
        for cid, size, total in zip(component_ids, sizes, totals)
        if size >= min_component_size
    ]
    if not candidates:
        raise ValueError(
            "All candidate components were smaller than `min_component_size`. "
            "Try lowering it, or lowering `mad_multiplier`/`window_size`."
        )

    candidates.sort(key=lambda x: x[1], reverse=True)

    blocks = []
    for cid, total in candidates:
        rs, cs = bounds[cid - 1]
        r0, r1, c0, c1 = rs.start, rs.stop, cs.start, cs.stop

        # Sum four disjoint exterior rectangles. Avoid a full-image mask and
        # copy for EVERY candidate, and avoid subtracting nearly equal totals.
        inside_mean = magnitude[r0:r1, c0:c1].mean()
        outside_size = magnitude.size - (r1 - r0) * (c1 - c0)
        outside_sum = (
            magnitude[:r0].sum()
            + magnitude[r1:].sum()
            + magnitude[r0:r1, :c0].sum()
            + magnitude[r0:r1, c1:].sum()
        )
        outside_mean = (
            outside_sum / outside_size + 1e-12 if outside_size else np.nan
        )
        density_ratio = float(inside_mean / outside_mean)

        # The sum order differs slightly; preserve the original decision for
        # ratios within floating-point roundoff of a requested filter cutoff.
        if min_density_ratio is not None and np.isclose(
            density_ratio, min_density_ratio, rtol=1e-12, atol=1e-15
        ):
            outside_mask = np.ones_like(magnitude, dtype=bool)
            outside_mask[r0:r1, c0:c1] = False
            density_ratio = float(
                inside_mean / (magnitude[outside_mask].mean() + 1e-12)
            )

        if min_density_ratio is not None and density_ratio < min_density_ratio:
            continue

        blocks.append(
            {
                "bbox": (r0, r1, c0, c1),
                "block": A[r0:r1, c0:c1],
                "size": int(sizes[cid - 1]),
                "total_density": total,
                "density_ratio": density_ratio,
            }
        )
        if max_blocks is not None and len(blocks) >= max_blocks:
            break

    if not blocks:
        raise ValueError(
            "No component passed `min_density_ratio`. Try lowering it."
        )

    if return_details:
        details = {"density_map": density_map, "mask": mask, "labels": labels}
        return blocks, details

    return blocks


def extract_dense_block(A, return_details=False, **kwargs):
    """
    Thin convenience wrapper around `extract_dense_blocks` that returns only
    the single strongest block. Kept for backward compatibility / the common
    case where you only expect one region. See `extract_dense_blocks` for
    full parameter documentation -- all its keyword arguments (aside from
    `max_blocks`, which is fixed to 1 here) are accepted via `**kwargs`.

    Returns
    -------
    bbox : (r0, r1, c0, c1)
    block : np.ndarray
    details : dict, only if `return_details=True` -- same as
        `extract_dense_blocks`'s details dict, plus a "density_ratio" key
        for this block.
    """
    kwargs["max_blocks"] = 1
    if return_details:
        blocks, details = extract_dense_blocks(
            A, return_details=True, **kwargs
        )
        details = dict(details)
        details["density_ratio"] = blocks[0]["density_ratio"]
        return blocks[0]["bbox"], blocks[0]["block"], details
    blocks = extract_dense_blocks(A, **kwargs)
    return blocks[0]["bbox"], blocks[0]["block"]


def _refine_order_within_groups(
    activity,
    group_labels,
    group_order,
    axis,
    metric="jaccard",
    method="average",
):
    """
    Within each group (as assigned by co-clustering), further order the
    rows (axis=0) or columns (axis=1) by their own similarity, so that
    members of the same cocluster aren't just lumped together arbitrarily
    but form a smooth local gradient too. Falls back to the co-clustering
    order untouched for any group too small to cluster meaningfully.
    """
    mat = activity if axis == 0 else activity.T
    order = np.empty(mat.shape[0], dtype=int)
    pos = 0
    # The many small matrix products in Jaccard ordering run much faster with
    # one BLAS thread. Restore the caller's thread settings afterward.
    with threadpool_limits(limits=1, user_api="blas"):
        for g in group_order:
            members = np.where(group_labels == g)[0]
            if members.size >= 3:
                sub_order = _cluster_leaf_order(
                    mat[members], axis=0, metric=metric, method=method
                )
                members = members[sub_order]
            order[pos : pos + members.size] = members
            pos += members.size
    return order


def _cluster_leaf_order(
    binary_matrix, axis, metric="jaccard", method="average"
):
    """
    Return a permutation of the rows (axis=0) or columns (axis=1) of a binary
    matrix that places similar rows/columns next to each other, using
    hierarchical clustering with optimal leaf ordering. Rows/columns that are
    entirely False carry no similarity signal to cluster on, so they are left
    in their original relative order and appended at the end.
    """
    mat = binary_matrix if axis == 0 else binary_matrix.T
    has_signal = mat.any(axis=1)
    signal_idx = np.where(has_signal)[0]
    empty_idx = np.where(~has_signal)[0]

    if signal_idx.size < 3:
        return np.concatenate([signal_idx, empty_idx])

    signal = mat[signal_idx]
    if (
        metric == "jaccard"
        and signal.dtype == np.bool_
        and signal.shape[1] < 2**24
    ):
        # Boolean intersections and cardinalities are exact in float32 up to
        # 2**24 features. BLAS computes all pair intersections in one call;
        # float64 division gives the same distances as scipy pdist(jaccard).
        values = signal.astype(np.float32)
        common = (values @ values.T).astype(np.float64)
        cardinality = values.sum(axis=1, dtype=np.float64)
        union = cardinality[:, None] + cardinality[None, :] - common
        full = np.divide(
            union - common, union, out=np.zeros_like(union), where=union != 0
        )
        distances = squareform(full, checks=False)
    else:
        distances = pdist(signal, metric=metric)
    distances = np.nan_to_num(distances, nan=1.0)
    Z = linkage(distances, method=method, optimal_ordering=True)
    order_within_signal = leaves_list(Z)
    return np.concatenate([signal_idx[order_within_signal], empty_idx])


def extract_dense_blocks_coclustered(
    A,
    n_clusters=10,
    row_labels=None,
    col_labels=None,
    activity_threshold=None,
    use_binary_activity=True,
    refine_within_cluster=True,
    cluster_metric="jaccard",
    cluster_method="average",
    window_size=15,
    mad_multiplier=6.0,
    min_component_size=4,
    closing_size=5,
    fill_holes=True,
    max_blocks=None,
    min_density_ratio=None,
    random_state=0,
    return_details=False,
):
    """
    Co-cluster rows and columns FIRST, so that dense blocks are made
    contiguous by construction, THEN run `extract_dense_blocks` on that
    reordered matrix to pin down each block's precise boundaries and filter
    out any coclusters that aren't actually dense.

    This is a stronger approach than ordering rows and columns independently
    by their own marginal similarity (which is a reasonable heuristic, but
    doesn't jointly optimize for row/column groupings that concentrate mass
    together -- it can leave many overlapping, redundant fragments). Spectral
    co-clustering (Dhillon 2001) instead treats the matrix as a bipartite
    graph and finds a joint row/column partition that directly maximizes
    within-cluster density via a normalized-cut spectral embedding -- the
    same principle behind, e.g., document/word co-clustering. After sorting
    rows and columns by their cocluster assignment, each cocluster occupies
    a contiguous rectangular region by construction; the density detector
    is then only needed to size that region precisely and to discard any
    cocluster that turned out to be a "leftover" group rather than a real
    dense block.

    Parameters
    ----------
    A : np.ndarray, shape (n_rows, n_cols)
        Same as `extract_dense_blocks`.

    n_clusters : int, default 10
        Number of row/column coclusters `SpectralCoclustering` looks for.
        This is the main knob to tune: too few and distinct dense groups get
        merged into one (or into the background); too many and a single
        real block gets needlessly split, or clusters become too small and
        noisy to be meaningful. There's no universal default -- start with
        a rough guess at how many distinct groups you expect (e.g. 5-15 for
        a few hundred to a few thousand rows/columns) and adjust based on
        the blocks you get back. Note this assumes a roughly diagonal
        correspondence (row-group i pairs mainly with column-group i); it
        will still surface off-diagonal dense blocks if they exist; a
        genuinely bipartite many-to-many structure may need a larger
        `n_clusters` to resolve.

    row_labels, col_labels : array-like or None, default None
        Optional names for each row/column (e.g. gene symbols, sample IDs).
        If given, each returned block includes the actual labels belonging
        to it, in coclustered order. If None, plain integer indices are
        used.

    activity_threshold : float or None, default None
        The value above which a cell counts as "active" -- computed once
        (auto: 50th percentile of nonzero |A|, if None) and reused for both
        building the co-clustering input and the density detection step.
        As with `extract_dense_blocks`, if your data mixes real signal with
        numerical noise across many orders of magnitude, set this
        explicitly above the noise floor; the automatic default will not
        distinguish the two.

    use_binary_activity : bool, default True
        If True (default), co-cluster on the binarized activity mask
        (`|A| > activity_threshold`) -- appropriate for sparse data where
        what matters is which cells are active, not their exact magnitude
        (e.g. a bipartite presence/absence graph). If False, co-cluster on
        `|A|` directly -- appropriate for dense/continuous data where the
        graded magnitude itself carries the block structure.

    refine_within_cluster : bool, default True
        After sorting rows/columns by cocluster assignment, further reorder
        the members of each cocluster by their own similarity (same
        leaf-ordering approach as marginal clustering, just applied inside
        each cocluster rather than globally). This tends to produce a
        cleaner internal gradient and tighter local density for the
        detector to lock onto; set False to skip it and use the raw
        co-clustering order within each group (faster, slightly rougher).

    cluster_metric, cluster_method :
        Passed to `scipy.spatial.distance.pdist` / `scipy.cluster.hierarchy.
        linkage` for the within-cluster refinement step only (ignored if
        `refine_within_cluster=False`). See `pdist`/`linkage` docs for
        options; "jaccard"/"average" are reasonable defaults for sparse
        binary activity patterns.

    window_size, mad_multiplier, min_component_size, closing_size,
    fill_holes, max_blocks, min_density_ratio :
        Passed straight through to `extract_dense_blocks` on the reordered
        matrix -- see its docstring for full documentation of each.

    random_state : int, default 0
        Passed to `SpectralCoclustering` for reproducibility.

    return_details : bool, default False
        If True, also return a dict with the row/column permutations, the
        raw cocluster assignments, and the reordered matrix itself.

    Returns
    -------
    blocks : list of dict, sorted by strength (strongest first), each with:
        - "row_indices", "col_indices" : np.ndarray of int, the original
          row/column indices belonging to this block (contiguous in
          coclustered order, not necessarily in `A`'s original order).
        - "row_labels", "col_labels" : corresponding entries of
          `row_labels`/`col_labels` (or the same integer indices if none
          were given).
        - "block" : np.ndarray, `A[np.ix_(row_indices, col_indices)]` --
          the actual submatrix, values exactly as they appear in `A`.
        - "bbox_reordered" : (r0, r1, c0, c1) in the reordered matrix.
        - "size", "total_density", "density_ratio" : same meaning as in
          `extract_dense_blocks`.

    details : dict, only if `return_details=True`, with keys "row_order",
        "col_order", "A_reordered", "row_cluster_labels", "col_cluster_labels".

    Raises
    ------
    ValueError
        If `A` is not 2-dimensional, or if `extract_dense_blocks` finds no
        block on the reordered matrix (try adjusting `n_clusters` or the
        density-detection parameters).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n_rows, n_cols = 150, 400
    >>> A = rng.random((n_rows, n_cols)) * 0.05
    >>> # scattered, not contiguous
    >>> row_block = rng.permutation(n_rows)[:40]
    >>> col_block = rng.permutation(n_cols)[:60]
    >>> A[np.ix_(row_block, col_block)] += rng.random((40, 60)) * 0.8
    >>> blocks = extract_dense_blocks_coclustered(
    ...     A, n_clusters=2, mad_multiplier=4.0)
    >>> sorted(blocks[0]["row_indices"]) == sorted(row_block)
    True
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"`A` must be 2-dimensional, got shape {A.shape}")
    n_rows, n_cols = A.shape

    row_labels_arr = (
        np.arange(n_rows) if row_labels is None else np.asarray(row_labels)
    )
    col_labels_arr = (
        np.arange(n_cols) if col_labels is None else np.asarray(col_labels)
    )

    magnitude = np.abs(A)
    if activity_threshold is None:
        nonzero = magnitude[magnitude > 0]
        activity_threshold = (
            np.percentile(nonzero, 50) if nonzero.size else 0.0
        )
    activity = magnitude > activity_threshold

    cocluster_input = (
        activity.astype(float) if use_binary_activity else magnitude
    )

    # SpectralCoclustering normalizes by each row's and column's sum, so an
    # all-zero row/column (common once a noise-aware activity_threshold is
    # set) causes a divide-by-zero. Cocluster only the active rows/columns,
    # then append the all-zero ones at the end of the ordering afterward --
    # they carry no similarity signal to assign to a cocluster anyway, and a
    # uniformly-empty region at the tail can't register as a dense block.
    active_row_idx = np.where(activity.any(axis=1))[0]
    empty_row_idx = np.where(~activity.any(axis=1))[0]
    active_col_idx = np.where(activity.any(axis=0))[0]
    empty_col_idx = np.where(~activity.any(axis=0))[0]

    if active_row_idx.size < n_clusters or active_col_idx.size < n_clusters:
        raise ValueError(
            f"Only {active_row_idx.size} active rows and "
            f"{active_col_idx.size} "
            f"active columns at this activity_threshold -- too few for "
            f"n_clusters={n_clusters}. Lower n_clusters or activity_threshold."
        )

    sub_input = cocluster_input[np.ix_(active_row_idx, active_col_idx)]
    model = SpectralCoclustering(
        n_clusters=n_clusters, random_state=random_state
    )
    model.fit(sub_input)

    row_cluster = np.full(n_rows, -1, dtype=int)
    row_cluster[active_row_idx] = model.row_labels_
    col_cluster = np.full(n_cols, -1, dtype=int)
    col_cluster[active_col_idx] = model.column_labels_

    row_group_order = list(np.unique(model.row_labels_)) + (
        [-1] if empty_row_idx.size else []
    )
    col_group_order = list(np.unique(model.column_labels_)) + (
        [-1] if empty_col_idx.size else []
    )

    if refine_within_cluster:
        row_order = _refine_order_within_groups(
            activity,
            row_cluster,
            row_group_order,
            axis=0,
            metric=cluster_metric,
            method=cluster_method,
        )
        col_order = _refine_order_within_groups(
            activity,
            col_cluster,
            col_group_order,
            axis=1,
            metric=cluster_metric,
            method=cluster_method,
        )
    else:
        row_sort_key = np.where(row_cluster == -1, n_clusters, row_cluster)
        col_sort_key = np.where(col_cluster == -1, n_clusters, col_cluster)
        row_order = np.argsort(row_sort_key, kind="stable")
        col_order = np.argsort(col_sort_key, kind="stable")

    A_reordered = A[np.ix_(row_order, col_order)]

    blocks_reordered = extract_dense_blocks(
        A_reordered,
        activity_threshold=activity_threshold,
        use_binary_activity=use_binary_activity,
        window_size=window_size,
        mad_multiplier=mad_multiplier,
        min_component_size=min_component_size,
        closing_size=closing_size,
        fill_holes=fill_holes,
        max_blocks=max_blocks,
        min_density_ratio=min_density_ratio,
        return_details=False,
    )

    blocks = []
    for b in blocks_reordered:
        r0, r1, c0, c1 = b["bbox"]
        row_idx = row_order[r0:r1]
        col_idx = col_order[c0:c1]
        blocks.append(
            {
                "bbox_reordered": (r0, r1, c0, c1),
                "row_indices": row_idx,
                "col_indices": col_idx,
                "row_labels": row_labels_arr[row_idx],
                "col_labels": col_labels_arr[col_idx],
                "block": A[np.ix_(row_idx, col_idx)],
                "size": b["size"],
                "total_density": b["total_density"],
                "density_ratio": b["density_ratio"],
            }
        )

    if return_details:
        details = {
            "row_order": row_order,
            "col_order": col_order,
            "A_reordered": A_reordered,
            "row_cluster_labels": row_cluster,
            "col_cluster_labels": col_cluster,
        }
        return blocks, details

    return blocks


def _grouped_modularity_scores(
    weights, own_strength, other_strength, other_labels
):
    """Grouped sums minus marginal correction; no full centered matrix."""
    groups, inverse = np.unique(other_labels, return_inverse=True)
    if len(groups) == 1:
        scores = np.asarray(weights.sum(axis=1)).reshape(-1, 1)
    elif len(groups) == len(other_labels):
        # Singleton starts need only column ordering. Avoid multiplying by
        # a sparse permutation matrix on the largest first sweep.
        if np.array_equal(inverse, np.arange(len(inverse))):
            scores = (
                weights.toarray()
                if sparse.issparse(weights)
                else weights.copy(order="K")
            )
        else:
            scores = weights[:, np.argsort(inverse)]
            if sparse.issparse(scores):
                scores = scores.toarray()
    else:
        indicator = sparse.csr_matrix(
            (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
            shape=(len(inverse), len(groups)),
        )
        scores = weights @ indicator
        if sparse.issparse(scores):
            scores = scores.toarray()
    scores -= (
        own_strength[:, None]
        * np.bincount(inverse, weights=other_strength, minlength=len(groups))[
            None, :
        ]
    )
    return groups, scores


def _brim_assign(
    weights, own_strength, other_strength, other_labels, current=None
):
    """Exact one-side maximization of weighted Barber modularity."""
    groups, scores = _grouped_modularity_scores(
        weights, own_strength, other_strength, other_labels
    )
    selected = np.argmax(scores, axis=1)
    if current is not None:
        previous = np.searchsorted(groups, current)
        present = previous < len(groups)
        present[present] &= groups[previous[present]] == current[present]
        positions = np.flatnonzero(present)
        # Keep existing memberships on numerical ties: avoid label oscillation.
        tolerance = 64 * np.finfo(float).eps * own_strength[positions]
        keep = (
            scores[positions, previous[positions]]
            >= scores[positions, selected[positions]] - tolerance
        )
        selected[positions[keep]] = previous[positions[keep]]
    return groups[selected], float(
        scores[np.arange(len(selected)), selected].sum()
    )


def _lpawb_assign(weights, own_strength, other_strength, other_labels, rng):
    """Beckett Eq. 2.5, uniformly selecting among numerical maximizers.

    A species can update together: its nodes have no edges to one another,
    so every decision depends solely on the other species' fixed labels.
    Random tie selection uses bounded batches, not Python gene-pair loops.
    """
    if sparse.issparse(weights):
        groups, inverse = np.unique(other_labels, return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
            shape=(len(inverse), len(groups)),
        )
        grouped = (weights @ indicator).tocsr()
        # Zero-edge group scores are strictly negative. Since a row's full
        # centered scores sum to zero, its maximizer is on a positive edge.
        # Use sparse segmented reductions, retaining dense treatment when
        # numerical-zero ties could include zero-edge candidates.
        if grouped.nnz < grouped.shape[0] * grouped.shape[1] / 4 and np.all(
            np.diff(grouped.indptr) > 0
        ):
            grouped.sort_indices()
            degree = np.diff(grouped.indptr)
            node = np.repeat(np.arange(len(own_strength)), degree)
            group_strength = np.bincount(
                inverse, weights=other_strength, minlength=len(groups)
            )
            values = (
                grouped.data
                - own_strength[node] * group_strength[grouped.indices]
            )
            best = np.maximum.reduceat(values, grouped.indptr[:-1])
            tolerance = 64 * np.finfo(float).eps * own_strength
            if np.all(best > tolerance):
                tied = values >= (best - tolerance)[node]
                counts = np.add.reduceat(
                    tied.astype(np.int32), grouped.indptr[:-1]
                )
                ranks = np.zeros(len(counts), dtype=int)
                positions = np.flatnonzero(counts > 1)
                ranks[positions] = (
                    rng.random(len(positions)) * counts[positions]
                ).astype(int)
                offsets = np.cumsum(counts) - counts
                selected = np.flatnonzero(tied)[offsets + ranks]
                return groups[grouped.indices[selected]], float(
                    values[selected].sum()
                )
    groups, scores = _grouped_modularity_scores(
        weights, own_strength, other_strength, other_labels
    )
    selected = np.argmax(scores, axis=1)
    best = scores[np.arange(len(selected)), selected]
    if len(groups) > 1:
        tied = scores >= best[:, None] - (
            64 * np.finfo(float).eps * own_strength[:, None]
        )
        counts = tied.sum(axis=1)
        positions = np.flatnonzero(counts > 1)
        # Uniform rank in each row's set of tied labels. Fixed-size chunks
        # cap temporary storage; no extra full-sized random/float matrix.
        ranks = (rng.random(len(positions)) * counts[positions]).astype(int)
        for start in range(0, len(positions), 128):
            pos = positions[start : start + 128]
            cumulative = np.cumsum(tied[pos], axis=1, dtype=np.int32)
            selected[pos] = (
                cumulative > ranks[start : start + 128, None]
            ).argmax(axis=1)
    return groups[selected], float(
        scores[np.arange(len(selected)), selected].sum()
    )


def _lpawb_stage_one(weights, rs, cs, rows, cols, objective, rng, max_iter):
    """Blue then red updates until no objective improvement; retain best."""
    trace = []
    iteration = 0
    while max_iter is None or iteration < max_iter:
        iteration += 1
        candidate_c, _ = _lpawb_assign(weights.T, cs, rs, rows, rng)
        candidate_r, candidate_objective = _lpawb_assign(
            weights, rs, cs, candidate_c, rng
        )
        if objective is not None and candidate_objective <= objective + 1e-12:
            return rows, cols, objective, trace, True
        rows, cols, objective = candidate_r, candidate_c, candidate_objective
        trace.append(objective)
    return rows, cols, objective, trace, False


def _lpawb_best_merge(weights, rs, cs, rows, cols, state=None):
    """Largest positive pair gain; necessarily mutually best (Algorithm 2)."""
    if (
        state
        and np.array_equal(state["rows"], rows)
        and np.array_equal(state["cols"], cols)
    ):
        gains = state["gains"]
        a, b = np.unravel_index(np.argmax(gains), gains.shape)
        gain = float(gains[a, b])
        return (
            (state["groups"][a], state["groups"][b], gain)
            if gain > 1e-12
            else None
        )
    groups = np.intersect1d(rows, cols)
    if len(groups) < 2:
        return None
    # Include one-sided labels in grouped sums, but only joint modules merge.
    all_groups = np.union1d(rows, cols)
    ri, ci = (
        np.searchsorted(all_groups, rows),
        np.searchsorted(all_groups, cols),
    )
    count = len(all_groups)
    rind = sparse.csr_matrix(
        (np.ones(len(rows)), (np.arange(len(rows)), ri)),
        shape=(len(rows), count),
    )
    cind = sparse.csr_matrix(
        (np.ones(len(cols)), (np.arange(len(cols)), ci)),
        shape=(len(cols), count),
    )
    cross = rind.T @ (weights @ cind)
    if sparse.issparse(cross):
        cross = cross.toarray()
    take = np.searchsorted(all_groups, groups)
    cross = cross[np.ix_(take, take)]
    r = np.bincount(ri, weights=rs, minlength=count)[take]
    c = np.bincount(ci, weights=cs, minlength=count)[take]
    gains = cross + cross.T - r[:, None] * c[None, :] - c[:, None] * r[None, :]
    np.fill_diagonal(gains, -np.inf)
    if state is not None:
        state.update(
            rows=rows.copy(),
            cols=cols.copy(),
            groups=groups,
            cross=cross,
            rs=r,
            cs=c,
            gains=gains,
            active=np.ones(len(groups), dtype=bool),
        )
    a, b = np.unravel_index(np.argmax(gains), gains.shape)
    gain = float(gains[a, b])
    return (groups[a], groups[b], gain) if gain > 1e-12 else None


def _lpawb_update_merge_state(state, a_label, b_label, rows, cols):
    """Incremental exact pair-gain formula for a no-change refinement."""
    groups, cross = state["groups"], state["cross"]
    a, b = np.searchsorted(groups, [a_label, b_label])
    cross[a, :] += cross[b, :]
    cross[:, a] += cross[:, b]
    state["rs"][a] += state["rs"][b]
    state["cs"][a] += state["cs"][b]
    state["active"][b] = False
    other = np.flatnonzero(state["active"])
    other = other[other != a]
    gain = (
        cross[a, other]
        + cross[other, a]
        - state["rs"][a] * state["cs"][other]
        - state["cs"][a] * state["rs"][other]
    )
    state["gains"][a, other] = gain
    state["gains"][other, a] = gain
    state["gains"][b, :] = state["gains"][:, b] = -np.inf
    state["rows"], state["cols"] = rows.copy(), cols.copy()


def _lpawb_optimize(weights, rs, cs, rng, max_iter):
    """Published singleton start, label propagation, merging and refinement."""
    transposed = len(rs) > len(cs)
    w, r, c = (weights.T, cs, rs) if transposed else (weights, rs, cs)
    rows = np.arange(len(r))  # Unique labels on the smaller species.
    rows, cols, objective, trace, converged = _lpawb_stage_one(
        w, r, c, rows, None, None, rng, max_iter
    )
    phases, merge_gains = [len(trace)], []
    merge_state = {}
    while converged:
        best = _lpawb_best_merge(w, r, c, rows, cols, merge_state)
        if best is None:
            break
        a, b, gain = best
        rows, cols = rows.copy(), cols.copy()
        rows[rows == b], cols[cols == b] = a, a
        _lpawb_update_merge_state(merge_state, a, b, rows, cols)
        objective += gain
        merge_gains.append(gain)
        trace.append(objective)
        rows, cols, objective, refinement, converged = _lpawb_stage_one(
            w, r, c, rows, cols, objective, rng, max_iter
        )
        trace += refinement
        phases.append(len(refinement))
    if transposed:
        rows, cols = cols, rows
    return objective, rows, cols, trace, converged, phases, merge_gains


def _brim_optimize(
    weights, row_strength, col_strength, col_labels, max_iter, row_labels=None
):
    """BRIM alternating updates; trace records actual normalized objectives."""
    history = []
    for _ in range(max_iter):
        old_rows, old_cols = row_labels, col_labels
        row_labels, _ = _brim_assign(
            weights, row_strength, col_strength, col_labels, row_labels
        )
        col_labels, objective = _brim_assign(
            weights.T, col_strength, row_strength, row_labels, col_labels
        )
        history.append(objective)
        if (
            old_rows is not None
            and np.array_equal(old_rows, row_labels)
            and np.array_equal(old_cols, col_labels)
        ):
            return row_labels, col_labels, history, True
    return row_labels, col_labels, history, False


def _merge_modularity_groups(weights, row_strength, col_strength, rows, cols):
    """Greedy positive-gain whole-module merges (BRIM runs afterwards)."""
    groups = np.union1d(rows, cols)
    ri, ci = np.searchsorted(groups, rows), np.searchsorted(groups, cols)
    count = len(groups)
    if count < 2:
        return rows, cols, 0
    r_indicator = sparse.csr_matrix(
        (np.ones(len(rows)), (np.arange(len(rows)), ri)),
        shape=(len(rows), count),
    )
    c_indicator = sparse.csr_matrix(
        (np.ones(len(cols)), (np.arange(len(cols)), ci)),
        shape=(len(cols), count),
    )
    cross = r_indicator.T @ (weights @ c_indicator)
    if sparse.issparse(cross):
        cross = cross.toarray()
    rs = np.bincount(ri, weights=row_strength, minlength=count)
    cs = np.bincount(ci, weights=col_strength, minlength=count)
    # Keep the original greedy maximum-gain order, updating only pairs
    # touching the merged group. Versioned heap entries discard stale gains.
    # Avoid rebuilding and copying K x K matrices after each of K merges.
    gains = (
        cross + cross.T - rs[:, None] * cs[None, :] - cs[:, None] * rs[None, :]
    )
    ii, jj = np.nonzero(np.triu(gains > 1e-12, k=1))
    queue = [
        (-float(gains[i, j]), int(i), int(j), 0, 0) for i, j in zip(ii, jj)
    ]
    heapq.heapify(queue)
    version = np.zeros(count, dtype=int)
    active = np.ones(count, dtype=bool)
    parent = np.arange(count)
    merged = 0
    while queue:
        _, a, b, va, vb = heapq.heappop(queue)
        if (
            not active[a]
            or not active[b]
            or version[a] != va
            or version[b] != vb
        ):
            continue
        cross[a, :] += cross[b, :]
        cross[:, a] += cross[:, b]
        rs[a] += rs[b]
        cs[a] += cs[b]
        active[b] = False
        parent[b] = a
        version[a] += 1
        version[b] += 1
        others = np.flatnonzero(active)
        others = others[others != a]
        new_gain = (
            cross[a, others]
            + cross[others, a]
            - rs[a] * cs[others]
            - cs[a] * rs[others]
        )
        for other, gain in zip(
            others[new_gain > 1e-12], new_gain[new_gain > 1e-12]
        ):
            i, j = sorted((a, int(other)))
            heapq.heappush(
                queue, (-float(gain), i, j, int(version[i]), int(version[j]))
            )
        merged += 1
    # Resolve merge ancestry once, rather than scanning every gene per merge.
    for group in range(count):
        parent[group] = parent[parent[group]]
    return groups[parent[ri]], groups[parent[ci]], merged


def extract_dense_blocks_modularity(
    A,
    *,
    min_component_size=4,
    max_blocks=None,
    row_labels=None,
    col_labels=None,
    max_iter=50,
    return_details=False,
):
    """Find weighted mouse/human modules without thresholding or a fixed K.

    Uses weighted Barber modularity:
        M = sum_b [W(R_b,C_b)/T - r(R_b)*c(C_b)/T**2], T=W.sum().
    It compares within-module transport with an independent coupling having
    the same marginals. All positive weights are used; row/column zero mass
    is excluded. BRIM alternately maximizes the objective over each species.
    Two deterministic singleton initializations (one per species), positive-
    gain module merging, and subsequent BRIM refinement reduce local traps.
    Empty modules disappear; no cluster count or GO annotations are supplied.

    References: Barber (2007), Phys Rev E 76:066102,
    https://doi.org/10.1103/PhysRevE.76.066102; weighted modularity and module
    merging are motivated by Beckett (2016), R Soc Open Sci 3:140536,
    https://doi.org/10.1098/rsos.140536. This implements weighted BRIM with
    our bounded deterministic search schedule, not the full LPAwb+/DIRTLPAwb+
    algorithms. A local-search heuristic, not a global optimum guarantee.

    Accepts a finite nonnegative dense array or scipy sparse matrix. Returned
    row_indices/col_indices refer to original axes. Blocks contain at least
    min_component_size positive entries, default 4. With no objective gain,
    return one active rectangle rather than invent subdivisions. With gain,
    retain modules whose contribution exceeds numerical tolerance. Modules
    need not be filled rectangles: this detects excess mass, not binary
    density.
    max_iter is a computational limit per BRIM phase, default 50; two starts
    are fixed. return_details adds memberships, optimizer traces/convergence,
    and the internal objective (not an additional GO ranking metric).

    Modularity can merge small modules (resolution limit); each gene belongs
    to at most one returned module, so overlapping biology is not modeled.
    OT weights are not independent observations, so no p-value or SBM
    recovery theorem is asserted. No tuning or selection uses Q, S, or GO.
    """
    return _extract_weighted_blocks(
        A,
        min_component_size=min_component_size,
        max_blocks=max_blocks,
        row_labels=row_labels,
        col_labels=col_labels,
        max_iter=max_iter,
        return_details=return_details,
        algorithm="brim",
    )


def extract_dense_blocks_hard_brim(A, **kwargs):
    """Disjoint weighted BRIM; alias of extract_dense_blocks_modularity.

    See that function for input, filtering, objective and max_iter details.
    Two deterministic starts, greedy merging and one refinement per start.
    """
    return extract_dense_blocks_modularity(A, **kwargs)


def extract_dense_blocks_brim(A, **kwargs):
    """Compatibility alias of extract_dense_blocks_hard_brim."""
    return extract_dense_blocks_hard_brim(A, **kwargs)


def _validate_overlap_eta(eta):
    if (
        isinstance(eta, (bool, np.bool_))
        or not isinstance(eta, (int, float, np.integer, np.floating))
        or not np.isfinite(eta)
        or not 0 < eta <= 1
    ):
        raise ValueError("eta must be a finite number in (0, 1] for soft_brim")
    return float(eta)


def _expand_brim_cores(
    A,
    cores,
    eta,
    *,
    row_labels=None,
    col_labels=None,
    return_details=False,
    include_blocks=True,
):
    """Frozen-core candidates, accepted by positive uncovered-cell gain.

    Private shared implementation also lets score_ot_plan cache hard cores
    independently of eta. Core dictionaries need only original-axis indices.
    """
    eta = _validate_overlap_eta(eta)
    original = (
        A.tocsr().astype(float, copy=False)
        if sparse.issparse(A)
        else np.asarray(A, dtype=float)
    )
    nr, nc = original.shape
    k = len(cores)
    names_r = np.arange(nr) if row_labels is None else np.asarray(row_labels)
    names_c = np.arange(nc) if col_labels is None else np.asarray(col_labels)
    if names_r.shape != (nr,) or names_c.shape != (nc,):
        raise ValueError("row_labels and col_labels must match A's axes")

    def indicators(axis, size):
        indices = (
            np.concatenate([np.asarray(b[axis], dtype=int) for b in cores])
            if k
            else np.empty(0, int)
        )
        groups = np.repeat(np.arange(k), [len(b[axis]) for b in cores])
        return sparse.csr_matrix(
            (np.ones(len(indices)), (indices, groups)), shape=(size, k)
        )

    core_r, core_c = (
        indicators("row_indices", nr),
        indicators("col_indices", nc),
    )
    details = dict(
        detector="soft_brim",
        eta=eta,
        core_blocks=cores,
        row_memberships=core_r.astype(bool),
        col_memberships=core_c.astype(bool),
        expansion_rule="positive_uncovered_excess",
        proposed_additions=0,
        accepted_additions=0,
        rejected_nonpositive=0,
        rejected_redundant=0,
        expansion_gains=np.empty(0),
        accepted_moves=[],
        structural_objective_before=0.0,
        structural_objective_after=0.0,
        structural_objective_trace=np.array([0.0]),
    )
    if not k:
        return ([], details) if return_details else []
    values = original.data if sparse.issparse(original) else original
    # Normalize before summing for the same extreme-mass stability as BRIM.
    weights = original / float(values.max())
    weights = weights / float(weights.sum())
    if (
        not sparse.issparse(weights)
        and np.count_nonzero(weights) < weights.size / 5
    ):
        weights = sparse.csr_matrix(weights)
    rs = np.asarray(weights.sum(axis=1)).ravel()
    cs = np.asarray(weights.sum(axis=0)).ravel()

    def secondary(w, own_strength, other_strength, opposite_cores, own_cores):
        observed = w @ opposite_cores
        if sparse.issparse(observed):
            observed = observed.toarray()
        expected = (
            own_strength[:, None]
            * np.asarray(opposite_cores.T @ other_strength).ravel()[None, :]
        )
        scores = observed - expected
        # Relative cancellation tolerance suppresses roundoff-only overlap
        # in rank-one transport; no mass or biological cutoff is introduced.
        numerical = 64 * np.finfo(float).eps
        tolerance = numerical * (np.abs(observed) + np.abs(expected))
        scores[scores <= tolerance] = 0.0
        threshold = eta * scores.max(axis=1, keepdims=True)
        joined = (scores > 0) & (scores >= threshold * (1 - numerical))
        ri, ci = own_cores.nonzero()
        joined[ri, ci] = True  # Keep every original core membership.
        return joined, scores

    # Candidate eligibility uses ORIGINAL cores; acceptance below uses the
    # current cover, including earlier accepted additions on either species.
    proposed_r, affinity_r = secondary(weights, rs, cs, core_c, core_r)
    proposed_c, affinity_c = secondary(weights.T, cs, rs, core_r, core_c)
    joined_r, joined_c = (
        core_r.toarray().astype(bool),
        core_c.toarray().astype(bool),
    )
    covered = np.zeros((nr, nc), dtype=bool)
    initial_objective = 0.0
    for core in cores:
        r, c = core["row_indices"], core["col_indices"]
        covered[np.ix_(r, c)] = True
        initial_objective += float(weights[np.ix_(r, c)].sum()) - float(
            rs[r].sum() * cs[c].sum()
        )
    # eta generates candidates against ORIGINAL cores only. A fixed order
    # prevents cascading eligibility. Recompute the true incremental gain
    # against CURRENT memberships and CURRENT covered cells before accepting.
    ri, rk = np.nonzero(proposed_r & ~joined_r)
    ci, ck = np.nonzero(proposed_c & ~joined_c)
    axes = np.r_[np.zeros(len(ri), dtype=int), np.ones(len(ci), dtype=int)]
    nodes, groups = np.r_[ri, ci], np.r_[rk, ck]
    priorities = np.r_[affinity_r[ri, rk], affinity_c[ci, ck]]
    # Round only the ordering key (not affinities/gains), avoiding unstable
    # numerical tie order. Normalized priorities use 14 decimal places.
    order = np.lexsort((groups, nodes, axes, -np.round(priorities, 14)))
    csr = weights.tocsr() if sparse.issparse(weights) else None
    csc = weights.tocsc() if csr is not None else None
    accepted, redundant, nonpositive = 0, 0, 0
    gains, moves = [], []
    for candidate in order:
        axis, node, group = (
            int(axes[candidate]),
            int(nodes[candidate]),
            int(groups[candidate]),
        )
        if axis == 0:
            opposite = np.flatnonzero(joined_c[:, group] & ~covered[node, :])
            if opposite.size:
                observed = float(
                    (
                        csr[node, opposite]
                        if csr is not None
                        else weights[node, opposite]
                    ).sum()
                )
                expected = float(rs[node] * cs[opposite].sum())
        else:
            opposite = np.flatnonzero(joined_r[:, group] & ~covered[:, node])
            if opposite.size:
                observed = float(
                    (
                        csc[opposite, node]
                        if csc is not None
                        else weights[opposite, node]
                    ).sum()
                )
                expected = float(cs[node] * rs[opposite].sum())
        if not opposite.size:
            redundant += 1
            continue
        gain = observed - expected
        tolerance = 64 * np.finfo(float).eps * (abs(observed) + abs(expected))
        if gain <= tolerance:
            nonpositive += 1
            continue
        if axis == 0:
            joined_r[node, group] = True
            covered[node, opposite] = True
        else:
            joined_c[node, group] = True
            covered[opposite, node] = True
        accepted += 1
        gains.append(gain)
        if return_details:
            moves.append((axis, node, group))
    if return_details:
        details.update(
            expansion_rule="positive_uncovered_excess",
            proposed_additions=len(order),
            accepted_additions=accepted,
            rejected_redundant=redundant,
            rejected_nonpositive=nonpositive,
            expansion_gains=np.asarray(gains),
            accepted_moves=moves,
            structural_objective_before=initial_objective,
            structural_objective_after=initial_objective + sum(gains),
            structural_objective_trace=initial_objective
            + np.r_[0.0, np.cumsum(gains)],
        )
    blocks, seen, retained = [], set(), []
    for group, core in enumerate(cores):
        ri, ci = (
            np.flatnonzero(joined_r[:, group]),
            np.flatnonzero(joined_c[:, group]),
        )
        identity = (tuple(ri), tuple(ci))
        if identity in seen:
            continue
        seen.add(identity)
        retained.append(group)
        if not include_blocks:
            blocks.append(dict(row_indices=ri, col_indices=ci))
            continue
        sub = original[np.ix_(ri, ci)]
        edges = (
            sub.count_nonzero()
            if sparse.issparse(sub)
            else np.count_nonzero(sub)
        )
        blocks.append(
            dict(
                row_indices=ri,
                col_indices=ci,
                row_labels=names_r[ri],
                col_labels=names_c[ci],
                block=sub.toarray() if sparse.issparse(sub) else sub,
                size=int(edges),
                core_row_indices=np.asarray(core["row_indices"]).copy(),
                core_col_indices=np.asarray(core["col_indices"]).copy(),
            )
        )
    if return_details:
        details.update(
            row_memberships=sparse.csr_matrix(joined_r[:, retained]),
            col_memberships=sparse.csr_matrix(joined_c[:, retained]),
        )
    return (blocks, details) if return_details else blocks


def extract_dense_blocks_soft_brim(
    A,
    *,
    eta,
    min_component_size=4,
    max_blocks=None,
    row_labels=None,
    col_labels=None,
    max_iter=50,
    return_details=False,
):
    """Hard BRIM cores with controlled overlapping, binary memberships.

    eta is required, finite, and in (0, 1]. For a mouse gene i and ORIGINAL
    human core C_k, a_ik=sum_{j in C_k} A_ij-r_i*sum_{j in C_k} c_j/T,
    where r/c are complete-plan marginals and T=A.sum(). Add membership k
    when a_ik>0 and a_ik>=eta*max_l a_il; use the symmetric human rule.
    These are CANDIDATES, not automatic memberships. All hard memberships
    stay. Process candidates once, in descending original normalized excess
    (priority rounded to 14 decimals; ties: mouse before human, index, core).
    Accept only when currently uncovered cells in the proposed strip have
    positive sum of P_ij/T-r_i*c_j/T**2, beyond roundoff tolerance. The strip
    uses CURRENT opposite memberships, accounting for newly added corners.
    Every accepted addition strictly increases union excess M; no GO is used.
    Candidate eligibility remains frozen, so no cascading proposals occur.
    Smaller eta admits more candidates, but final memberships and M need not
    be nested/monotone in eta due to interactions and greedy ordering.
    eta=1 can still add strongest ties. This is a deterministic one-pass
    heuristic; rejected candidates are not revisited, so no local/global
    optimum or improved biological Q/S is guaranteed.

    This is our transport-only core-expansion heuristic, not fuzzy entropic
    Barber modularity or a new global modularity optimizer. Memberships are
    binary, although genes may belong to multiple blocks. It cannot recover
    a module already merged by hard BRIM. The core count/order is retained,
    except identical expanded rectangles are deduplicated. Size/max_blocks
    filters apply to hard cores before expansion; no GO information is used.
    Sparse grouped products avoid the full centered matrix. Labels refer to
    original axes, inputs are unchanged, and extreme mass scaling is stable.
    return_details includes the HARD optimizer under hard_core_details plus
    binary row/col_memberships, candidate/rejection counts, accepted_moves
    (axis 0=mouse, 1=human), per-move gains, and the union-objective trace.
    M=sum_{covered cells}(P_ij/T-r_i*c_j/T**2) counts each cell once; it is
    a proposed detection objective, not the GO score Q or S. Fully overlapping
    covers with the same union have the same M; redundant memberships are
    rejected. A gain gate cannot recover modules lost in the hard cores.
    """
    eta = _validate_overlap_eta(eta)
    cores, hard_details = extract_dense_blocks_hard_brim(
        A,
        min_component_size=min_component_size,
        max_blocks=max_blocks,
        row_labels=row_labels,
        col_labels=col_labels,
        max_iter=max_iter,
        return_details=True,
    )
    output = _expand_brim_cores(
        A,
        cores,
        eta,
        row_labels=row_labels,
        col_labels=col_labels,
        return_details=return_details,
    )
    if return_details:
        output[1]["hard_core_details"] = hard_details
    return output


def extract_dense_blocks_lpawb(
    A,
    *,
    min_component_size=4,
    max_blocks=None,
    row_labels=None,
    col_labels=None,
    seed=0,
    max_iter=None,
    return_details=False,
):
    """Beckett (2016) LPAwb+ for finite nonnegative weighted bipartite plans.

    Implements Algorithm 1 and Eq. 2.5: unique singleton labels on the
    smaller species; alternate blue/red modularity-maximizing updates with
    uniform random choices among tied maxima; merge the largest positive-
    gain pair of joint modules (satisfying Algorithm 2), then repeat label
    propagation. Stop when no node sweep or module merge increases the
    objective, to numerical tolerance 1e-12 in normalized modularity.
    Equal or worse sweeps are discarded, avoiding tie oscillation; reported
    objective always belongs to the retained memberships. Same-species
    updates are vectorized because they depend only on the other species.
    Reference: https://doi.org/10.1098/rsos.140536, Algorithms 1-2. This is
    an independent Python implementation of the published workflow, not
    a bit-for-bit port of the author software. No DIRTLPAwb+ repeated search.

    seed is a nonnegative integer, default 0. It controls stochastic tie
    choices through a local RNG; no global RNG is changed. One start only.
    max_iter=None runs each label phase until its objective plateaus; an
    optional positive integer bounds each phase and reports converged=False
    when reached. return_details includes optimizer_traces, phase_iterations,
    merge_gains, converged and internal weighted_modularity. These are
    structural diagnostics, not extra GO ranking scores.

    Shares BRIM input validation, sparse handling, scale normalization,
    zero-axis removal, original-axis indexing and minimum-size filtering.
    With no modularity gain, return one active rectangle. No GO information
    is used. Modularity has a resolution limit, disjoint memberships and
    local optima; more modular does not guarantee more biological relevance.
    """
    return _extract_weighted_blocks(
        A,
        min_component_size=min_component_size,
        max_blocks=max_blocks,
        row_labels=row_labels,
        col_labels=col_labels,
        max_iter=max_iter,
        return_details=return_details,
        algorithm="lpawb",
        seed=seed,
    )


def _extract_weighted_blocks(
    A,
    *,
    min_component_size,
    max_blocks,
    row_labels,
    col_labels,
    max_iter,
    return_details,
    algorithm,
    seed=0,
):
    """Shared normalized transport preparation and membership extraction."""
    if (
        not isinstance(min_component_size, (int, np.integer))
        or isinstance(min_component_size, (bool, np.bool_))
        or min_component_size < 1
    ):
        raise ValueError("min_component_size must be a positive integer")
    if max_iter is None:
        if algorithm != "lpawb":
            raise ValueError("max_iter must be a positive integer")
    elif (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, (bool, np.bool_))
        or max_iter < 1
    ):
        raise ValueError(
            "max_iter must be None (lpawb only) or a positive integer"
        )
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    if max_blocks is not None and (
        not isinstance(max_blocks, (int, np.integer))
        or isinstance(max_blocks, (bool, np.bool_))
        or max_blocks < 1
    ):
        raise ValueError("max_blocks must be None or a positive integer")
    original = (
        A.tocsr(copy=True).astype(float)
        if sparse.issparse(A)
        else np.asarray(A, dtype=float)
    )
    if original.ndim != 2:
        raise ValueError("A must be two-dimensional")
    values = original.data if sparse.issparse(original) else original
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("A must be finite and nonnegative")
    nr, nc = original.shape
    names_r = np.arange(nr) if row_labels is None else np.asarray(row_labels)
    names_c = np.arange(nc) if col_labels is None else np.asarray(col_labels)
    if names_r.shape != (nr,) or names_c.shape != (nc,):
        raise ValueError("row_labels and col_labels must match A's axes")
    details = dict(
        row_cluster_labels=np.full(nr, -1, dtype=int),
        col_cluster_labels=np.full(nc, -1, dtype=int),
        weighted_modularity=0.0,
        optimizer_traces=[],
        converged=True,
        detector="lpawb" if algorithm == "lpawb" else "weighted_brim",
    )
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= 0:
        return ([], details) if return_details else []
    # Normalize before summing: stable even for tiny/very large OT masses.
    normalized = original / maximum
    active_r = np.flatnonzero(np.asarray(normalized.sum(axis=1)).ravel() > 0)
    active_c = np.flatnonzero(np.asarray(normalized.sum(axis=0)).ravel() > 0)
    weights = normalized[np.ix_(active_r, active_c)]
    weights = weights / float(weights.sum())
    if sparse.issparse(weights):
        weights = weights.tocsr()
        weights.eliminate_zeros()
    elif np.count_nonzero(weights) < weights.size / 5:
        weights = sparse.csr_matrix(weights)
    rs = np.asarray(weights.sum(axis=1)).ravel()
    cs = np.asarray(weights.sum(axis=0)).ravel()
    if algorithm == "lpawb":
        # Domain-separated stream leaves the GO-permutation seed sequence
        # unchanged and makes detector output independent of replication R.
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x4C5041])
        )
        objective, rows, cols, trace, converged, phases, gains = (
            _lpawb_optimize(weights, rs, cs, rng, max_iter)
        )
        details.update(
            optimizer_traces=[trace],
            phase_iterations=phases,
            merge_gains=gains,
            seed=int(seed),
        )
        best = objective, rows, cols, converged
    else:
        best = None
        for transpose in (False, True):
            w, r, c = (weights.T, cs, rs) if transpose else (weights, rs, cs)
            rows, cols, trace, converged = _brim_optimize(
                w, r, c, np.arange(len(c)), max_iter
            )
            rows, cols, merges = _merge_modularity_groups(w, r, c, rows, cols)
            if merges:
                rows, cols, refinement, done = _brim_optimize(
                    w, r, c, cols, max_iter, rows
                )
                trace += refinement
                converged &= done
            details["optimizer_traces"].append(trace)
            candidate = (
                (trace[-1], cols, rows, converged)
                if transpose
                else (trace[-1], rows, cols, converged)
            )
            if best is None or candidate[0] > best[0] + 1e-12:
                best = candidate
    objective, rows, cols, converged = best
    if objective <= 1e-12:
        rows, cols = np.zeros(len(rs), dtype=int), np.zeros(len(cs), dtype=int)
        objective = 0.0
    _, merged_labels = np.unique(np.r_[rows, cols], return_inverse=True)
    rows, cols = merged_labels[: len(rs)], merged_labels[len(rs) :]
    details.update(weighted_modularity=float(objective), converged=converged)
    details["row_cluster_labels"][active_r] = rows
    details["col_cluster_labels"][active_c] = cols
    candidates = []
    for group in np.intersect1d(rows, cols):
        r, c = np.flatnonzero(rows == group), np.flatnonzero(cols == group)
        sub = weights[np.ix_(r, c)]
        edges = sub.nnz if sparse.issparse(sub) else np.count_nonzero(sub)
        mass = float(sub.sum())
        contribution = mass - float(rs[r].sum() * cs[c].sum())
        if edges < min_component_size or (
            objective > 0 and contribution <= 1e-12
        ):
            continue
        ri, ci = active_r[r], active_c[c]
        block = original[np.ix_(ri, ci)]
        if sparse.issparse(block):
            block = block.toarray()
        candidates.append(
            (
                mass,
                {
                    "row_indices": ri,
                    "col_indices": ci,
                    "row_labels": names_r[ri],
                    "col_labels": names_c[ci],
                    "block": block,
                    "size": int(edges),
                },
            )
        )
    candidates.sort(
        key=lambda item: (
            -item[0],
            tuple(item[1]["row_indices"]),
            tuple(item[1]["col_indices"]),
        )
    )
    blocks = [item[1] for item in candidates[:max_blocks]]
    return (blocks, details) if return_details else blocks


def extract_dense_blocks_adaptive(
    A, n_clusters=10, activity_threshold=None, **kwargs
):
    """Detect separated dense support components, otherwise use coclustering.

    This optional detector uses only the transport, never GO annotations.
    Build the bipartite graph of above-threshold cells. When it contains at
    least two nontrivial components and every nontrivial component has active
    edges in at least half its mouse x human rectangle, return those exact
    gene memberships directly. Each must contain at least
    `min_component_size` active edges (default 4); isolated matches are not
    treated as blocks. This recovers small scattered rectangles without
    imposing a spatial smoothing window or a number of clusters on them.

    For connected/diffuse support, use extract_dense_blocks_coclustered with
    the supplied arguments. The majority-activity rule is a fixed structural
    heuristic; this method is not guaranteed to improve biological rankings.
    For raw-magnitude detection, min_density_ratio filtering, or full spatial
    details, also use the original coclustering path. Output gene indices
    refer to the original array, exactly as in the existing detector.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"`A` must be 2-dimensional, got shape {A.shape}")
    magnitude = np.abs(A)
    if activity_threshold is None:
        positive = magnitude[magnitude > 0]
        activity_threshold = (
            np.nextafter(float(np.median(positive)), -np.inf)
            if positive.size
            else 0.0
        )
    fallback = dict(
        n_clusters=n_clusters, activity_threshold=activity_threshold, **kwargs
    )
    if (
        not kwargs.get("use_binary_activity", True)
        or kwargs.get("min_density_ratio") is not None
        or kwargs.get("return_details", False)
    ):
        return extract_dense_blocks_coclustered(A, **fallback)
    support = sparse.csr_matrix(magnitude > activity_threshold)
    graph = sparse.bmat([[None, support], [support.T, None]], format="csr")
    _, labels = connected_components(graph, directed=False)
    rows = labels[: A.shape[0]]
    cols = labels[A.shape[0] :]
    row_degree = np.diff(support.indptr)
    col_degree = np.asarray(support.sum(axis=0)).ravel()
    groups = np.intersect1d(
        np.unique(rows[row_degree > 0]), np.unique(cols[col_degree > 0])
    )
    minimum = kwargs.get("min_component_size", 4)
    candidates = []
    for group in groups:
        r = np.flatnonzero(rows == group)
        c = np.flatnonzero(cols == group)
        edges = int(row_degree[r].sum())
        if edges < minimum:
            continue
        if edges * 2 < len(r) * len(c):
            return extract_dense_blocks_coclustered(A, **fallback)
        candidates.append((edges, r, c))
    if len(candidates) < 2:
        return extract_dense_blocks_coclustered(A, **fallback)
    candidates.sort(key=lambda item: item[0], reverse=True)
    if kwargs.get("max_blocks") is not None:
        candidates = candidates[: kwargs["max_blocks"]]
    total = float(magnitude.sum())
    row_labels = (
        np.arange(A.shape[0])
        if kwargs.get("row_labels") is None
        else np.asarray(kwargs["row_labels"])
    )
    col_labels = (
        np.arange(A.shape[1])
        if kwargs.get("col_labels") is None
        else np.asarray(kwargs["col_labels"])
    )
    blocks = []
    for edges, r, c in candidates:
        block = A[np.ix_(r, c)]
        inside = float(np.abs(block).sum())
        outside_size = A.size - block.size
        outside_mean = (
            max(0.0, total - inside) / outside_size + 1e-12
            if outside_size
            else np.nan
        )
        blocks.append(
            {
                "row_indices": r,
                "col_indices": c,
                "row_labels": row_labels[r],
                "col_labels": col_labels[c],
                "block": block,
                "size": edges,
                "total_density": float(edges),
                "density_ratio": float(inside / block.size / outside_mean),
            }
        )
    return blocks


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.random((300, 1000)) * 0.05
    A[20:140, 10:230] += rng.random((120, 220)) * 0.8  # block 1 (strong)
    A[180:260, 700:950] += rng.random((80, 250)) * 0.5  # block 2 (weaker)

    blocks = extract_dense_blocks(A, mad_multiplier=4.0)
    for i, b in enumerate(blocks, 1):
        print(
            f"Block {i}: bbox={b['bbox']}  size={b['size']}  "
            f"density_ratio={b['density_ratio']:.2f}x"
        )

    # Same idea, but with rows/columns scattered (not contiguous) beforehand,
    # recovered by co-clustering FIRST, then extracting.
    print("\nWith scattered rows/columns, recovered via co-clustering:")
    n_rows, n_cols = 150, 400
    B = rng.random((n_rows, n_cols)) * 0.05
    row_block = rng.permutation(n_rows)[:40]
    col_block = rng.permutation(n_cols)[:60]
    B[np.ix_(row_block, col_block)] += rng.random((40, 60)) * 0.8

    coclustered_blocks = extract_dense_blocks_coclustered(
        B, n_clusters=2, mad_multiplier=4.0, min_density_ratio=1.5
    )
    for i, b in enumerate(coclustered_blocks, 1):
        row_overlap = len(set(b["row_indices"]) & set(row_block)) / len(
            set(b["row_indices"]) | set(row_block)
        )
        col_overlap = len(set(b["col_indices"]) & set(col_block)) / len(
            set(b["col_indices"]) | set(col_block)
        )
        print(
            f"Block {i}: {len(b['row_indices'])}x{len(b['col_indices'])}  "
            f"density_ratio={b['density_ratio']:.2f}x  "
            f"row_jaccard={row_overlap:.2f}  col_jaccard={col_overlap:.2f}"
        )
