"""Experimental overlapping KL-NMF and Bayesian ARD-MAP block detectors.

Only NumPy and SciPy are required. An optional local C compiler accelerates
CPU updates through nmf_native.py; backend='numpy' keeps the reference path.
Zeros participate in the objective even
though the ratio X/(UV.T) is evaluated only at nonzero observations.
Bayesian updates follow Psorakis et al. (2011), Algorithm 1, extended to a
rectangular matrix. For continuous transport this is a regularized KL/MAP
adaptation, not a literal count likelihood or a posterior uncertainty estimate.
"""
from time import perf_counter
import numpy as np
from scipy import sparse
from .extract_dense_block import extract_dense_blocks_hard_brim, _validate_overlap_eta


def _matrix(A):
    x = sparse.csr_matrix(A, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError('NMF input must be finite and nonnegative')
    return x


def _predictions(x, u, v):
    # Dense BLAS is faster for these moderate matrices; bound scratch memory.
    if x.nnz >= 30_000 and x.shape[0] * x.shape[1] <= 8_000_000:
        prediction = u @ v.T
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        return prediction[rows, x.indices]
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    result = np.empty(x.nnz)
    batch = max(1, 1_000_000 // max(1, u.shape[1]))
    for start in range(0, x.nnz, batch):
        end = min(start + batch, x.nnz)
        result[start:end] = np.einsum('ij,ij->i', u[rows[start:end]], v[x.indices[start:end]])
    return result


def _ratio(x, u, v, predict=None):
    return sparse.csr_matrix((x.data / np.maximum((predict or _predictions)(x, u, v), np.finfo(float).tiny),
                             x.indices, x.indptr), shape=x.shape)


def objective(x, u, v, beta=None, a=5., b=2., predict=None, constant=None):
    """Full generalized KL, including zero cells, plus optional ARD terms.

    constant is sum(x*log(x)-x) over observed cells; it depends only on x, so
    a fitting loop can compute it once and pass it back in.
    """
    pred = np.maximum((predict or _predictions)(x, u, v), np.finfo(float).tiny)
    if constant is None:
        constant = float(np.sum(x.data * np.log(x.data) - x.data))
    loss = constant - float(np.dot(x.data, np.log(pred))) + float(u.sum(0) @ v.sum(0))
    if beta is not None:
        c = (x.shape[0] + x.shape[1]) / 2 + a - 1
        energy = .5 * (np.square(u).sum(0) + np.square(v).sum(0))
        loss += float(beta @ (energy + b) - c * np.log(beta).sum())
    return loss


def _step(x, u, v, beta=None, a=5., b=2., predict=None):
    """One alternating H, W, beta update; mutates the supplied factors."""
    tiny = np.finfo(float).tiny
    if predict is not None and hasattr(predict, 'update'):
        predict.update(u, v, beta)
    else:
        denominator = u.sum(0)[None, :] + (v * beta if beta is not None else 0.)
        v *= (_ratio(x, u, v, predict).T @ u) / np.maximum(denominator, tiny)
        denominator = v.sum(0)[None, :] + (u * beta if beta is not None else 0.)
        u *= (_ratio(x, u, v, predict) @ v) / np.maximum(denominator, tiny)
    if beta is not None:
        c = (x.shape[0] + x.shape[1]) / 2 + a - 1
        beta[:] = c / (.5 * (np.square(u).sum(0) + np.square(v).sum(0)) + b)
    return u, v, beta


def fit_nmf(A, *, method='kl_nmf', cores=None, max_iter=500, tol=1e-5,
            seed=42, weight_scale=1., n_components=None, a=5., b=2., backend='auto',
            device='cpu', dtype='float64'):
    """Fit once, then call blocks_from_fit for each eta.

    KL fits P/sum(P). Bayesian fits P/sum(P)*nnz(P)*weight_scale,
    with a=5,b=2 from the paper. Default initial rank is the hard BRIM
    core count; n_components can increase capacity using random extra factors.
    This is a local MAP fit, not posterior sampling. Convergence refers to
    relative objective decrease over five updates, not a global optimum.
    backend='auto' tries native sparse CPU updates and falls back to NumPy;
    'native' requires a C compiler, and 'numpy' uses the original reference.
    CPU reference/native paths use float64. A non-CPU device selects PyTorch
    with backend='auto'; backend='torch' also supports CPU validation.
    dtype='float32' is explicit and required for MPS. Initialization and stopping
    settings are shared, but reduced precision can change fitted memberships.
    """
    started = perf_counter()
    if method not in {'kl_nmf', 'bayesian_nmf'}:
        raise ValueError('method must be kl_nmf or bayesian_nmf')
    if backend not in {'auto','numpy','native','torch'}:
        raise ValueError('backend must be auto, numpy, native, or torch')
    if not isinstance(device,str) or not (device in {'cpu','mps','cuda'} or (device.startswith('cuda:') and device[5:].isdigit())):
        raise ValueError('device must be cpu, mps, cuda, or cuda:index')
    if dtype not in {'float32','float64'}:
        raise ValueError('dtype must be float32 or float64')
    use_torch = backend=='torch' or (backend=='auto' and device!='cpu')
    if not use_torch and (device!='cpu' or dtype!='float64'):
        raise ValueError('NumPy/native backends use CPU float64; choose backend="torch" for other settings')
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be finite and positive')
    if not all(np.isfinite(z) and z > 0 for z in [weight_scale, a, b]):
        raise ValueError('weight_scale, a and b must be finite and positive')
    x = _matrix(A)
    nr, nc = x.shape
    active_r = np.flatnonzero(np.asarray(x.sum(1)).ravel() > 0)
    active_c = np.flatnonzero(np.asarray(x.sum(0)).ravel() > 0)
    if cores is None:
        cores = extract_dense_blocks_hard_brim(x)
    k = len(cores) if n_components is None else n_components
    if isinstance(k, bool) or int(k) != k or k < len(cores) or k < 0:
        raise ValueError('n_components must be an integer at least the hard core count')
    k = int(k)
    u, v = np.zeros((nr, k)), np.zeros((nc, k))
    details = dict(method=method, initial_rank=k, core_count=len(cores),requested_backend=backend,device=device,dtype=dtype,
                   weight_scale=weight_scale, max_iter=max_iter, tol=tol,
                   seed=int(seed), a=a, b=b, iterations=0, converged=False,
                   objective_trace=[], objective_iterations=[])
    if not x.nnz or not k:
        details.update(fit_seconds=perf_counter()-started, converged=True,
                       active_components=0, target_mass=0., fitted_mass=0.,backend='none')
        return dict(U=u, V=v, details=details)
    x.data /= x.data.max()
    x.data /= x.data.sum()
    target = float(x.nnz) * weight_scale if method == 'bayesian_nmf' else 1.
    x.data *= target
    rng = np.random.default_rng(seed)
    floor = 1e-3 * np.sqrt(target / (max(1, len(active_r)) * max(1, len(active_c)) * k))
    u[active_r] = floor * rng.uniform(.5, 1.5, (len(active_r), k))
    v[active_c] = floor * rng.uniform(.5, 1.5, (len(active_c), k))
    for j, core in enumerate(cores):
        rr, cc = np.asarray(core['row_indices']), np.asarray(core['col_indices'])
        sub = x[rr][:, cc]
        mass = float(sub.sum())
        if mass > 0:
            u[rr, j] += np.asarray(sub.sum(1)).ravel() / np.sqrt(mass)
            v[cc, j] += np.asarray(sub.sum(0)).ravel() / np.sqrt(mass)
    # Extra components receive comparable, diffuse initial model mass.
    if k > len(cores):
        base = np.sqrt(target / (len(active_r) * len(active_c) * k))
        u[np.ix_(active_r, np.arange(len(cores), k))] *= base / floor
        v[np.ix_(active_c, np.arange(len(cores), k))] *= base / floor
    beta = np.ones(k) if method == 'bayesian_nmf' else None
    if use_torch:
        try:
            from .nmf_torch import fit_arrays
        except ImportError:
            from nmf_torch import fit_arrays
        result=fit_arrays(x,u,v,beta,device=device,dtype=dtype,max_iter=int(max_iter),tol=tol,a=a,b=b,target=target)
        details.update(result['details']);details['fit_seconds']=perf_counter()-started
        result['details']=details
        return result
    predict = None
    if backend in {'auto','native'}:
        try:
            try:
                from .nmf_native import SparsePredictor
            except ImportError:
                from nmf_native import SparsePredictor
            predict = SparsePredictor(x)
        except (ImportError,RuntimeError) as exc:
            if backend == 'native':
                raise
            details['backend_fallback'] = str(exc)
    details['backend'] = 'native' if predict is not None else 'numpy'
    kl_constant = float(np.sum(x.data * np.log(x.data) - x.data))  # fixed for this fit
    previous = objective(x, u, v, beta, a, b, predict, kl_constant)
    best = (previous, u.copy(), v.copy(), None if beta is None else beta.copy(), 0)
    details['objective_trace'].append(previous); details['objective_iterations'].append(0)
    for iteration in range(1, int(max_iter)+1):
        _step(x, u, v, beta, a, b, predict)
        # Avoid slow denormal arithmetic. This only removes factors so small
        # their squared model contributions are below float64 resolution.
        u[u < 1e-250] = 0.; v[v < 1e-250] = 0.
        if iteration % 5 == 0 or iteration == max_iter:
            value = objective(x, u, v, beta, a, b, predict, kl_constant)
            if not np.isfinite(value) or not np.isfinite(u).all() or not np.isfinite(v).all():
                raise FloatingPointError('Nonfinite NMF fit')
            details['objective_trace'].append(value); details['objective_iterations'].append(iteration)
            if value < best[0]:
                best = (value, u.copy(), v.copy(), None if beta is None else beta.copy(), iteration)
            # Signed ARD offsets can dominate. Scale convergence by data mass
            # or prior-adjusted objective magnitude, whichever is larger.
            offset = 0. if beta is None else k * ((nr+nc)/2+a-1) * (1-np.log(((nr+nc)/2+a-1)/b))
            relative = (previous-value) / max(target, abs(previous-offset), np.finfo(float).tiny)
            if -1e-10 <= relative <= tol:
                details['converged'] = True
                break
            previous = value
    # Published ARD fixed-point updates can oscillate. Return the lowest
    # checked objective, and expose both its iteration and the complete trace.
    _, u, v, beta, best_iteration = best
    mass = u.sum(0) * v.sum(0)
    active = mass > target * 1e-12
    details.update(iterations=iteration, best_iteration=best_iteration,
                   objective_increases=int(np.sum(np.diff(details['objective_trace']) > 1e-9)),
                   active_components=int(active.sum()),
                   target_mass=target, fitted_mass=float(mass.sum()),
                   fit_seconds=perf_counter()-started)
    return dict(U=u, V=v, beta=beta, details=details)


def blocks_from_fit(A, fit, eta=.5, min_component_size=4):
    """Threshold each gene's component mass relative to its strongest one.

    Component mass uses U_ik*sum(V_k) and V_jk*sum(U_k), so reciprocal
    rescaling of factors cannot change memberships. Dead components below
    1e-12 of the fitted target mass are excluded. No core is forced to stay.
    """
    eta = _validate_overlap_eta(eta)
    x = _matrix(A)
    u, v = fit['U'], fit['V']
    if u.shape[0] != x.shape[0] or v.shape[0] != x.shape[1] or u.shape[1] != v.shape[1]:
        raise ValueError('factor shapes must match A')
    if not u.shape[1]:
        return []
    row = u * v.sum(0); col = v * u.sum(0)
    alive = row.sum(0) > fit['details']['target_mass'] * 1e-12
    row[:, ~alive] = 0.; col[:, ~alive] = 0.
    mr = (row > 0) & (row >= eta * row.max(1, keepdims=True))
    mc = (col > 0) & (col >= eta * col.max(1, keepdims=True))
    blocks, seen = [], set()
    for j in np.flatnonzero(alive):
        rr, cc = np.flatnonzero(mr[:, j]), np.flatnonzero(mc[:, j])
        key = (tuple(rr), tuple(cc))
        if len(rr) and len(cc) and key not in seen and x[rr][:, cc].nnz >= min_component_size:
            seen.add(key)
            blocks.append(dict(row_indices=rr, col_indices=cc))
    return blocks


def extract_dense_blocks_nmf(A, *, method='kl_nmf', eta=.5, return_details=False, **kwargs):
    fit = fit_nmf(A, method=method, **kwargs)
    blocks = blocks_from_fit(A, fit, eta)
    return (blocks, fit['details']) if return_details else blocks
