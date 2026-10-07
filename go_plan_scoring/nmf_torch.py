"""Optional PyTorch NMF engine for CUDA, Apple MPS, and CPU validation.

Dense GPU matrix products are intentional for moderate D1-sized matrices.
Input and factors remain on the device during updates. CUDA objective checks
use float64 reductions. MPS objective checks use CPU float64 reductions because
MPS lacks float64; these transfers occur once per five updates, not each one.
No GPU speed or membership-equivalence claim is made without running the
included validation/benchmark on the target hardware.
"""

from time import perf_counter
import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "GPU NMF requires PyTorch. Install the build appropriate for "
            "your hardware from https://pytorch.org/get-started/locally/"
        ) from exc
    return torch


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _step(torch, x, u, v, beta, a=5.0, b=2.0):
    """Same alternating V, U, beta equations as the NumPy reference."""
    tiny = torch.finfo(u.dtype).tiny
    ratio = x / (u @ v.T).clamp_min(tiny)
    denominator = u.sum(dim=0)[None, :] + (
        v * beta if beta is not None else 0.0
    )
    v.mul_((ratio.T @ u) / denominator.clamp_min(tiny))
    ratio = x / (u @ v.T).clamp_min(tiny)
    denominator = v.sum(dim=0)[None, :] + (
        u * beta if beta is not None else 0.0
    )
    u.mul_((ratio @ v) / denominator.clamp_min(tiny))
    if beta is not None:
        c = (x.shape[0] + x.shape[1]) / 2 + a - 1
        beta.copy_(
            c / (0.5 * (u.square().sum(dim=0) + v.square().sum(dim=0)) + b)
        )


def fit_arrays(
    x,
    initial_u,
    initial_v,
    initial_beta,
    *,
    device="cuda",
    dtype="float64",
    max_iter=500,
    tol=1e-5,
    a=5.0,
    b=2.0,
    target=1.0,
):
    torch = _torch()
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "Supported torch devices are cpu, cuda[:index], and mps"
        )
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is not available in this PyTorch environment"
        )
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if dev.type == "mps" and dtype == "float64":
        raise ValueError(
            "MPS does not support float64; explicitly choose "
            'dtype="float32" and validate memberships'
        )
    tensor_dtype = getattr(torch, dtype)
    converted = x.data.astype(np.float32 if dtype == "float32" else np.float64)
    if not np.isfinite(converted).all() or (converted == 0).any():
        raise FloatingPointError(
            "Requested dtype loses positive input weights; use float64 on "
            "CPU/CUDA"
        )
    nr, nc = x.shape
    k = initial_u.shape[1]
    rows = np.repeat(np.arange(nr), np.diff(x.indptr))
    cols = x.indices
    synchronize(torch, dev)
    started = perf_counter()
    details = dict(
        backend="torch",
        device=str(dev),
        dtype=dtype,
        torch_version=str(torch.__version__),
        device_name=torch.cuda.get_device_name(dev)
        if dev.type == "cuda"
        else str(dev),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        objective_trace=[],
        objective_iterations=[],
        converged=False,
    )
    with torch.no_grad():
        tx = torch.as_tensor(x.toarray(), dtype=tensor_dtype, device=dev)
        u = torch.as_tensor(initial_u, dtype=tensor_dtype, device=dev).clone()
        v = torch.as_tensor(initial_v, dtype=tensor_dtype, device=dev).clone()
        beta = (
            None
            if initial_beta is None
            else torch.as_tensor(
                initial_beta, dtype=tensor_dtype, device=dev
            ).clone()
        )
        rr = torch.as_tensor(rows, dtype=torch.int64, device=dev)
        cc = torch.as_tensor(cols, dtype=torch.int64, device=dev)
        data64 = (
            None
            if dev.type == "mps"
            else torch.as_tensor(x.data, dtype=torch.float64, device=dev)
        )

        def objective():
            prediction = (u @ v.T)[rr, cc]
            c = (nr + nc) / 2 + a - 1
            if dev.type == "mps":
                pred = np.maximum(
                    prediction.cpu().numpy().astype(float),
                    np.finfo(float).tiny,
                )
                uu = u.cpu().numpy().astype(float)
                vv = v.cpu().numpy().astype(float)
                loss = float(
                    np.sum(x.data * (np.log(x.data) - np.log(pred)) - x.data)
                    + uu.sum(0) @ vv.sum(0)
                )
                if beta is not None:
                    bb = beta.cpu().numpy().astype(float)
                    loss += float(
                        bb
                        @ (
                            0.5 * (np.square(uu).sum(0) + np.square(vv).sum(0))
                            + b
                        )
                        - c * np.log(bb).sum()
                    )
                return loss
            pred = prediction.to(torch.float64).clamp_min(np.finfo(float).tiny)
            uu = u.to(torch.float64)
            vv = v.to(torch.float64)
            loss = (
                data64 * (data64.log() - pred.log()) - data64
            ).sum() + uu.sum(dim=0) @ vv.sum(dim=0)
            if beta is not None:
                bb = beta.to(torch.float64)
                loss = (
                    loss
                    + (
                        bb
                        * (
                            0.5
                            * (uu.square().sum(dim=0) + vv.square().sum(dim=0))
                            + b
                        )
                    ).sum()
                    - c * bb.log().sum()
                )
            return float(loss.item())

        previous = objective()
        if not np.isfinite(previous):
            raise FloatingPointError("Nonfinite initial torch NMF objective")
        best = (
            previous,
            u.clone(),
            v.clone(),
            None if beta is None else beta.clone(),
            0,
        )
        details["objective_trace"].append(previous)
        details["objective_iterations"].append(0)
        cutoff = (
            1e-250 if dtype == "float64" else torch.finfo(tensor_dtype).tiny
        )
        for iteration in range(1, max_iter + 1):
            _step(torch, tx, u, v, beta, a, b)
            u.masked_fill_(u < cutoff, 0.0)
            v.masked_fill_(v < cutoff, 0.0)
            if iteration % 5 == 0 or iteration == max_iter:
                value = objective()
                if not np.isfinite(value) or not bool(
                    (torch.isfinite(u).all() & torch.isfinite(v).all()).item()
                ):
                    raise FloatingPointError(
                        "Nonfinite torch NMF fit; try float64 on CPU/CUDA"
                    )
                details["objective_trace"].append(value)
                details["objective_iterations"].append(iteration)
                if value < best[0]:
                    best = (
                        value,
                        u.clone(),
                        v.clone(),
                        None if beta is None else beta.clone(),
                        iteration,
                    )
                offset = (
                    0.0
                    if beta is None
                    else k
                    * ((nr + nc) / 2 + a - 1)
                    * (1 - np.log(((nr + nc) / 2 + a - 1) / b))
                )
                relative = (previous - value) / max(
                    target, abs(previous - offset), np.finfo(float).tiny
                )
                if -1e-10 <= relative <= tol:
                    details["converged"] = True
                    break
                previous = value
        _, u, v, beta, best_iteration = best
        result = dict(
            U=u.cpu().numpy().astype(float),
            V=v.cpu().numpy().astype(float),
            beta=None if beta is None else beta.cpu().numpy().astype(float),
        )
    synchronize(torch, dev)
    mass = result["U"].sum(0) * result["V"].sum(0)
    details.update(
        iterations=iteration,
        best_iteration=best_iteration,
        objective_increases=int(
            np.sum(np.diff(details["objective_trace"]) > 1e-9)
        ),
        active_components=int((mass > target * 1e-12).sum()),
        target_mass=target,
        fitted_mass=float(mass.sum()),
        device_work_seconds=perf_counter() - started,
    )
    result["details"] = details
    return result
