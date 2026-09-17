#!/usr/bin/env python3
"""Convert the Python analysis objects into audited, browser-friendly JSON assets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


def finite_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def clean_gene(value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("Empty gene symbol")
    return result


def compact_float32(values: np.ndarray) -> list[float]:
    """Serialize float32 values without Python's expanded binary round-off tails."""
    return [float(f"{float(value):.8g}") for value in values]


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    required = {"genes", "LFC", "is_mouse", "is_human"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cluster frame is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "gene": clean_gene(row.genes),
                "lfc": finite_float(row.LFC),
                "isMouse": bool(row.is_mouse),
                "isHuman": bool(row.is_human),
            }
        )
    return records


def bicluster_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    required = {
        "mouse_genes",
        "human_genes",
        "transport_mass",
        "n_mouse",
        "n_human",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Bicluster frame is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        mouse_genes = [clean_gene(gene) for gene in row.mouse_genes]
        human_genes = [clean_gene(gene) for gene in row.human_genes]
        if len(mouse_genes) != int(row.n_mouse) or len(human_genes) != int(row.n_human):
            raise ValueError("Bicluster has inconsistent gene counts")
        records.append(
            {
                "mouseGenes": mouse_genes,
                "humanGenes": human_genes,
                "transportMass": finite_float(row.transport_mass),
                "nMouse": int(row.n_mouse),
                "nHuman": int(row.n_human),
            }
        )
    return records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--biclusters", required=True, type=Path)
    parser.add_argument("--categories", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = joblib.load(args.plans)
    with args.clusters.open("rb") as handle:
        cluster_info = pickle.load(handle)
    with args.biclusters.open("rb") as handle:
        biclusters = pickle.load(handle)

    matrices = bundle["matrices"]
    keys = list(bundle.get("keys", matrices.keys()))
    if set(keys) != set(matrices) or set(keys) != set(cluster_info) or set(keys) != set(biclusters):
        raise ValueError("Rho keys do not align across the three scientific datasets")
    keys = [rho for rho in keys if float(rho) < 0.1]

    row_names = [clean_gene(value) for value in bundle["index"]]
    column_names = [clean_gene(value) for value in bundle["columns"]]
    shape = [len(row_names), len(column_names)]
    if tuple(shape) != tuple(bundle["shape"]):
        raise ValueError("Shared labels do not match the declared matrix shape")
    if len(row_names) != len(set(row_names)) or len(column_names) != len(set(column_names)):
        raise ValueError("Transport-plan labels must be unique")

    categories = pd.read_csv(args.categories)
    required_categories = {"genes", "Categorie", "human_gene"}
    missing_categories = required_categories.difference(categories.columns)
    if missing_categories:
        raise ValueError(f"Category file is missing columns: {sorted(missing_categories)}")
    category_records = [
        {
            "gene": clean_gene(row.genes),
            "category": str(row.Categorie).strip(),
            "humanGene": clean_gene(row.human_gene),
        }
        for row in categories.itertuples(index=False)
    ]

    manifest: dict[str, object] = {
        "format": "coclust-browser-v1",
        "shape": shape,
        "rowNames": row_names,
        "columnNames": column_names,
        "categories": category_records,
        "rhos": [],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for position, rho in enumerate(keys):
        matrix = matrices[rho]
        if not sparse.isspmatrix_csr(matrix):
            matrix = sparse.csr_matrix(matrix)
        matrix = matrix.astype(np.float32, copy=False)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.shape != tuple(shape):
            raise ValueError(f"Matrix shape mismatch for rho={rho!r}")

        rho_clusters = cluster_info[rho]
        cluster_payload = {
            str(int(cluster_id)): frame_records(frame)
            for cluster_id, frame in sorted(rho_clusters.items(), key=lambda item: int(item[0]))
        }
        bicluster_payload = sorted(
            bicluster_records(biclusters[rho]),
            key=lambda record: float(record["transportMass"]),
            reverse=True,
        )
        cluster_ids = [int(value) for value in cluster_payload]
        if len(cluster_ids) != len(bicluster_payload):
            raise ValueError(
                f"Cluster counts do not align for rho={rho!r}: "
                f"info={len(cluster_ids)}, biclusters={len(bicluster_payload)}"
            )
        for rank, record in enumerate(bicluster_payload):
            record["cluster"] = rank

        filename = f"rho-{position:02d}.json"
        payload = {
            "rho": finite_float(rho),
            "matrix": {
                "indptr": matrix.indptr.astype(np.int32).tolist(),
                "indices": matrix.indices.astype(np.int32).tolist(),
                "data": compact_float32(matrix.data),
            },
            "clusters": cluster_payload,
            "biclusters": bicluster_payload,
        }
        write_json(args.output / filename, payload)
        manifest["rhos"].append(
            {
                "value": finite_float(rho),
                "label": f"{float(rho):.12g}",
                "mass": finite_float(matrix.sum()),
                "file": filename,
                "clusterIds": cluster_ids,
                "nnz": int(matrix.nnz),
            }
        )

    write_json(args.output / "manifest.json", manifest)
    print(f"Prepared {len(keys)} rho datasets in {args.output}")


if __name__ == "__main__":
    main()
