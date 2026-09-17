# Scientific input data

These are the canonical inputs used to generate `public/data/`.

| File | Purpose | SHA-256 |
|---|---|---|
| `all_sparse_dataframes.joblib` | Shared-label sparse mouse × human transport plans | `9057f6a6756ae9e681211bd8b2df5916016f59299cf533f69df04812bea33270` |
| `dict_info_by_coclust.pkl` | Per-rho, per-bicluster gene/LFC/species tables | `9b136297f22eac0007d9f22cc86071caef42bbb0c259a6b4992305595ef0a125` |
| `biclusters_by_rho.pkl` | Per-rho bicluster membership and transport mass | `2c0d0e5e4439800e60ee2e9ff2d5386c630d6909081776d0c3c574dea422b622` |
| `mouse_genes_categories.csv` | Mouse gene categories and corresponding human symbols | `8340dca1a1d9a4b9fe393a171a1280a125e2e76c5a321398b0708a8a1ab1519e` |

Python must have `pyarrow` installed before loading the joblib bundle because its shared labels use pandas Arrow-backed string arrays.

Treat pickle and joblib files as trusted inputs only. Python deserialization can execute code; never load replacements from an untrusted source.
