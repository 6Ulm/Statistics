#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$project_dir/scripts/prepare_scientific_data.py" \
  --plans "$project_dir/input_data/all_sparse_dataframes.joblib" \
  --clusters "$project_dir/input_data/dict_info_by_coclust.pkl" \
  --biclusters "$project_dir/input_data/biclusters_by_rho.pkl" \
  --categories "$project_dir/input_data/mouse_genes_categories.csv" \
  --output "$project_dir/public/data"

python "$project_dir/scripts/render_bicluster_plots.py" \
  --data "$project_dir/public/data" \
  --output "$project_dir/public/data/plots"

python -m unittest discover -s "$project_dir/tests" -p 'test_*.py'
