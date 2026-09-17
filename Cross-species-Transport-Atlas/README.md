# Cross-species Transport Atlas

Interactive exploration of mouse-to-human optimal-transport plans, bicluster-specific STRING physical networks, connected network components, and category-annotated biclusters.

## Agent handoff: start here

This repository is self-contained for continuation by Claude, Codex, or another engineering agent. It includes:

- the complete Next.js/Vinext application source;
- all four canonical scientific input files in `input_data/`;
- the already-generated browser assets in `public/data/`;
- deterministic data-preparation and category-plot scripts;
- the package lockfile and Python dependency list;
- validation tests; and
- the Sites project identity in `.openai/hosting.json`.

Do **not** collapse these distinct concepts:

| Identifier | Meaning |
|---|---|
| rho | Internal key of a complete sparse transport plan |
| mass | User-facing selector label: sum of the complete plan, rounded to 2 decimals |
| bicluster `B` | One gene list within a rho dataset and the unit submitted independently to STRING |
| STRING cluster `C` | One connected component inside a bicluster's filtered physical network |

The most important invariant is: **one heatmap represents the whole filtered physical network for one rho + bicluster**. STRING connected components only group/order/color the ticks and define interactive highlight blocks. They must never replace the whole heatmap with one component-specific sub-plan.

### Fastest reproducible setup

```bash
# JavaScript application
corepack enable
pnpm install --frozen-lockfile
cp .env.example .env.local
# Edit .env.local and choose ACCESS_CODE.

# Python scientific asset pipeline (use a virtual environment)
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt

# Optional: regenerate all browser JSON and category-plot SVG assets
./scripts/rebuild_scientific_assets.sh

# Validate and run
python -m unittest discover -s tests -p 'test_*.py'
pnpm exec tsc --noEmit
pnpm lint
pnpm build
pnpm dev
```

The committed `public/data/` is already usable, so Python regeneration is unnecessary when changing only the interface. Regenerate it when input data, cutoff rules, serialization, or category-plot formatting changes.

### Current product behavior

- The selector displays transport **mass**, while rho remains the exact internal/cache key.
- Only `rho < 0.1` is exposed, producing 17 selectable transport scales.
- STRING is run independently for each selected bicluster.
- Physical-network evidence is limited to experiments and databases with combined confidence ≥ 700.
- Isolates are removed; components require at least 2 nodes and 1 edge.
- The heatmap always retains the complete bicluster-specific filtered-network sub-plan.
- Hovering a `B · C` group previews its block; clicking/tapping pins the highlight without hiding the global plan.
- The physical-network graph uses fixed coordinates, small nodes, pan/zoom, and individual-node dragging without global reflow.
- The category plot uses horizontal, category-coloured labels for both mouse and human genes.
- The hosted Site is public at the platform level but application content requires the server-side `ACCESS_CODE`.

### Repository map

| Path | Role |
|---|---|
| `app/page.tsx` | Main client application, live STRING workflow, connected components, heatmap, graph, CSV exports |
| `app/layout.tsx` | Server-side access gate decision |
| `app/api/access/route.ts` | Access-code verification and session cookie |
| `app/api/string/route.ts` | Session-protected, validated STRING proxy |
| `components/access-gate.tsx` | Public access-code screen |
| `components/echart.tsx` | ECharts lifecycle and zoom preservation |
| `components/bicluster-plot.tsx` | Per-rho category SVG display |
| `lib/access.ts` | Server-only code/session helpers |
| `input_data/` | Canonical original scientific inputs and checksums |
| `public/data/` | Generated compact JSON plus category-plot SVGs consumed by the browser |
| `scripts/prepare_scientific_data.py` | Validates/converts Python inputs to browser JSON |
| `scripts/render_bicluster_plots.py` | Rebuilds category-bicluster SVGs |
| `scripts/rebuild_scientific_assets.sh` | One-command data + plot regeneration + Python validation |
| `tests/test_scientific_data.py` | Manifest, sparse-array, bicluster, and cutoff consistency checks |
| `.openai/hosting.json` | Existing Sites project identity; do not replace with a new project ID |

### Safe continuation rules

1. Preserve unrelated scientific data and the existing Sites project ID.
2. Keep secrets out of Git and ZIP files. Set `ACCESS_CODE` in `.env.local` locally and as a secret hosting environment variable in production.
3. Never load untrusted pickle/joblib files.
4. After changing data preparation, run the rebuild script and commit both source and regenerated `public/data/` assets.
5. After changing category-plot layout, regenerate all 17 SVGs.
6. After changing heatmap logic, verify that the displayed plan still contains all nonzero human columns linked to every retained mouse node in the complete filtered bicluster network.
7. Preserve the graph's `layout: 'none'`; a force layout causes the unwanted whole-network movement previously rejected.
8. Before publishing, run Python tests, TypeScript, lint, and the production build.

## Scientific hierarchy

The application keeps these three identifiers separate:

1. **rho (`ρ`)** selects one sparse mouse × human transport plan.
2. **bicluster (`B`)** selects one gene list inside that rho dataset.
3. **STRING cluster (`C`)** is a connected component found after filtering the physical network generated independently for that bicluster.

The processing hierarchy is therefore:

```text
rho
└── bicluster gene list
    └── filtered STRING physical network
        ├── connected cluster 1
        ├── connected cluster 2
        └── ...
```

A bicluster can produce zero, one, or many STRING clusters. A cluster number is local to its bicluster and is displayed as `B<bicluster> · C<cluster>`.

## Input data

The source analysis is included under `input_data/` and consists of:

- `all_sparse_dataframes.joblib`: sparse transport plans with shared row and column labels;
- `dict_info_by_coclust.pkl`: per-rho, per-bicluster gene tables;
- `biclusters_by_rho.pkl`: per-rho bicluster membership and transport mass;
- `mouse_genes_categories.csv`: category annotations used by the category-bicluster plot.

All transport plans must have identical dimensions and identical row/column labels. The source dataset contains 22 rho values and matrices of 2,459 mouse genes × 1,881 human genes. The browser excludes every `rho >= 0.1`, leaving 17 displayed transport scales.

### Transport-plan bundle

The joblib object is expected to contain:

```python
{
    "keys": [...],
    "shape": (n_mouse, n_human),
    "index": [...],       # mouse gene symbols
    "columns": [...],     # human gene symbols
    "matrices": {rho: scipy.sparse.csr_matrix, ...},
}
```

Matrices are converted to CSR `float32`, explicit zeros are eliminated, column indices are sorted, and the following arrays are serialized:

- `indptr`: row offsets;
- `indices`: nonzero column indices;
- `data`: nonzero transport values.

### Bicluster gene information

For each rho and bicluster, the dataframe in `dict_info_by_coclust.pkl` must contain:

- `genes`;
- `LFC`;
- `is_mouse`;
- `is_human`.

Only the `genes` column is sent to STRING. The other columns remain in the app and are used to encode graph nodes.

### Bicluster metadata

Each dataframe in `biclusters_by_rho.pkl` must contain:

- `mouse_genes`;
- `human_genes`;
- `transport_mass`;
- `n_mouse`;
- `n_human`.

The source columns `row_cluster` and `column_cluster` are intentionally ignored. Biclusters are ordered by decreasing `transport_mass`, and their rank becomes the browser-facing bicluster ID.

### Category annotations

`mouse_genes_categories.csv` must contain:

- `genes`;
- `Categorie`;
- `human_gene`.

Category order follows the CSV. A gene may occur in multiple categories.

## Browser data preparation

Prepare the compact JSON assets with:

```bash
python scripts/prepare_scientific_data.py \
  --plans input_data/all_sparse_dataframes.joblib \
  --clusters input_data/dict_info_by_coclust.pkl \
  --biclusters input_data/biclusters_by_rho.pkl \
  --categories input_data/mouse_genes_categories.csv \
  --output public/data
```

This produces `manifest.json` and one `rho-XX.json` asset per retained rho. The preparation script validates key alignment, shared dimensions, unique gene labels, required dataframe columns, finite values, and bicluster gene counts. `input_data/README.md` records the SHA-256 digest of every bundled input.

The UI selector is labeled by total transport mass rather than rho. For each retained rho, the manifest stores the equivalent of:

```python
mass = dict_coclust[rho]["res_benchmark"]["plan"].values.sum()
```

In the prepared sparse-plan bundle this is computed as `matrix.sum()`. Selector labels use `np.round(mass, 2)` semantics (two fixed decimal places), while the unrounded rho remains the internal dataset/cache key. Rho values greater than or equal to `0.1` are omitted from the manifest and therefore cannot be selected in the app.

## STRING physical-network procedure

The network procedure is run independently for every selected rho and bicluster.

### Fixed settings

| Setting | Value |
|---|---|
| Species | *Mus musculus*, NCBI taxon `10090` |
| Network type | Physical |
| Active evidence sources | Experiments and databases only |
| Minimum confidence | `700` on STRING's 0–1000 scale |
| Component definition | Undirected connected components |
| Minimum component size | 2 nodes |
| Minimum internal edges | 1 edge |
| Isolates | Removed |

The user can choose STRING 12.0 stable or the 12.5 preview endpoint. Transfer evidence is disabled because it is not one of the selected sources.

### Mapping and edge retrieval

For one bicluster:

1. Deduplicate its gene symbols.
2. Resolve symbols with STRING `get_string_ids`.
3. Keep one record per distinct STRING protein ID.
4. Fetch the physical network with `required_score=700`, `network_type=physical`, and `add_nodes=0`.
5. Recalculate confidence from the experiment (`escore`) and database (`dscore`) channels only.
6. Remove self-loops.
7. Deduplicate undirected protein pairs, keeping the largest qualifying score.
8. Remove every mapped protein that has no retained edge.
9. Find connected components in the retained graph.
10. Rank components by decreasing node count, then decreasing internal-edge count, then the first STRING ID.
11. Number components from 1 inside each bicluster.

### Source-specific confidence

STRING channel scores contain a prior probability of `p = 0.041`. For every available selected score `s`, the app removes that prior:

```text
adjusted(s) = max(0, (s - p) / (1 - p))
```

The selected channels are combined as independent evidence and the prior is restored:

```text
combined_without_prior = 1 - product(1 - adjusted(s))
combined = combined_without_prior + p * (1 - combined_without_prior)
```

An edge is retained only when `combined × 1000 >= 700`.

### API safeguards

`app/api/string/route.ts` proxies the allowed STRING calls and enforces:

- POST requests;
- 2–2,000 identifiers;
- numeric species IDs;
- confidence between 0 and 1,000;
- physical or functional network types from an allowlist;
- evidence-source allowlists;
- a 500 KB request-body limit;
- a 90-second upstream timeout.

Client requests are spaced by at least one second. Network results are cached only in browser memory for the current session, keyed by rho, bicluster, and STRING release.

## Transport heatmap

The heatmap unit is **one rho + one bicluster**. Connected components do not create separate heatmaps.

For the chosen bicluster:

1. Take the union of all non-isolated STRING nodes across every retained connected component.
2. Keep the nodes marked as mouse genes.
3. Match those symbols to rows of the rho-specific sparse transport plan, with an exact match followed by a case-insensitive fallback.
4. Extract one sub-plan containing every matched mouse row and every human column with at least one nonzero value in those rows.
5. Remove all-zero rows and columns.
6. Order mouse rows by STRING cluster and then gene symbol.
7. Display every human column once. If it receives transport from several STRING clusters, assign it to the cluster contributing the largest summed absolute transport mass; ties use the lower cluster number.
8. Order human columns by that assigned cluster and then gene symbol.

Thus the heatmap contains the **whole filtered physical network for the bicluster**, while connected components are used only to group and color its axis ticks.

### Heatmap encoding

- Cells: transport values, using a white-to-dark-blue sequential scale.
- Y-axis: mouse genes, grouped and colored by their STRING component.
- X-axis: corresponding human genes, grouped and colored by their dominant transported STRING component.
- Tooltips: bicluster, mouse gene and component, human gene and assigned tick group, and transport value.
- Navigation: wheel/pinch zoom, drag zoom, and horizontal/vertical slider controls.
- Selector: one bicluster at a time by default; an “all selected biclusters” view concatenates complete bicluster sub-plans without merging their identities.

### Interactive cluster focus

A physical-network group strip above the heatmap lists every retained `B · C` connected component and its mouse-row × assigned-human-column dimensions. Hovering or keyboard-focusing a group previews its block. Clicking or tapping pins the group until it is clicked again or another group is selected. In both cases the app:

- keeps the complete rho/bicluster sub-plan visible;
- draws a component-colored boundary around that component's row × column block;
- keeps transport cells inside the block at full opacity and dims cells outside it;
- emphasizes the corresponding mouse and human tick labels while muting the other groups; and
- preserves the user's current horizontal and vertical zoom positions.

The highlighted state never extracts, replaces, or filters the heatmap: the whole filtered physical-network sub-plan remains plotted. Human genes in a highlighted block are those assigned to that component by the dominant transported-mass rule described above.

## Physical-network graph

The graph can display all retained biclusters/components or focus on one `B · C` component.

### Node encoding

| Gene type | Shape |
|---|---|
| `is_mouse=1`, `is_human=0` | Circle |
| `is_mouse=1`, `is_human=1` | Diamond |
| `is_mouse=0`, `is_human=1` | Triangle |

Node color represents LFC with a blue–white–red diverging scale centered on zero. Labels show the preferred STRING gene name. Tooltips include bicluster, component, LFC, and species flags.

Edges retain the source-specific confidence score and are colored by their `B · C` component.

### Stable interactive layout

The graph uses deterministic, precomputed coordinates rather than a live force simulation. Components are placed on a grid; nodes within a component use a golden-angle spiral ordered by degree and gene name. Users can pan, zoom, and drag an individual node without causing the whole graph to reflow.

## Category bicluster plot

The third view reproduces the supplied Matplotlib reference plot for every rho. Generate the SVG assets with:

```bash
python scripts/render_bicluster_plots.py \
  --data public/data \
  --output public/data/plots
```

The plot includes:

- biclusters ordered by decreasing transport mass;
- black diagonal blocks annotated with bicluster, mass, and dimensions;
- omission of biclusters containing no gene of interest;
- category-colored mouse and human gene labels;
- exact gene-position ticks, with both mouse and human gene names laid out horizontally in category-sorted groups;
- cluster brackets and connector lines;
- multi-category square markers;
- cross-cluster dagger markers;
- category counts, missing-gene notes, and small-block notes.

This view is independent of the live STRING filtering procedure.

## Exports

The physical-network view exports two CSV files.

### `df_edges.csv`

```text
source,target,source_gene,target_gene,score,bicluster,cluster
```

### `df_nodes.csv`

```text
string_id,gene,query_gene,LFC,is_mouse,is_human,bicluster,cluster
```

`bicluster` identifies the input gene set. `cluster` identifies the connected component inside that bicluster's filtered physical network.

## Application architecture

- `app/page.tsx`: client state, STRING workflow, connected components, heatmap construction, graph construction, and exports.
- `app/api/string/route.ts`: validated server-side STRING proxy.
- `app/api/access/route.ts`: access-code verification and HTTP-only session cookie issuance.
- `components/access-gate.tsx`: public-link access-code screen.
- `lib/access.ts`: server-only access-code and session-token helpers.
- `components/echart.tsx`: ECharts lifecycle wrapper.
- `components/bicluster-plot.tsx`: exact per-rho reference-plot display.
- `scripts/prepare_scientific_data.py`: conversion and validation of scientific inputs.
- `scripts/render_bicluster_plots.py`: Matplotlib category-bicluster renderer.
- `public/data/manifest.json`: shared labels, category metadata, and rho index.
- `public/data/rho-XX.json`: one sparse plan plus bicluster data for a rho.
- `public/data/plots/`: pre-rendered category-bicluster SVGs.

The frontend uses React 19, TypeScript, Next.js/Vinext, ECharts 6, and the existing UI primitives. The hosted server output targets Cloudflare Workers through Vinext.

## Public link with access code

The Site itself is public, so anyone can open its URL. Application content is rendered only after the visitor submits the correct access code. The code is stored as the secret production environment variable `ACCESS_CODE`; it is never committed to the source repository or sent to the browser. A successful submission creates a SHA-256-derived, HTTP-only, `Secure`, `SameSite=Lax` cookie valid for 30 days. The STRING proxy independently validates that session cookie before accepting requests.

## Local development

Requirements:

- Node.js 22.13 or newer;
- pnpm 11;
- Python 3.10+ when regenerating data or plots;
- the packages in `requirements-data.txt`, including `pyarrow` for the Arrow-backed pandas labels stored in the joblib bundle.

Install and run:

```bash
pnpm install
cp .env.example .env.local
# Set ACCESS_CODE in .env.local.
pnpm dev
```

Validation and production build:

```bash
pnpm lint
pnpm exec tsc --noEmit
pnpm build
python -m unittest discover -s tests -p 'test_*.py'
```

### Complete scientific rebuild

```bash
python -m pip install -r requirements-data.txt
./scripts/rebuild_scientific_assets.sh
```

The helper regenerates JSON, regenerates all category SVGs, and runs the scientific-data tests. Its outputs are deterministic apart from differences that may arise from major Matplotlib/font-rendering changes.

### Hosting

This is an existing Sites project. Reuse the `project_id` already recorded in `.openai/hosting.json`; creating another Site would produce the wrong URL and split deployment history. Configure `ACCESS_CODE` as a secret runtime variable, deploy the saved source/build version, and set the Site audience to `public`. The application gate—not the platform viewer list—then controls entry by code.

The access code itself is intentionally absent from source and from this archive. Whoever deploys the project must choose/provide it through environment configuration.

## Reproducibility and limitations

- STRING 12.5 preview data may change; STRING 12.0 is the reproducible stable option exposed by the app.
- Gene-symbol mapping uses STRING's best match and may be ambiguous. The original query symbol is retained in exports.
- A STRING ID produced from multiple query symbols is represented once.
- LFC and species flags are matched first by exact preferred/query symbol and then case-insensitively; unmatched values fall back to LFC `0`, mouse `true`, human `false`.
- Human heatmap tick groups are visualization assignments based on dominant transported mass; they are not additional STRING clusters.
- Network state is not persisted server-side. Reloading the page requires rebuilding the requested networks.
- Only gene identifiers are sent to STRING. Transport values, LFC values, and category annotations are not transmitted.
- The access-code gate protects rendered application content and the STRING proxy. Files under `public/`, including prepared scientific JSON, remain ordinary public static assets on a public deployment; do not treat this mechanism as protection for confidential source data.
- Live STRING results depend on upstream identifier mapping and availability. Browser-session caching avoids repeated requests only until reload.
