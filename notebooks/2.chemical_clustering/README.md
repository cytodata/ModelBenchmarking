# Step 2: Compound clustering

**Goal:** identify compact families of closely related compounds. Step 3 uses these families to ask a focused
question: when compounds clearly belong to one structural or functional family, do image-based models place the
family members close together?

## Notebook

[`2.1.cluster_chemical_embeddings`](2.1.cluster_chemical_embeddings.ipynb) applies the same procedure to two
representations:

| Part | Representation | Input | Output |
|---|---|---|---|
| A | MiniMol (chemical structure) | `minimol_embeddings__Kim_et_al_subset_final.parquet` (Step 0.2) | `minimol_clusters.csv` |
| B | Chemical Checker sign3 (bioactivity) | `JUMPCP_Sign3_CC_final.parquet` (Step 0.1) | `sign3_clusters.csv` |

* **UMAP** projects the embeddings to two dimensions for visualization only.
* **HDBSCAN** is a density-based clustering method. It groups compounds that lie in regions denser than their
  surroundings and labels the remaining compounds as *noise* (cluster −1). The number of clusters is not fixed in
  advance, and the chosen settings favour small, compact clusters.
* A **hierarchical-clustering heatmap** of the clustered compounds, computed on 20 principal components, provides a
  visual check of cluster coherence.

Each output CSV contains one row per compound with its cluster label, UMAP coordinates and metadata. Step 3 keeps
only the clustered compounds: 749 for MiniMol and 1,436 for sign3. About 92% of compounds are labelled noise.

## Things to know

* Clustering is performed on the full 512- and 3,200-dimensional vectors.
* The sign3 signatures include the five Chemical Checker chemistry spaces (A1–A5), so sign3 clusters partly reflect
  chemical structure.
* Features are selected as the numeric columns minus a fixed list of metadata names.
* The notebook was saved without outputs.

Details are in [`docs/notebook_review.md`](../../docs/notebook_review.md).

## Results

See [`results/README.md`](results/README.md) for the output inventory. Cluster-assignment CSVs are written to Google
Cloud Storage. The local `results/` folder holds exported cluster plots, diagnostic tables and small summaries.

## Requirements

This notebook requires an **NVIDIA GPU with [RAPIDS cuML](https://docs.rapids.ai/install/)**, which provides GPU
implementations of UMAP, HDBSCAN and PCA. cuML is available only on Linux with NVIDIA GPUs and is therefore not
listed in `pyproject.toml`. Installation options:

* **Google Colab (GPU runtime):** uncomment the `!pip install cuml-cu12 ...` line at the top of each cell.
* **Linux with an NVIDIA GPU, using uv:** `uv pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12`
  (choose the package that matches your CUDA version).
