# Notebooks

The analysis is organized as numbered steps. The outputs of each step are the inputs of the next, so the folders
are meant to be read and run in order. Every folder has a README, and every notebook starts with a plain-language
overview.

| Step | Folder | Notebooks | Summary |
|---|---|---|---|
| 0 | [`0.download_and_assemble`](0.download_and_assemble) | 0.1–0.5 | Assemble each source dataset into a table indexed by compound (InChIKey). |
| 1 | [`1.preprocessing`](1.preprocessing) | 1.1–1.3 | Normalize morphological profiles to DMSO negative controls, filter wells for phenotypic activity, and reduce gene expression to 43 latent factors. |
| 2 | [`2.chemical_clustering`](2.chemical_clustering) | 2.1 | Identify compact families of structurally or functionally related compounds. |
| 3 | [`3.knn_neighborhood_analysis`](3.knn_neighborhood_analysis) | 3.1 | Test whether two representations assign the same nearest neighbours to each compound. |
| 4 | [`4.morphology_vs_transcriptomics`](4.morphology_vs_transcriptomics) | 4.1–4.3 | Quantify and interpret the variation that morphological profiles share with transcriptional profiles. |

## Results folders

Every numbered step contains a `results/` folder with an inventory of its outputs. These folders are the standard
location for exported tables and figures:

```text
notebooks/
├── 0.download_and_assemble/results/
├── 1.preprocessing/results/
├── 2.chemical_clustering/results/
├── 3.knn_neighborhood_analysis/results/
└── 4.morphology_vs_transcriptomics/results/
```

Step 0 caches source downloads under the repository-level `data/raw/` directory, keeps its assembled outputs in its
`results/` folder and uploads the same files to `gs://calico-reference-embeddings`. Each `results/README.md` lists
the outputs of its step and separates durable data products from figures, notebook-only tables and temporary
caches. Git ignores raw data and generated results and tracks only the README inventories.

**Before running:** see *Getting started* in the [main README](../README.md) for the environment, Cloud Storage
access and hardware. Open issues are listed in [`docs/code_review.md`](../docs/code_review.md) and, cell by cell, in
[`docs/notebook_review.md`](../docs/notebook_review.md).

**Saved outputs.** Notebooks 1.3, 3.1 and 4.1–4.3 keep the outputs of the runs that produced the project's
results. The other notebooks of Steps 0–2 were saved without outputs. Where a saved output no longer matches the
code, for example after a bug fix, a ⚠️ note above the cell says so.
