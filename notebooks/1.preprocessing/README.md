# Step 1: Preprocessing

**Goal:** make the profiles comparable across experiments and reduce noise before the analyses.

* **Normalization.** Morphological profiles vary with the plate and the experimental batch in addition to the
  perturbation (technical variation, or batch effects). Normalization expresses each well relative to negative-control wells
  treated only with DMSO, the compound solvent. Notebook 1.1 computes this DMSO reference once over the whole
  dataset (see *Things to know*).
* **Phenotypic activity filtering.** Many compounds produce no detectable morphological change at the tested dose.
  Their profiles resemble DMSO, and two such compounds appear similar only because both resemble untreated cells.
  Filtering keeps the wells whose profiles differ from the negative controls.
* **Dimensionality reduction of gene expression.** Transcriptional profiles contain thousands of correlated, noisy
  genes. Factor analysis summarizes them as a few dozen latent factors, each corresponding to a co-regulated gene
  program.

## Notebooks

| Notebook | Processing | Input → output |
|---|---|---|
| [`1.1.normalize_openphenom`](1.1.normalize_openphenom.ipynb) | Robust z-scoring to DMSO (median and median absolute deviation), followed by sphering (ZCA whitening) so that features are uncorrelated among control wells | `openphenom_{384,1920}.parquet` → `openphenom_{384,1920}_sphered_mad_robustized.parquet` |
| [`1.2.filter_inactive_compounds`](1.2.filter_inactive_compounds.ipynb) | Keeps wells whose distance from the DMSO centroid exceeds that of 95% of DMSO wells (*phenotypic activity score*); the filter acts on single wells, so each model's file contains its own set of compounds | normalized DINO and OpenPhenom → `...-ACTIVE.parquet`, `...-actives_only.parquet` |
| [`1.3.normalize_drugseq`](1.3.normalize_drugseq.ipynb) | Summarizes the DRUG-seq dataset (4 doses, a single 24 h time point); selects the 2,000 most variable genes, applies an asinh transform, sets the number of factors with Horn's parallel analysis (43) and fits factor analysis with varimax rotation | `novartis_drugseq_u2os_10uM_24h.parquet` → `..._normalized_factors.parquet` |

Run 1.1 before 1.2. Notebook 1.3 is independent of both.

## Results

See [`results/README.md`](results/README.md) for the output inventory. Normalized and filtered tables are written to
Google Cloud Storage. The local `results/` folder holds small diagnostics and validation summaries; the large Parquet
products stay in the bucket.

## Methods in plain language

* **Robust z-scoring (MAD normalization).** For each feature, the median of the DMSO wells is subtracted and the
  result is divided by their median absolute deviation. A value of 3 means three robust standard deviations away
  from untreated cells. Medians are insensitive to outlier wells.
* **Sphering.** The feature space is rotated and rescaled so that, among control wells, features are uncorrelated
  with equal variance. Euclidean distance from DMSO then measures the strength of a compound's effect, with every
  independent direction weighted equally.
* **Horn's parallel analysis.** The number of factors is the number of components that explain more variance in the
  real data than in data where each gene was shuffled independently across compounds. Shuffling preserves each
  gene's variance and removes co-regulation between genes.

## Things to know

* **⚠️ Normalization version.** The original repository reports that normalization errors were found on 09/23/2026,
  that all morphological datasets were re-normalized (`*_0922_normed.parquet`) and that the earlier files were
  deprecated. The re-normalization code is in neither repository. Notebook 1.1 contains the earlier method. Its
  outputs are still in the bucket and are read by Steps 1.2, 3, 4.1 and 4.2. Step 4.3 reads the re-normalized
  files.
* **⚠️ Plate-level effects remain.** Notebook 1.1 estimates the DMSO median, the median absolute deviation and the
  sphering matrix once over all DMSO wells and applies the same transform to every well. A global transform cannot
  remove plate or batch offsets. These offsets therefore remain in the normalized files and influence the activity
  calls of 1.2. Standard practice in image-based profiling normalizes each plate to its own negative controls. The
  Bayer DINO normalized embeddings come from the authors' own pipeline, so DINO and OpenPhenom profiles were
  normalized by different procedures.
* **⚠️ Provenance.** The DINO active-only file used downstream (`..._normed_final-actives_only.parquet`) comes from
  a different run than the saved code, which writes `...ACTIVE_ONLY-test2.parquet`. The 10 µM / 24 h DRUG-seq subset
  read by 1.3 was extracted outside this repository.
* **Dose selection.** The 43 factors use only the 10 µM profiles. Later gene-level analyses combine all four doses.
* **No saved outputs** in 1.1 and 1.2. Both write directly to the shared bucket.
* Notebooks 1.1 and 1.2 were written for Google Colab. Notebook 1.3 runs on a laptop CPU, and a GPU shortens it.
* Details and open issues: [`docs/code_review.md`](../../docs/code_review.md) and
  [`docs/notebook_review.md`](../../docs/notebook_review.md).
