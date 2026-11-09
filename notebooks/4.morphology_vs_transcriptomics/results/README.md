# Step 4 results - morphology versus transcriptomics

This folder is the local results location for benchmark tables, figures and compact decoding/enrichment exports.
The notebooks currently retain these analytical results in output cells rather than writing durable result files.

## Benchmark tables

- Linear CKA scores and permutation-null comparisons.
- SVCCA decay summaries and adjusted areas under the decay curve.
- Equal-compound and equal-dimensionality model comparisons from design C in notebook 4.2.
- Counts of shared axes exceeding the permutation baseline, with a configurable noise floor (notebook 4.3).

## Figures

- CKA and SVCCA model-comparison plots.
- SVCCA decay curves and permutation-null overlays.
- Gene-axis volcano plots, alongside morphology-coherence density plots (notebook 4.3).
- Driver-compound similarity and neighbourhood-enrichment plots.

## Decoding and enrichment tables

- Genes positively and negatively associated with shared axes.
- GSEA and over-representation pathway results.
- PubChem/ChEMBL compound annotations.
- Chemical Checker and MiniMol driver-similarity results.
- Driver-neighbourhood enrichment and retrieval metrics.
- Pairwise ECFP4 Tanimoto similarity and morphology coherence of each axis's driver compounds (notebook 4.3).
- Axis purity sweep (loading kurtosis, Hoyer sparsity, pan-stress contamination) and microtubule gene-module / known-drug
  axis scans (notebook 4.3).

## Files that do not belong here

Files such as `cached_*.parquet`, `raw_drugseq_temp.parquet` and `jump_metadata.duckdb` are re-downloadable runtime
caches, not analytical results. They remain covered by the repository-level `.gitignore` and should not be moved
into this folder.

Before publishing exported results, re-run each notebook from top to bottom and review the saved-output mismatches
and statistical limitations documented in [`../../../docs/code_review.md`](../../../docs/code_review.md).
