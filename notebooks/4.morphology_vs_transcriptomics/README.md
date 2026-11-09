# Step 4: Morphology vs. transcriptomics

**Goal:** rank the image-based (Cell Painting) models by the variation their morphological profiles share with
transcriptional profiles (DRUG-seq) of the same compounds, and characterize the biology of that shared variation.

**Rationale.** A compound alters both cell morphology and gene expression. A model whose embeddings capture
biologically meaningful variation should share structure with the transcriptional response to the same compounds.
This agreement can be measured without manual annotation.

## Methods in plain language

* **CKA** (linear centered kernel alignment) summarizes in one number, between 0 and 1, whether compounds with
  similar morphological profiles also have similar transcriptional profiles.
* **SVCCA** (singular vector canonical correlation analysis) first reduces each dataset to its principal components.
  It then finds pairs of directions, one per dataset, with maximal correlation. Each pair is a *canonical axis* of
  shared variation. Sorting the canonical correlations from strongest to weakest gives a *decay curve*; more and
  stronger axes indicate more shared information.
* **Permutation null.** The analysis is repeated after shuffling which transcriptional profile belongs to which
  compound. Results count as significant only when they exceed this chance level.
* **Axis interpretation.** For the top-ranked model, genes are correlated with each canonical axis and tested for
  pathway enrichment (GSEA). The compounds with the highest scores on an axis (*driver compounds*) are annotated
  through PubChem and ChEMBL and tested for similarity in bioactivity (Chemical Checker), chemical structure (MiniMol
  or ECFP4 fingerprints) and morphology.
* **Axis rotation** (4.3). Varimax or FastICA re-express the canonical axes as more separable components. A purity
  analysis then checks whether each component depends on a small set of genes and whether it reflects a generic
  stress response.
* **Hypothesis-driven axis tests** (4.3). Axes are scored against a predefined gene module (microtubule and
  mitotic-arrest genes) and against known tubulin-targeting drugs.

## Notebooks

| Notebook | Transcriptional reference | Main features |
|---|---|---|
| [`4.1.morphology_vs_drugseq_genes_v2`](4.1.morphology_vs_drugseq_genes_v2.ipynb) | 14,370 genes; the four doses combined by max-absolute pooling (single 24 h time point; the "24 h/48 h" filter in the code keeps every row) | First version: benchmark of 8 models; interpretation with Chemical Checker, MiniMol and GO/Hallmark gene sets; neighbourhood analysis of driver compounds |
| [`4.2.morphology_vs_drugseq_factors_v3`](4.2.morphology_vs_drugseq_factors_v3.ipynb) | 43 latent factors from Step 1.3 (10 µM, 24 h) | Second version: a permutation null for each model; three benchmark designs, the last with identical compounds (N = 523) and equal dimensionality for every model; axes fitted on factors and interpreted with genes |
| [`4.3.morphology_vs_drugseq_anchors_v4`](4.3.morphology_vs_drugseq_anchors_v4.ipynb) | Selectable reference: genes, 43 factors or ChemPerturb-Bridge latent embeddings (saved run: latent embeddings) | Latest version: configurable PCA, compound cohort and significance threshold; Varimax and FastICA axis rotation; structural (ECFP4) and morphological checks of driver compounds; axis purity analysis; microtubule hypothesis tests |

**For model comparisons, use 4.2 (design C) or 4.3 with `COHORT_STRATEGY = "common"`.** Notebook 4.1 is kept for its
gene-level benchmark and JUMP metadata queries. The MiniMol-based interpretation and the driver-neighbourhood
analysis (Parts 4–5) exist in 4.1 and 4.2 only.

**Inputs:** morphological profiles from Steps 0.3, 0.4 and 1.1, plus CP-CNN and ImageNet embeddings prepared
elsewhere (4.3 reads harmonized copies in `indexed_by_inchikey`); DRUG-seq factors (Step 1.3) and signatures;
ChemPerturb-Bridge latent embeddings (4.3); Chemical Checker sign3 and MiniMol for all JUMP compounds; standardized
SMILES (4.3).
**Hardware:** a GPU is recommended (the saved runs used an NVIDIA V100); every step also runs on a CPU.
**Web services:** MyGene.info, Enrichr, PubChem and ChEMBL; 4.1 also downloads the JUMP metadata database from
GitHub.
**Environment:** the notebooks use the `model_benchmarking` package, so run them in the uv environment.
**Execution order:** run each notebook from top to bottom, because later parts reuse variables and cached files.

## Results

See [`results/README.md`](results/README.md) for the inventory of benchmark, interpretation and enrichment outputs.
Figures and result tables are currently kept inside the notebooks. Exported figures, tables and compact summaries
belong in this step's `results/` folder. Downloaded Parquet caches are temporary runtime files.

## ⚠️ Read before interpreting

* **Saved outputs and code.** Some saved outputs come from sessions with different settings; each case is marked in
  the notebook. Re-run the notebooks from top to bottom before publishing figures.
* **Model ranking is unresolved.** Differences between the top-ranked models are smaller than the changes caused by
  the benchmark design or the similarity metric, and no confidence intervals are reported.
* **Pilot analyses (Part 1 of 4.1 and 4.2).** Each well is paired with the compound-level DRUG-seq profile, so
  replicate wells repeat the same transcriptional profile. This inflates CKA and the number of significant axes.
  Use Part 2 for model comparisons.
* **Number of axes per model.** With `COHORT_STRATEGY = "pairwise"` (4.3) and in designs A–B of 4.2, the score
  averages over a different number of canonical axes for each model; design C fixes this.
* **Driver selection.** Driver compounds are selected from axes fitted on morphology and expression, so their
  clustering in those two spaces is partly expected by construction.
* **Execution-order problems.** In 4.1, Part 4 uses the 5,000 most variable genes when the notebook runs in order,
  because Part 3 redefines `drugseq_features`. In 4.2, Parts 4–5 are copies of 4.1 and were not adapted to the
  factor-based pipeline; Part 4 stops with a `KeyError` because it reads the 4.1 column `SVCCA Mean (AUC)`.
* **⚠️ 4.3 reference data.** The reference of the saved run (`cpb_observed_molecules_lpm_latents-extra.parquet`)
  holds ChemPerturb-Bridge latents of the molecules observed by the model, described as "in distribution" in the
  original repository. It was created on 09/25/2026, and its generating code is in neither repository. Whether these
  latents come from a trained model is unconfirmed. The `cpb_latent_all` option is a different file: the Step 0.5
  projection of all JUMP compounds, generated with an untrained network.
* **Two generations of morphological profiles.** Notebooks 4.1 and 4.2 read the earlier normalized profiles, which
  the original author has deprecated. Notebook 4.3 reads the profiles re-normalized on 09/22/2026. A different
  ranking in 4.3 can therefore come from the transcriptomic reference or from the normalization.
* **4.3 saved outputs** come from the draft notebook that 4.3 replaces (reference `cpb_latent_distr`, PCA retaining
  99% of the variance, a +3σ significance threshold, pairwise cohorts, FastICA). Part 4 has no saved output: the
  draft's output came from an older session and used incorrect InChIKeys, which are corrected in 4.3.
* **Fixes applied during the reorganization.** The Part 1 morphology cache is named after its input file (the saved
  Part 1 output comes from an outdated DINO cache). The DRUG-seq factor cache of 4.2 has its own file name, separate
  from the gene-level cache. Numeric dose columns are excluded from the features in 4.2 Part 3.
* **Local caches.** Notebook 4.3 names caches after the source folder and file (`cached_<folder>__<file>.parquet`).
  In 4.1 and 4.2 caches are named after the model, so delete them after changing a model's file path.

Cell-by-cell issues and suggested fixes: [`docs/notebook_review.md`](../../docs/notebook_review.md). Full review:
[`docs/code_review.md`](../../docs/code_review.md).
