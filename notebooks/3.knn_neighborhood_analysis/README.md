# Step 3: Neighbourhood agreement between representations

**Goal:** test whether different representations of the same compounds agree on which compounds are similar to
each other.

## Approach

In any representation, each compound has a set of most similar compounds, its **k nearest neighbours**. When an
image-based model captures the same biology as a reference, such as the Chemical Checker bioactivity signatures, a
compound should have largely the same neighbours in both.

For every pair of representations, the notebook finds the k nearest neighbours of each compound by cosine
similarity and converts them into undirected neighbour pairs, the edges of a k-nearest-neighbour graph. The number
of edges shared by the two graphs is then compared with the number expected by chance:

* **fold enrichment** is the observed overlap divided by the overlap expected by chance (1 means chance level);
* the **hypergeometric p-value** is the probability of at least this overlap by chance.

## Notebook

[`3.1.knn_neighborhood_overlap`](3.1.knn_neighborhood_overlap.ipynb)

| Part | Question |
|---|---|
| A | How similar are all 12 representations to each other? (heatmaps at k = 10) |
| B | How does agreement change as the neighbourhood grows from k = 1 to 500? |
| C | Does chemical structure (MiniMol) agree with the image-based models? |
| D | Does bioactivity (Chemical Checker sign3) agree with the image-based models? |
| E | Which individual compounds keep their neighbours across two representations? |
| F–G | Within the compact structural and bioactivity families of Step 2, do image-based models place family members together? |

**Inputs:** tables from Steps 0–2, namely MiniMol, Chemical Checker sign2 and sign3, the raw, normalized and
active-only DINO and OpenPhenom profiles, and the cluster CSVs. **Outputs:** tables and figures inside the notebook
(the CSV export lines are commented out).

## Results

See [`results/README.md`](results/README.md) for the table and figure inventory. Results stay in the notebook unless
exported. Exported files belong in this step's `results/` folder.

**Hardware:** CPU. The notebook uses functions from the `model_benchmarking` package, so run it in the uv
environment (see the main README).

## ⚠️ Read before interpreting

* **Inputs use the earlier normalization.** The normalized and active-only profiles read here were produced before
  the 09/22/2026 re-normalization and have since been deprecated by the original author. The saved results have not
  been recomputed on the re-normalized profiles.
* **Replicate wells are not aggregated.** For tables with several wells per compound, only the first well of each
  compound is used. This adds noise and mixes doses. Step 4 uses the median of all replicate wells, the usual
  consensus profile.
* **All numeric columns are treated as features.** Numeric metadata, such as a `concentration` column or the activity
  score kept in the DINO active-only file, would enter the similarity calculation.
* **Each row uses a different set of compounds.** Each pair of representations is restricted to the compounds both
  contain, and each "Actives" table keeps the compounds that its own model called active in Step 1.2. Fold
  enrichments are therefore not comparable across rows.
* **Chemical Checker overlaps with the other data types.** Its 3,200 dimensions include chemistry spaces (A1–A5) and
  a Cell Painting morphology space (D4). Missing sign2 blocks are set to 0 before similarities are computed.
* **Reported statistics.** `Likelihood_GT_Chance` equals (1 − p) × 100 and is not the probability that an
  association is real. p-values shown as 0 are below floating-point precision.

See [`docs/code_review.md`](../../docs/code_review.md) and [`docs/notebook_review.md`](../../docs/notebook_review.md)
for details and suggested fixes.
