# Step 0: Data assembly

**Goal:** convert every source dataset into a table with one row per compound or per microscopy well and one
column per feature. Rows are labelled with the compound's **InChIKey**, a standard identifier computed from the
chemical structure. A shared identifier allows later steps to match the same compounds across chemical,
bioactivity, morphological and transcriptional data.

## Notebooks

| Notebook | Representation | Processing | Output |
|---|---|---|---|
| [`0.1.chemical_checker_signatures`](0.1.chemical_checker_signatures.ipynb) | Bioactivity (Chemical Checker) | Converts the 25 sign3 signature spaces from HDF5 to Parquet and joins them into one 3,200-dimensional table; builds the equivalent sign2 table | `JUMPCP_Sign3_CC_final.parquet`, `JUMPCP_CC_Sign2_Global_Signatures.parquet` |
| [`0.2.minimol_embeddings`](0.2.minimol_embeddings.ipynb) | Chemical structure (MiniMol) | Parses 512-dimensional embeddings stored as text and adds compound metadata | `minimol_embeddings__Kim_et_al_subset_final.parquet` |
| [`0.3.bayer_dino_embeddings`](0.3.bayer_dino_embeddings.ipynb) | Morphological profiles (Bayer DINO) | Adds compound metadata to raw and normalized well-level embeddings | `BAYER_DINO_embeddings_{nonorm,normed}_final.parquet` |
| [`0.4.openphenom_embeddings`](0.4.openphenom_embeddings.ipynb) | Morphological profiles (OpenPhenom) | Adds compound metadata to 384- and 1,920-dimensional well-level embeddings and labels DMSO negative-control wells | `openphenom_{384,1920}.parquet` |
| [`0.5.chemperturb_bridge_embeddings`](0.5.chemperturb_bridge_embeddings.ipynb) | Predicted perturbation response | Encodes Morgan fingerprints into 256-dimensional embeddings with a compound encoder | `jump_cp_latent_perturbation_embeddings.parquet` ⚠️ |

The notebooks are independent of each other and can be run in any order.

## Results

See [`results/README.md`](results/README.md) for the output inventory. Assembled datasets are kept in `results/` and
uploaded to Google Cloud Storage. Source downloads are cached by dataset under [`../../data/raw/`](../../data/raw/).
Git ignores both locations and tracks only their README inventories.

## Things to know

* **Execution environment.** These notebooks read from and write to `gs://calico-reference-embeddings`. Some cells
  require `gsutil` or Google Cloud credentials.
* **⚠️ ChemPerturb-Bridge (0.5).** The saved embeddings were generated **without trained weights**. They are a
  random projection of chemical fingerprints and carry no learned perturbation information. No saved analysis uses
  this file. Step 4.3 can select it as the reference `cpb_latent_all`; avoid this until the file is regenerated with
  trained weights.
* **A second ChemPerturb-Bridge file exists.** `cpb_observed_molecules_lpm_latents-extra.parquet` holds latents of
  the molecules observed by the model and is the reference of the saved Step 4.3 run. It was created on 09/25/2026,
  one week after the Step 0.5 product, and covers a few thousand compounds. Notebook 0.5 does not produce it, and
  its generating code is in neither this repository nor the original one. The warning above concerns only the
  Step 0.5 product, whose copy in the bucket has been unchanged since 09/18/2026.
* **Chemical Checker sign2 and sign3.** sign2 is derived from experimental data and is missing for many compounds.
  sign3 is predicted by a neural network for every molecule and is complete.
* **Well-level profiles.** The morphological tables (0.3, 0.4) contain one row per well. A compound usually occupies
  several wells (replicates and doses), and later steps aggregate them into compound-level profiles.
* **No saved outputs.** The notebooks were saved without outputs, so the counts quoted in their text cannot be
  checked from the repository. Running them uploads to the shared bucket and replaces files read by later steps.
* **Change from the original code.** In 0.3 Part C, the local staging files were renamed so that a leftover file
  from Part A (raw profiles) can never be saved as the normalized table (see
  [`docs/code_review.md`](../../docs/code_review.md)). Open issues are listed cell by cell in
  [`docs/notebook_review.md`](../../docs/notebook_review.md).
