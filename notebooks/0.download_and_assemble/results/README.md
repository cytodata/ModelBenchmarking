# Step 0 results - assembled datasets

This folder contains the local assembled outputs from Step 0. Each notebook also uploads the same durable product
to `gs://calico-reference-embeddings`. The files are large and therefore ignored by Git; this README remains
tracked as their artifact inventory. Immutable source downloads are kept separately under
[`../../../data/raw/`](../../../data/raw/).

## Durable outputs

| Notebook | Type | Output |
|---|---|---|
| 0.1 | Parquet data | `embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_Sign3_CC_final.parquet` |
| 0.1 | Parquet data | `Annotations_from_Roman/Bayer_data_harmonized/annotations_from_Roman/JUMPCP_CC_Sign2_Global_Signatures.parquet` |
| 0.2 | Parquet data | `Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_embeddings__Kim_et_al_subset_final.parquet` |
| 0.3 | Parquet data | `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_nonorm_final.parquet` |
| 0.3 | Parquet data (legacy normalized path) | `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_norm_final.parquet` |
| 0.3 | Parquet data | `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_normed_final.parquet` |
| 0.4 | Parquet data | `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384.parquet` |
| 0.4 | Parquet data | `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920.parquet` |
| 0.5 | Parquet data (experimental) | `embeddings/ChemPerturb-Bridge/latent_perturbation_embeddings/jump_cp_latent_perturbation_embeddings.parquet` |

The local files use the basenames shown above and are stored directly in this folder. Paths in the table are the
corresponding locations relative to the bucket named above. Notebook 0.1 also retains per-space Chemical Checker
Parquet files under `results/sign3_spaces/` and uploads them to
`embeddings/Chemical_Checker_Embeddings/sign3_parquets/`.

## Important provenance notes

- The sign2 file consumed downstream is copied to a different bucket location outside notebook 0.1.
- Notebook 0.3 retains the older `BAYER_DINO_embeddings_norm_final.parquet` result for provenance; downstream code
  uses the `normed_final` file.
- The current ChemPerturb-Bridge output was generated without trained encoder weights and must not be interpreted
  as a learned perturbation representation.
