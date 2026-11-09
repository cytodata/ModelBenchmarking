# Step 1 results - preprocessed profiles

This folder is the local results location for preprocessing diagnostics, quality-control summaries and small
exported samples. Full normalized datasets are written directly to `gs://calico-reference-embeddings` and are
not committed here.

## Durable outputs

| Notebook | Type | Output |
|---|---|---|
| 1.1 | Parquet data | `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized.parquet` |
| 1.1 | Parquet data | `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized.parquet` |
| 1.2 | Parquet data | `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/Bayer_DINO_embeddings__Kim_et_al_subset__sphering_mad_robustize_well_features__ACTIVE_ONLY-test2.parquet` |
| 1.2 | Parquet data | `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized-ACTIVE.parquet` |
| 1.3 | Parquet data | `embeddings/Novartis_DRUGseq_expression_data/novartis_drugseq_u2os_10uM_24h_normalized_factors.parquet` |

Paths are relative to the bucket named above.

## Notebook-only diagnostics

- DMSO distance distributions and activity thresholds.
- Horn's parallel-analysis curve and selected factor count.
- Factor-loading heatmaps and explained-variance summaries.

## Known output gaps

- No cell currently creates `openphenom_384_sphered_mad_robustized-ACTIVE.parquet`, although Step 3 expects it.
- Step 3 expects a differently named DINO active-only file. The exact mismatch is documented in the main code
  review and must be resolved before treating the pipeline as reproducible from a clean run.
