# Step 2 results - chemical and bioactivity clusters

This folder is the local results location for exported cluster tables, UMAP plots, heatmaps and diagnostics.
The two durable cluster-assignment tables are written directly to `gs://calico-reference-embeddings`.

## Durable outputs

| Representation | Type | Output |
|---|---|---|
| MiniMol | CSV table | `Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_clusters.csv` |
| Chemical Checker sign3 | CSV table | `Chemical_Checker_Embeddings/parquets/sign3_clusters.csv` |

Paths are relative to the bucket named above.

## Notebook-only diagnostics

- Two-dimensional UMAP projections colored by HDBSCAN cluster.
- Cluster-size and noise summaries.
- PCA-based hierarchical-clustering heatmaps used to inspect cluster coherence.

These diagnostics remain embedded in the notebook unless they are explicitly exported to this folder.
