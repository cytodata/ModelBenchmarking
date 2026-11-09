# Step 3 results - nearest-neighbour agreement

This folder is the local results location for nearest-neighbour comparison tables and figures. The notebook
currently retains its results in notebook output cells; CSV export statements are present but commented out.

## Result tables

- All-by-all representation comparison at the selected neighbourhood size.
- Neighbourhood-size sweep across representation pairs.
- MiniMol-anchored and Chemical Checker-anchored comparisons.
- Per-compound neighbour-overlap statistics.
- Chemical-cluster and bioactivity-cluster restricted comparisons.

The historical optional export names are:

- `All_By_All_KNN_Results.csv`
- `All_By_All_KNN_Sweep_Results.csv`

If those exports are enabled, local copies should be written here.

## Figures

- Pairwise overlap and fold-enrichment heatmaps.
- Neighbourhood-size sweep plots.
- Anchor-model comparison plots.
- Cluster-restricted comparison plots.

No result file in this folder should be treated as authoritative unless its generating notebook, input data
versions and parameters are recorded alongside it.
