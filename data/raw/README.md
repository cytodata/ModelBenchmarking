# Raw data

This directory contains local, unmodified source files downloaded by the Step 0 notebooks. Dataset-specific
subdirectories are created automatically:

- `compound_metadata/` - shared JUMP compound metadata.
- `chemical_checker/` - Chemical Checker HDF5 and source-block Parquet files.
- `minimol/` - source MiniMol CSV embeddings.
- `bayer_dino/` - raw and normalized Bayer DINO source CSV files.
- `openphenom/` - OpenPhenom 384- and 1,920-dimensional source CSV files.
- `chemperturb_bridge/` - compound metadata used by the ChemPerturb-Bridge extraction.

Do not commit downloaded datasets. They can be recreated from the GCS URIs recorded in the notebooks.
