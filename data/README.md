# Data directory

Notebook source downloads are cached under [`raw/`](raw/). Keeping source data at the repository root separates
immutable inputs from notebook code and generated results.

```text
data/
└── raw/
    ├── compound_metadata/
    ├── chemical_checker/
    ├── minimol/
    ├── bayer_dino/
    ├── openphenom/
    └── chemperturb_bridge/
```

The datasets are large and may be access-controlled, so their contents are ignored by Git. Each Step 0 notebook
creates its required dataset directory automatically and reuses an existing download when possible.

Processed tables, figures and validation summaries do not belong here. Step 0 writes those under
[`notebooks/0.download_and_assemble/results/`](../notebooks/0.download_and_assemble/results/).
