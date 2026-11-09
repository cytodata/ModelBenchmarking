"""Shared helper functions for the ModelBenchmarking analysis notebooks.

The notebooks in ``notebooks/`` compare how different representations of the same small
molecules (chemical structure, bioactivity signatures, Cell Painting image embeddings and
DRUG-seq transcriptomic profiles) agree with each other. Functions used by the notebooks
live here so that every notebook uses one tested implementation.

Modules
-------
alignment
    Global similarity between two representations (linear CKA) and per-axis agreement
    via SVCCA, each with permutation-based null baselines.
anchors
    Load morphology tables against an interchangeable transcriptomic "anchor" by InChIKey.
annotation
    Look up a compound's common name and mechanism of action from PubChem/ChEMBL.
chemistry
    2D chemical similarity (ECFP4 Tanimoto) of compounds.
chemperturb_bridge
    Compound encoder mapping Morgan fingerprints to latent perturbation embeddings.
data
    Small data-handling helpers (e.g. indexing tables by InChIKey).
explainers
    Test whether canonical axes track a known gene module or a known drug class.
neighborhoods
    Nearest-neighbour enrichment and retrieval metrics for groups of compounds.
paths
    Stable repository paths for raw data and stage-local notebook results.
plotting
    Figures for the nearest-neighbour and SVCCA analyses.
pooling
    Collapse several profiles of the same compound into one.
purity
    Sparsity and stress-contamination diagnostics of canonical axes.
unmixing
    Varimax and FastICA unmixing of canonical axes.
utils
    Device selection, progress logging and download progress bars.
"""

__version__ = "0.1.0"
