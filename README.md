# ModelBenchmarking

Probing the correspondence between the morphological, transcriptional and chemical descriptions of small-molecule
perturbations.

## The problem

Most drugs act on several targets at once, yet their mechanism of action (MOA) is usually described with a single
label such as "HDAC inhibitor". Image-based models of cell morphology, such as DINO and OpenPhenom, are typically
evaluated by how well they predict these labels. Because the labels are incomplete, this evaluation cannot show how
much biology a model really captures.

## Our approach

We compare each model's morphological profiles (Cell Painting images from the
[JUMP](https://jump-cellpainting.broadinstitute.org/) dataset) with richer descriptions of the same compounds:

| Comparison | Question | Data |
|---|---|---|
| Morphology and chemistry | Do structurally similar compounds look alike? | MiniMol |
| Morphology and bioactivity | Do compounds with similar targets look alike? | Chemical Checker |
| Morphology and transcription | Do compounds that look alike also change gene expression alike? | Novartis DRUG-seq, ChemPerturb-Bridge |

A model whose profiles agree more with these references captures more of the underlying biology, and no MOA labels
are needed to measure it.

## The notebooks

The analysis is split into numbered steps in [`notebooks/`](notebooks), run in order:

* **[Step 0](notebooks/0.download_and_assemble)** downloads each dataset and matches compounds across datasets
  by InChIKey.
* **[Step 1](notebooks/1.preprocessing)** normalizes the image profiles to DMSO controls, removes inactive
  compounds and summarizes DRUG-seq as 43 gene programs.
* **[Step 2](notebooks/2.chemical_clustering)** groups compounds into families of related molecules.
* **[Step 3](notebooks/3.knn_neighborhood_analysis)** tests whether each compound has the same nearest neighbours
  in the image profiles and in the references.
* **[Step 4](notebooks/4.morphology_vs_transcriptomics)** measures how much variation the image profiles share with
  gene expression (CKA and SVCCA), ranks the models and interprets the shared signal with genes, pathways and the
  compounds that drive it.

Each step folder has its own README with details, and each notebook explains its code in plain language.

## Representations compared

| Representation | Type | Description | Source |
|---|---|---|---|
| **Bayer DINO** | morphological profile (Cell Painting) | self-supervised vision transformer; well-level embeddings, raw and after sphering and robust normalization to DMSO controls | Kim *et al.* (Bayer), *Self-supervision advances morphological profiling by unlocking powerful image representations* |
| **OpenPhenom** | morphological profile (Cell Painting) | masked-autoencoder vision transformer for microscopy released by Recursion; 384-d and 1,920-d well-level embeddings aggregated from single cells | Recursion |
| **CP-CNN**, **ImageNet CNN** | morphological profile (Cell Painting) | convolutional-network embeddings, normalized and prepared outside this repository | Moshkov *et al.*, 2024 |
| **MiniMol** | chemical structure and molecular properties | graph-neural-network molecular embedding pretrained on molecular property and bioassay data (512-d) | Kläser *et al.*, 2024 |
| **Chemical Checker** sign2 / sign3 | bioactivity | 25 signature spaces organized in five levels from chemistry to clinics, 3,200-d | Duran-Frigola *et al.*, 2020; Bertoni *et al.*, 2021 |
| **Novartis DRUG-seq** | transcriptional profile | U2OS cells, 3,755 compounds at 4 doses (0.01–10 µM), a single 24 h time point, 14,370 genes | Novartis (DRUG-seq: Ye *et al.*, 2018) |
| **ChemPerturb-Bridge** latent embeddings | latent perturbation response | 256-d compound embeddings used as a transcriptomic reference in Step 4.3. Two files exist: latents of the molecules observed by the model (reference of the saved 4.3 run) and a projection of all JUMP compounds (Step 0.5, ⚠️ see the limitations below) | n/a |

Compounds come from the [JUMP Cell Painting](https://jump-cellpainting.broadinstitute.org/) (JUMP-CP) dataset
(Chandrasekaran *et al.*, 2023). Datasets are matched by **InChIKey**, a standard compound identifier derived from
the chemical structure.

## Analysis pipeline

```mermaid
flowchart LR
    subgraph S0["0 · Data assembly"]
        CC["Chemical Checker<br/>sign2 / sign3"]
        MM["MiniMol"]
        DINO["Bayer DINO"]
        OP["OpenPhenom<br/>384 / 1,920"]
        CPB["ChemPerturb-Bridge ⚠️"]
    end
    subgraph S1["1 · Preprocessing"]
        NORM["OpenPhenom normalization<br/>(robust z-score + sphering)"]
        ACT["Phenotypic activity<br/>filtering"]
        DSEQ["DRUG-seq →<br/>43 latent factors"]
    end
    EXT["Re-normalized, InChIKey-indexed tables<br/>(produced outside the repository)"]
    CLU["2 · Compound<br/>clustering"]
    KNN["3 · Neighbourhood<br/>agreement"]
    CCA["4 · Morphology vs<br/>transcriptomics"]

    OP --> NORM --> ACT
    DINO --> ACT
    MM --> CLU
    CC --> CLU
    CC & MM & DINO & NORM & ACT --> KNN
    CLU --> KNN
    OP & DINO & NORM --> CCA
    DSEQ --> CCA
    CC & MM --> CCA
    EXT --> CCA
    CPB -. optional reference in 4.3 .-> CCA
```

## Repository structure

```
ModelBenchmarking/
├── data/
│   └── raw/                          # source downloads cached by Step 0 (ignored by Git)
├── notebooks/                        # analysis, one sub-folder per step (see table above)
│   ├── 0.download_and_assemble/
│   ├── 1.preprocessing/
│   ├── 2.chemical_clustering/
│   ├── 3.knn_neighborhood_analysis/
│   └── 4.morphology_vs_transcriptomics/
│       # every step contains results/README.md with its artifact inventory
├── src/model_benchmarking/           # shared, tested helper functions used by the notebooks
│   ├── alignment.py                  #   CKA, SVCCA, PCA compression and permutation nulls (Step 4)
│   ├── anchors.py                    #   loading morphological profiles against a selected transcriptomic reference (4.3)
│   ├── annotation.py                 #   compound names and mechanisms of action from PubChem and ChEMBL
│   ├── chemistry.py                  #   2D structural similarity of compounds (ECFP4 Tanimoto) (4.3)
│   ├── chemperturb_bridge.py         #   compound encoder used for the ChemPerturb-Bridge embeddings (0.5)
│   ├── data.py                       #   table helpers (InChIKey indexing)
│   ├── decoding.py                   #   axis interpretation: gene statistics, GSEA/ORA, driver compounds, volcano plots
│   ├── explainers.py                 #   tests of axes against a gene module or a known drug class (4.3)
│   ├── knn.py                        #   k-nearest-neighbour graphs and their overlap (Step 3)
│   ├── neighborhoods.py              #   neighbourhood enrichment and mean average precision of driver compounds
│   ├── paths.py                      #   repository paths, data/raw and results/ folders, GCS downloads (Step 0)
│   ├── plotting.py                   #   kNN heatmaps and sweeps, SVCCA decay figures
│   ├── pooling.py                    #   aggregation of several profiles of one compound
│   ├── profiles.py                   #   caching, selection of treated wells and replicate medians (4.1/4.2)
│   ├── purity.py                     #   sparsity and stress-signature diagnostics of axes (4.3)
│   ├── unmixing.py                   #   Varimax and FastICA rotation of canonical axes (4.3)
│   └── utils.py                      #   device selection, memory release, logging
├── tests/                            # unit tests for src/model_benchmarking (synthetic data, no cloud access)
├── docs/code_review.md               # review of the analysis code: fixes made and open issues
├── docs/notebook_review.md           # open problems in each notebook, with cell numbers
├── .github/workflows/ci.yaml         # continuous integration: pre-commit hooks, then unit tests
├── .pre-commit-config.yaml           # ruff, notebook validation, spelling and file-hygiene checks
├── pyproject.toml, uv.lock           # dependencies, managed with uv
└── LICENSE                           # Apache 2.0
```

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and all dependencies.

```bash
# 1. Install uv (once): https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment (Python 3.12 and every dependency, pinned in uv.lock)
git clone https://github.com/axiomcura/ModelBenchmarking.git
cd ModelBenchmarking
uv sync

# 3. Check the installation
uv run pytest

# 4. Open the notebooks
uv run jupyter lab
```

### Running the notebooks

* **Data access.** The repository contains no data. The notebooks read and write Parquet and CSV files in a Google
  Cloud Storage bucket. Running them requires read access to the bucket (write access for Steps 0–2) and the
  [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`, `gsutil`), authenticated with
  `gcloud auth application-default login`.
* **Google Colab.** Most notebooks were developed on Google Colab Enterprise. Step 0 caches source downloads under
  `data/raw/` and keeps its assembled outputs under `notebooks/0.download_and_assemble/results/`. Steps 1–2 still
  contain some Colab-specific staging code. The `!pip install` lines are commented out because uv manages the local
  environment. On Colab, uncomment them; for Steps 3–4, install this package with
  `!pip install git+https://github.com/axiomcura/ModelBenchmarking.git`.
* **Hardware.** Step 2 requires an NVIDIA GPU with [RAPIDS cuML](https://docs.rapids.ai/install/). Step 4 is
  faster on a GPU and also runs on a CPU. All other steps run on a CPU.
* **Web services.** Step 4 queries MyGene.info, Enrichr (rate-limited), PubChem and ChEMBL, and 4.1 downloads the
  JUMP metadata database from GitHub. Failed queries are reported as notes and the run continues.
* **Local caches.** Step 4 notebooks download their inputs once into the working directory (`cached_*.parquet`, up
  to about 1.5 GB each, plus `jump_metadata.duckdb` and `jump_smiles_standard_lookup.parquet`). These files are
  ignored by Git. In 4.1 and 4.2 a cache is named after the model, so delete it after changing that model's file
  path. Notebook 4.3 names caches after the source file.
* **Execution order.** Run each Step 4 notebook from top to bottom, because later parts reuse variables such as
  `leaderboard_df`. Parts 4–5 of 4.1 and 4.2 have known execution-order problems, listed in
  [`docs/notebook_review.md`](docs/notebook_review.md).
* **⚠️ Shared bucket.** Steps 0–1 write their outputs to the shared bucket. Running them again replaces the inputs
  of every later step for all users of the bucket.

### Development

```bash
uv run pytest                          # unit tests
uv run pre-commit install              # run the checks before every commit
uv run pre-commit run --all-files      # ruff lint and format (code and notebooks), notebook validation, spelling
```

The same hooks and tests run in continuous integration
([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)) on every pull request.

## Data products

Each numbered notebook folder contains a `results/` directory that documents the tables, figures and data products
of that step. Step 0 keeps its assembled outputs in `results/` and uploads the same files to Google Cloud Storage;
its raw downloads are stored under `data/raw/`. Tables and figures exported by later notebooks belong in the
`results/` folder of their step. Git ignores raw data and generated results and tracks only the README inventories.

### Produced by this repository

| File | Contents | Made in |
|---|---|---|
| `embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_Sign3_CC_final.parquet` | CC sign3, 3,200-d | 0.1 |
| `embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_CC_Sign2_Global_Signatures.parquet` | CC sign2, 3,200-d (sparse) | 0.1 ¹ |
| `Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_embeddings__Kim_et_al_subset_final.parquet` | MiniMol, 512-d | 0.2 |
| `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_nonorm_final.parquet` | Bayer DINO, raw, well level | 0.3 |
| `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_normed_final.parquet` | Bayer DINO, normalized, well level | 0.3 |
| `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_{384,1920}.parquet` | OpenPhenom, raw, well level | 0.4 |
| `embeddings/ChemPerturb-Bridge/latent_perturbation_embeddings/jump_cp_latent_perturbation_embeddings.parquet` | ⚠️ generated with an **untrained** encoder; see Step 0.5 | 0.5 |
| `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_{384,1920}_sphered_mad_robustized.parquet` | OpenPhenom, normalized (earlier method ³) | 1.1 |
| `embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_{384,1920}_sphered_mad_robustized-ACTIVE.parquet` | OpenPhenom, normalized (earlier method ³), phenotypically active wells | 1.2 |
| `Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/Bayer_DINO_embeddings_normed_final-actives_only.parquet` | Bayer DINO, normalized (earlier method ³), phenotypically active wells | 1.2 ² |
| `embeddings/Novartis_DRUGseq_expression_data/novartis_drugseq_u2os_10uM_24h_normalized_factors.parquet` | DRUG-seq 10 µM / 24 h, 43 latent factors | 1.3 |
| `Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_clusters.csv` | MiniMol HDBSCAN clusters | 2 |
| `Chemical_Checker_Embeddings/parquets/sign3_clusters.csv` (at the bucket root, outside `embeddings/`) | CC sign3 HDBSCAN clusters | 2 |

¹ Written to `Annotations_from_Roman/.../annotations_from_Roman/` and copied to this location outside the notebook.
² Produced by a different run of the Step 1.2 code; the saved code writes a file ending in `ACTIVE_ONLY-test2.parquet`.
³ The original author re-normalized all morphological datasets on 09/22/2026 and deprecated the earlier files. The
earlier files are still in the bucket (dated May 2026) and are the ones read by Steps 1.2, 3, 4.1 and 4.2. The same
applies to `BAYER_DINO_embeddings_normed_final.parquet`.

### Inputs and files produced outside this repository

| File | Contents | Used in |
|---|---|---|
| `embeddings/Novartis_DRUGseq_expression_data/Novartis_U2OS_Dose_Time_Signatures.parquet` | DRUG-seq log₂ fold changes, all doses | 1.3, 4.1, 4.2, 4.3 (gene-level interpretation) |
| `embeddings/Novartis_DRUGseq_expression_data/novartis_drugseq_u2os_10uM_24h.parquet` | 10 µM / 24 h subset of the above | 1.3 |
| `Annotations_from_Roman/.../JUMP_CP_compound_metadata_with_SMILES_standard.csv` | JUMP compound metadata with standardized SMILES | 0, 4.3 |
| `embeddings/Chemical_Checker_Embeddings/JUMP_CP_Global_Signatures_full.parquet` | CC sign3 for all JUMP compounds | 4 |
| `Annotations_from_Roman/.../minimol_embeddings/minimol_embeddings__full_final.parquet` | MiniMol for all JUMP compounds | 4 |
| `embeddings/CP-CNN/*.parquet`, `embeddings/Imagenet/*.parquet` | CP-CNN and ImageNet CNN embeddings | 4 |
| `embeddings/JUMP MOA annotations/annotations_compound_gene_curated.parquet` | curated compound–gene (mechanism of action) annotations from the JUMP consortium | n/a |
| `*_with_JUMPMOA.parquet`, `*_JUMP_MOA.parquet`, `*_with_MOA.parquet`, `sign3_big_clusters.csv`, `JUMP_CP_Global_Signatures_Clustered_Only.parquet` | datasets joined with mechanism-of-action labels, and additional clusterings | n/a |
| `*_0922_normed.parquet`, `CP-CNN_JUMP-CP_9.22_normalized.parquet`, `Imagenet_JUMP-CP_9.22_normalized.parquet` | re-normalized morphological profiles (09/22/2026), which replace the earlier normalized files ³; the code is in neither repository | 4.3 |
| `indexed_by_inchikey/*.parquet` (12 files, written 09/25/2026) | current versions of the morphological profiles, DRUG-seq signatures and factors, and ChemPerturb-Bridge latents, rewritten with an InChIKey column | 4.3 |
| `indexed_by_inchikey/cpb_observed_molecules_lpm_latents-extra.parquet` | ChemPerturb-Bridge latents of the molecules observed by the model ("in distribution" in the original repository); created 09/25/2026; ⚠️ the code that produced it is in neither repository; reference of the saved 4.3 run | 4.3 |
| `indexed_by_inchikey/jump_cp_latent_perturbation_embeddings.parquet` | ChemPerturb-Bridge projection of all JUMP compounds (the Step 0.5 product, re-indexed); ⚠️ untrained encoder; the `cpb_latent_all` option | 4.3 (option) |

## Code review and known limitations

The notebooks originate from [cytodata/ModelBenchmarking](https://github.com/cytodata/ModelBenchmarking). This
repository reorganizes them and moves their shared functions into a tested package. The analysis code was reviewed
during this reorganization. Issues that could silently produce wrong
results were fixed. Issues that require a scientific decision, or a re-run with data access, are documented.
**Read [`docs/code_review.md`](docs/code_review.md) and [`docs/notebook_review.md`](docs/notebook_review.md) before
interpreting or publishing results.** The main points are:

* **Model ranking is unresolved.** Differences between the top-ranked models are smaller than the changes caused by
  the choice of benchmark design or similarity metric (CKA or SVCCA), and no confidence intervals are reported.
* **Data provenance.** Several Step 4 inputs are produced outside this repository, including every file read by 4.3
  and the transcriptomic reference of its saved run. Two ChemPerturb-Bridge files exist. The projection of all JUMP
  compounds (Step 0.5) was generated with an untrained network, and its copy in the bucket has been unchanged since
  09/18/2026. The latents of observed molecules, used by the saved 4.3 run, are a separate file whose generating code
  is unavailable.
* **Two generations of input files.** Steps 3, 4.1 and 4.2 read the earlier normalized profiles, which the original
  author has deprecated. Step 4.3 reads the re-normalized profiles. Results from 4.3 therefore differ from 4.1 and
  4.2 in both the transcriptomic reference and the normalization of the morphological profiles.
* **Normalization.** Step 1.1 estimates the robust z-score and the sphering transform from DMSO wells pooled across
  all plates. Plate and batch effects therefore remain in the normalized OpenPhenom profiles. Standard practice
  normalizes each plate to its own negative controls. The code of the 09/22/2026 re-normalization is unavailable,
  so its method cannot be checked.
* **Comparability.** Step 3 compares models on different compound sets, uses a single well per compound and keeps a
  model-specific subset of active compounds. The Step 4 pilot analyses (Part 1 of 4.1 and 4.2) treat each well as an
  independent sample, although replicate wells share one DRUG-seq profile.

## References

* Bray *et al.* (2016). Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. *Nature Protocols*.
* Chandrasekaran *et al.* (2023). JUMP Cell Painting dataset: morphological impact of 136,000 chemical and genetic perturbations. *bioRxiv*.
* Kim *et al.* Self-supervision advances morphological profiling by unlocking powerful image representations (Bayer).
* Moshkov *et al.* (2024). Learning representations for image-based profiling of perturbations. *Nature Communications*.
* Kläser *et al.* (2024). MiniMol: a parameter-efficient foundation model for molecular learning. *arXiv:2404.14986*.
* Duran-Frigola *et al.* (2020). Extending the small-molecule similarity principle to all levels of biology with the Chemical Checker. *Nature Biotechnology*.
* Bertoni *et al.* (2021). Bioactivity descriptors for uncharacterized chemical compounds. *Nature Communications*.
* Ye *et al.* (2018). DRUG-seq for miniaturized high-throughput transcriptome profiling in drug discovery. *Nature Communications*.
* Kornblith *et al.* (2019). Similarity of neural network representations revisited (CKA). *ICML*.
* Raghu *et al.* (2017). SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability. *NeurIPS*.
* Subramanian *et al.* (2005). Gene set enrichment analysis. *PNAS*.
* Horn (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*.
* Kaiser (1958). The varimax criterion for analytic rotation in factor analysis. *Psychometrika*.
* Hyvärinen and Oja (2000). Independent component analysis: algorithms and applications. *Neural Networks*.
* Rogers and Hahn (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*.

## License

[Apache License 2.0](LICENSE).
