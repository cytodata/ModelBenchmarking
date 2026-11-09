# ModelBenchmarking

Probing the correspondence between the morphological, transcriptional and chemical descriptions of small-molecule
perturbations.

## The problem

Most drugs act on several targets at once, yet their mechanism of action (MOA) is usually described with a single
label such as "HDAC inhibitor". Image-based models of cell morphology, such as DINO and OpenPhenom, are typically
evaluated by how well they predict these labels. Because the labels are incomplete, this evaluation cannot show how
much biology a model really captures.

#MORPHOLOGY_DATASETS
    #"DINO (Raw)": "gs://calico-reference-embeddings/indexed_by_inchikey/BAYER_DINO_embeddings_nonorm_final.parquet",
    #"DINO (Normed)": "gs://calico-reference-embeddings/indexed_by_inchikey/BAYER_DINO_embeddings_0922_normed.parquet",
    #'CP CNN (normalized)': "gs://calico-reference-embeddings/indexed_by_inchikey/CP-CNN_JUMP-CP_9.22_normalized.parquet",
    #'Imagenet (normalized)': "gs://calico-reference-embeddings/indexed_by_inchikey/Imagenet_JUMP-CP_9.22_normalized.parquet",
    #"OpenPhenom 384 (Raw)": "gs://calico-reference-embeddings/indexed_by_inchikey/openphenom_384.parquet",
    #"OpenPhenom 384 (Normed)": "gs://calico-reference-embeddings/indexed_by_inchikey/openphenom_384_0922_normed.parquet",
    #"OpenPhenom 1920 (Raw)": "gs://calico-reference-embeddings/indexed_by_inchikey/openphenom_1920.parquet",
    #"OpenPhenom 1920 (Normed)": "gs://calico-reference-embeddings/indexed_by_inchikey/openphenom_1920_0922_normed.parquet",

#TRANSCRIPTOMIC_DATASETS
    #"drugseq_normalized": "Novartis DRUG-seq 10 µM / 24H Normalized Varimax Factors: "gs://calico-reference-embeddings/indexed_by_inchikey/novartis_drugseq_u2os_10uM_24h_normalized_factors.parquet",
    #"cpb_latent": "ChemPerturb-Bridge (CPB) / CPA Latent Perturbation Representations - in distribution: "gs://calico-reference-embeddings/indexed_by_inchikey/cpb_observed_molecules_lpm_latents-extra.parquet",
    #"cpb_latent_all": "ChemPerturb-Bridge (CPB) / CPA Latent Perturbation Representations - projected all jump: "gs://calico-reference-embeddings/indexed_by_inchikey/jump_cp_latent_perturbation_embeddings.parquet",
    #"drugseq_raw":"Novartis DRUG-seq Raw Expression Signatures (~15k genes): "gs://calico-reference-embeddings/indexed_by_inchikey/Novartis_U2OS_Dose_Time_Signatures.parquet",
The KNN sweep notebook contains the code to run the analyses for kNN comparisons between all of the data sets above

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
