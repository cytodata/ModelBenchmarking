# Morphology_vs_MOA
Notebooks for comparing chemical representations with morphological representations

make_parquets notebook is just for reference.  The output are the following files:

#minimol: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_embeddings__Kim_et_al_subset_final.parquet
#minimol with BROAD MOA labels: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_embeddings__Kim_et_al_subset_final_with_JUMPMOA.parquet
#minimol cluster: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_clusters.csv

#chemical checker sign2: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_CC_Sign2_Global_Signatures.parquet
#chemical checker sign3: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_Sign3_CC_final.parquet
#chemical checker sign3 clustered: gs://calico-reference-embeddings/Chemical_Checker_Embeddings/parquets/sign3_clusters.csv
#chemical checker sign3 with MOA labels: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_Sign3_CC_final_JUMP_MOA.parquet'
#chemical checker sign3 ALL: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/JUMP_CP_Global_Signatures_full.parquet
#chemical checker sign3 ALL_CLUSTERED: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/JUMP_CP_Global_Signatures_Clustered_Only.parquet
#cc_imputed_clusters gs://calico-reference-embeddings/Chemical_Checker_Embeddings/parquets/sign3_big_clusters.csv

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

6/12/202 Update: I joined in MOA labels from Jump consortium/Shantanu's group.  
The MOA labels are located here: gs://calico-reference-embeddings/embeddings/JUMP MOA annotations/annotations_compound_gene_curated.parquet

I have added a notebook to run CCA between the morphology datasets (including imagenet and CP-CNN) and Novartis U2OS drug-seq data.

09/21/2026 Update: I have added a notebooks to do the following:
1) Normalize the Novartis Drugseq dataset (2026_09_16_Normalizing_drug_seq.ipynb)
2) Extract the latent transcriptional perturbation representations from chemperturb-bridge (20260920-ChemPerturbRidge_extract_latent_representation.ipynb)
3) Added the Morphology_vs_Drugseq_v3.ipynb, updating the approach to running CCA

09/23/2026 Update: I found errors in my normalization methods and have renormalized all the morphological datasets.  I have deprecated many datasets, but still have the links elsewhere if needed.
