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

#bayer DINO: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_nonorm_final.parquet
#bayer DINO+post processing: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_normed_final.parquet
#bayer DINO+post processing ACTIVES: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/Bayer_DINO_embeddings_normed_final-actives_only.parquet
#bayer DINO actives with moa: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/Bayer_DINO_embeddings_normed_final-actives_only_with_MOA.parquet

#openphenom-384: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384.parquet
#openphenom-384 sphered+madrobustize: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized.parquet
#openphenom-1920: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920.parquet
#openphenom-1920 sphered+madrobustize: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized.parquet
#openphenom-384 sphered+madrobustize ACTIVES: 'gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized-ACTIVE.parquet'
#openphenom-1920 sphered+madrobustize ACTIVES:'gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized-ACTIVE.parquet'
#openphenom-384 actives + MOA gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized-ACTIVE_with_MOA.parquet
#openphenom-1920 actives + MOA gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized-ACTIVE_with_MOA.parquet

#CP CNN (normalized): gs://calico-reference-embeddings/embeddings/CP-CNN/master_cp-cnn_normalized.parquet
#CP CNN drugseq subset (normalized)": "gs://calico-reference-embeddings/embeddings/CP-CNN/master_cp_cnn_normed_drugseq_subset.parquet"
#"CP CNN drugseq subset (normalized, 512d)": "gs://calico-reference-embeddings/embeddings/CP-CNN/master_cp_cnn_512D_sphered_mad_consensus.parquet",

#Imagenet (normalized): gs://calico-reference-embeddings/embeddings/Imagenet/master_imagenet_normalized.parquet
#Imagenet drugseq subset (normalized)": "gs://calico-reference-embeddings/embeddings/Imagenet/master_imagenet_normed_drugseq_subset.parquet"

#novartis drugseq u2os:    gs://calico-reference-embeddings/embeddings/Novartis_DRUGseq_expression_data/Novartis_U2OS_Dose_Time_Signatures.parquet
#novartis drugseq u2os normalized: gs://calico-reference-embeddings/embeddings/Novartis_DRUGseq_expression_data/novartis_drugseq_u2os_10uM_24h_normalized_factors.parquet

#Chemperturb-bridge latent cross-cell line transcriptional perturbation representations: gs://calico-reference-embeddings/embeddings/ChemPerturb-Bridge/latent_perturbation_embeddings/jump_cp_latent_perturbation_embeddings.parquet

The KNN sweep notebook contains the code to run the analyses for kNN comparisons between all of the data sets above

6/12/202 Update: I joined in MOA labels from Jump consortium/Shantanu's group.  
The MOA labels are located here: gs://calico-reference-embeddings/embeddings/JUMP MOA annotations/annotations_compound_gene_curated.parquet

I have added a notebook to run CCA between the morphology datasets (including imagenet and CP-CNN) and Novartis U2OS drug-seq data.

09/21/2026 Update: I have added a notebooks to do the following:
1) Normalize the Novartis Drugseq dataset (2026_09_16_Normalizing_drug_seq.ipynb)
2) Extract the latent transcriptional perturbation representations from chemperturb-bridge (20260920-ChemPerturbRidge_extract_latent_representation.ipynb)
3) Added the Morphology_vs_Drugseq_v3.ipynb, updating the approach to running CCA
