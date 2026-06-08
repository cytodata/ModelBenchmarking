# Morphology_vs_MOA
Notebooks for comparing chemical representations with morphological representations

make_parquets notebook is just for reference.  The output are the following files:

#minimol: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_embeddings__Kim_et_al_subset_final.parquet
#minimol cluster: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/minimol_embeddings/minimol_clusters.csv


#chemical checker sign2: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_CC_Sign2_Global_Signatures.parquet
#chemical checker sign3: gs://calico-reference-embeddings/embeddings/Chemical_Checker_Embeddings/parquets/JUMPCP_Sign3_CC_final.parquet
#chemical checker sign3 clustered: gs://calico-reference-embeddings/Chemical_Checker_Embeddings/parquets/sign3_clusters.csv

#bayer DINO: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_nonorm_final.parquet
#bayer DINO+post processing: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/BAYER_DINO_embeddings_normed_final.parquet
#bayer DINO+post processing ACTIVES: gs://calico-reference-embeddings/Annotations_from_Roman/Bayer_data_harmonized/Bayer_DINO_embeddings/Bayer_DINO_embeddings_normed_final-actives_only.parquet

#openphenom-384: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384.parquet
#openphenom-384 sphered+madrobustize: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized.parquet
#openphenom-1920: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920.parquet
#openphenom-1920 sphered+madrobustize: gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized.parquet
#openphenom-384 sphered+madrobustize ACTIVES: 'gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_384_sphered_mad_robustized-ACTIVE.parquet'
#openphenom-1920 sphered+madrobustize ACTIVES:'gs://calico-reference-embeddings/embeddings/JUMP_Bayer_compounds/OpenPhenom/parquets/openphenom_1920_sphered_mad_robustized-ACTIVE.parquet'



The KNN sweep notebook contains the code to run the analyses for kNN comparisons between all of the data sets above
