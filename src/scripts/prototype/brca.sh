#!/bin/bash

gpuid=$1

dataroot='/media/u2092920/Data/TCGA/BLCA/CLAM_features/pt_files'


# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="/home/u2092920/dev/MMP-main/src/splits/survival/TCGA_BLCA_overall_survival_k=${k}"
	split_names="train"
	bash "/home/u2092920/dev/MMP-main/src/scripts/prototype/clustering.sh" $gpuid $split_dir $split_names "$dataroot}"
done