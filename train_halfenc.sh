#!/bin/bash
# Train one config with nnUNetTrainer_100epochs, fold 0.
# Usage: bash train_halfenc.sh DATASET_ID CFG GPU
set -euo pipefail
dataset_id=$1
cfg=$2
gpu=$3

export nnUNet_raw="../nntinyunet_orig/data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results/nnunet"
export nnUNet_compile=false
export PYTHONPATH="src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$gpu

.venv/bin/python -m nnunetv2.run.run_training "$dataset_id" "$cfg" 0 -p nnUNetPlans -tr nnUNetTrainer_100epochs
