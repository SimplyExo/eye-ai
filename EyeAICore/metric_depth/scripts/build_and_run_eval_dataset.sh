#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 <midas_model_filepath> <prepare_dataset.py> <dataset_directory> <evaluation_dataset_directory>"
    exit 1
fi

scripts_directory="$(dirname "$0")"

echo "Building..."

cmake --build "$scripts_directory/../../build" -j8 --target EvaluateDataset

echo "Running..."

"$scripts_directory/../../build/metric_depth/EvaluateDataset" $1 $2 $3 $4