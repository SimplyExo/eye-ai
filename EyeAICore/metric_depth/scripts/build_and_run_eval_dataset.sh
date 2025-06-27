#!/bin/bash

# exit immediately on error
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset_directory> <evaluation_dataset_directory>"
    exit 1
fi

scripts_directory="$(dirname "$0")"

echo "Building..."

cmake --build "$scripts_directory/../../build" -j8 --target EvaluateDataset

echo "Running..."
midas_model_filepath="$scripts_directory/../../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite"
"$scripts_directory/../../build/metric_depth/EvaluateDataset" "$midas_model_filepath" "$1" "$2"