# Metric Depth: Converting relative depth estimation to absolute(/metric) depth using the DIODE dataset

### Requirements for prepare_dataset.py python script:

- numpy

### How to enable OpenCL (GPU) support on Linux (optional):

Install opencl dev package: (ubuntu)

```bash
sudo apt install ocl-icd-opencl-dev
```

If you have a NVIDIA card, also install this package:

```bash
sudo apt install nvidia-opencl-dev
```

<br>

### How to evaluate the DIODE dataset:

1. Download and extract the dataset (prefer the testing dataset, as it is much smaller) by visiting <https://diode-dataset.org/>

2. Evaluate the dataset:

   (for example: midas_model_filepath could be "../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite" and prepare_dataset.py could be "./scripts/prepare_dataset.py")

   ```bash
   ./scripts/build_and_run_eval_dataset.sh <midas_model_filepath.tflite> <prepare_dataset.py> <prepared_dataset_directory> <evaluation_dataset_directory>
   ```

3. (optional) Visualize the evaluation:

   Visualize a single scan evaluation:

   ```bash
   python3 ./scripts/visualize_result_file.py <filepath_to_result_file.csv>
   ```

   Or visualilze all evaluations of either indoors or outdoors:

   ```bash
   python3 ./scripts/visualize_all_trendlines.py <indoors/outdoor evaluation_directory>
   ```
