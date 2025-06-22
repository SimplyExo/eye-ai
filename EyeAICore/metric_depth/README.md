# Metric Depth: Converting relative depth estimation to absolute(/metric) depth using the DIODE dataset

### How to enable OpenCL support (optional):

Install opencl dev package: (linux/ubuntu)

```bash
sudo apt install ocl-icd-opencl-dev
```

### How to prepare and evaluate the DIODE dataset:

1. Download the dataset (prefer the testing dataset, as it is much smaller) by visiting <https://diode-dataset.org/>
2. Prepare the dataset (converting .png and .npy files to simple binary .bin files)

```bash
python3 ./scripts/prepare_dataset.py <path_to_diode_dataset_directory> <path_to_prepared_diode_dataset_directory>
```

3. Evaluate the prepared dataset:

(for example: midas_model_filepath could be "../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite")

```bash
./scripts/build_and_run_eval_dataset.sh <midas_model_filepath.tflite> <prepared_dataset_directory> <evaluation_dataset_directory>
```

4. (optional) Visualize the evaluation:

Visualize a single scan evaluation:

```bash
python3 ./scripts/visualize_result_file.py <filepath_to_result_file.csv>
```

Or visualilze all evaluations of either indoors or outdoors:

```bash
python3 ./scripts/visualize_all_trendlines.py <indoors/outdoor evaluation_directory>
```
