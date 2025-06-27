# Metric Depth: Converting relative depth estimation to absolute(/metric) depth using the DIODE dataset

### Requirements for prepare_dataset.py python script:

- numpy
- matplotlib

### How to enable OpenCL (GPU) support on Linux (optional):

Install opencl dev package: (ubuntu)

```bash
sudo apt install ocl-icd-opencl-dev
```

If you have a NVIDIA card, also install this package:

```bash
sudo apt install nvidia-opencl-dev
```

Verify OpenCL installation:

```
clinfo
```

> [!NOTE]
> The NVIDIA OpenCL driver will not load correctly when AddressSanitizer is enabled in cmake (`cmake -B build -DENABLE_ASAN=ON`).
>
> An error `clGetPlatformIDs returned -1001` will occur, and we will fallback into CPU only mode (super slow!).
>
> To use TFLite GPU delegate using OpenCL with ASAN enabled, you need to set this environment variable when running the program:
>
> `ASAN_OPTIONS=protect_shadow_gap=0 ./build/metric_depth/EvaluateDataset ...`
>
> See this stackoverflow post for further information: <https://stackoverflow.com/questions/55750700/opencl-usable-when-compiling-host-application-with-address-sanitizer>

<br>

### How to evaluate the DIODE dataset:

1. Download and extract the dataset (prefer the testing dataset, as it is much smaller) by visiting <https://diode-dataset.org/>

2. Evaluate the dataset:

   ```bash
   ./scripts/build_and_run_eval_dataset.sh <dataset_directory> <dataset_evaluation_directory>
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
