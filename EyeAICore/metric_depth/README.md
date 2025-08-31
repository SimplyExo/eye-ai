\page MetricDepth Metric Depth

Converting relative depth estimation to absolute(/metric) depth using SUNRGB-D (or DIODE) dataset

## Training the Rel2Abs model

Installing python requirements for training the Rel2Abs model:

```bash
pip install -r ./rel2abs_training/requirements.txt
```

Preparing the dataset:

1. Download and extract the dataset:

- SUNRGB-D: <https://rgbd.cs.princeton.edu>
- DIODE: <https://diode-dataset.org> (not recommended as images are not too usable, download testing data as sample)

2. Prepare the dataset:

   ```bash
   cmake --preset=release

   ./scripts/build_and_run_prepare_dataset.sh <diode or sun_rgbd> <dataset_directory> <dataset_evaluation_directory>
   ```

Running the Rel2Abs model training:

```bash
cd rel2abs_training
python train.py <prepared_dataset_directory>
```

The model will be trained and exported as `rel2abs_model.tflite`

<br>

## Rel2Abs model documentation/specification

input shape: float32[256 * 256 * 4], RGB-D
output shape: float32[5], polynomial coeffs for degree 4 polynomial function

the input layer consists of 3 float32 rgb channels in sRGB colorspace in the range [-1, 1].
the fourth channel is a float32 relative depth channel in the range [-1, 1], that is fed the raw relative depth output of MiDaS (in the range of 0 to 1500), but remapped to [-1, 1].
Raw relative depth values larger than 1500 are clamped to 1500, as they are hard to encounter and have no practical relevence, as you need an object <1cm close to the camera to produce such values, which is why clamping is not a problem here.

The output of the model are the coefficients of a polynomial function that is able to convert relative depth values to absolute/metric depth values.
For example, a degree 4 polynomial function would require 5 coefficients.
However, since neural networks do not train well when the output coeffs can be very small (< 1e-11),
as the MSE and MAE appear to be very small, the diffs of the coeffs get ignored while training.
To solve this, we scale each coeff individually while training such that the majority of coeffs land in the range of [-1, 1],
such that training works as expected.
While inference, we scale the output coeffs back down using the inverse of the scaling factor used during training.

<br>

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
> The NVIDIA OpenCL driver will not load correctly when AddressSanitizer is enabled in cmake (`cmake -B build -DEYE_AI_CORE_ENABLE_ASAN=ON`).
>
> An error `clGetPlatformIDs returned -1001` will occur, and we will fallback into CPU only mode (super slow!).
>
> To use TFLite GPU delegate using OpenCL with ASAN enabled, you need to set this environment variable when running the program:
>
> `ASAN_OPTIONS=protect_shadow_gap=0 ./build/metric_depth/PrepareDataset ...`
>
> See this stackoverflow post for further information: <https://stackoverflow.com/questions/55750700/opencl-usable-when-compiling-host-application-with-address-sanitizer>
