# TFLite Dynamic

Loads in a `tensorflowlite_c` dynamic library at runtime and provides bindings to it.

## Building TensorFlow Light DLL

This is not necessary for compiling a project with this library, as it's loaded in at runtime, but
it's still necessary to run.

```
bazel build -c opt --action_env PYTHON_BIN_PATH="C://ProgramData//Miniconda3//python.exe" //tensorflow/lite/c:tensorflowlite_c --config=monolithic
```

## XNNPACK

This library includes bindings for XNNPACK, which is built with the TensorFlow Lite C DLL by default
on Windows.
You should be using this on x86, as it's faster than the default CPU delegate, which is optimized
for specific ARM instructions.
