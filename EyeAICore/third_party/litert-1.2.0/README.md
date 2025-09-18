# LiteRT (current version: 1.2.0)

## Steps to replicate when updating to newer version
1. extracted from the .aar file hosted on the Maven Repository [litert-1.2.0.aar](https://maven.google.com/com/google/ai/edge/litert/litert/1.2.0/litert-1.2.0.aar)
2. rename to `litert-1.2.0.zip` and extract
3. copy everything from `headers` into the `include` directory
4. copy `include/tflite/c/common.h` to `include/tensorflow/lite/c/common.h` so that the qnn npu delegate headers can find it (they still expected tensorflow folder structure instead of LiteRT's new structure)
5. copy all libraries from `jni` to `lib` directory
6. copy `LICENSE` if it changed