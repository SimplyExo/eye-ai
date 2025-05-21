# OnnxRuntime Android (current version: 1.22.0)

## Steps to replicate when updating to newer version
1. extracted from the .aar file hosted on the Maven Repository [onnxruntime-android-1.22.0.aar](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.22.0/onnxruntime-android-1.22.0.aar)
2. rename to `onnxruntime-android-1.22.0.zip` and extract
3. copy everything from `headers` into the `include` directory
4. copy all `libonnxruntime.so` libraries from `jni` to `lib` directory
5. copy `LICENSE` if it changed from the git repo: <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>
