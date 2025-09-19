# QNN LiteRT Delegate from Qualcomm (current version: 2.38.0)

This source code is not by us, but from Qualcomm.
Qualcomm states it's LICENSE in `LICENSE.pdf`
We do not claim ownership of the code or the license.
The code is not modified, simply extracted.

## Steps to replicate when updating to newer version

1. extracted from the .aar file hosted on the Maven Repository [qnn-litert-delegate-2.38.0.aar](https://repo.maven.apache.org/maven2/com/qualcomm/qti/qnn-litert-delegate/2.38.0/qnn-litert-delegate-2.38.0.aar)
2. rename to `qnn-litert-delegate-2.38.0.zip` and extract
3. copy everything from `headers` into the `include` directory
4. copy all libraries from `jni` to `lib` directory
5. copy `LICENSE.pdf` if it changed
