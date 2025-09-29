#!/bin/bash

echo "Compiling eye-ai-core-rs-native-lib..."
cargo ndk -t arm64-v8a -o ../../EyeAIApp/app/src/main/jniLibs build --release

echo ""
echo "Generating kotlin bindings..."
cargo run --release -p uniffi-bindgen -- generate --library ../target/aarch64-linux-android/release/libeye_ai_core_rs_native_lib.so --language kotlin --out-dir ../../EyeAIApp/app/src/main/java/com/algorithmic_alliance/eyeaiapp --no-format

echo ""
echo "Finished!"