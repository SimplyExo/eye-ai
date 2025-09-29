# eye-ai-core-rs

## Compiling eye-ai-core-rs-native-lib for Android

1. Install rust:

See <https://rust-lang.org/tools/install/>

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install Android targets for rust toolchain:

```bash
rustup target add aarch64-linux-android
```

3. Install Cargo NDK:

```bash
cargo install cargo-ndk
```

4. Compile eye-ai-core-rs-native-lib for Android:

```bash
cargo build-android
```
