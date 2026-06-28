# eye-ai-core-rs

## Compiling eye-ai-core-rs-native-lib for Android

### Using Nix:

Enter the provided devShell by either run `nix develop` or enable `direnv` by running `direnv allow` if you have that installed.

After that simply run:
```bash
cargo build-android
```

Thats it.

### Without Nix:

1. Install rust:

    See <https://rust-lang.org/tools/install/>

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2. Install Android targets for rust toolchain:

    ```bash
    rustup target add aarch64-linux-android
    ```

3. Install requirements (only needed on linux)

    Install the alsa development package (`libasound2-dev` on debian, `alsa-lib-devel` on fedora)

4. Install Cargo NDK:

    ```bash
    cargo install cargo-ndk
    ```

5. Install Android Sdk including the Android NDK (version `29.0.14206865`)

    (at least one of these environment variables need to be set correctly: `ANDROID_NDK_ROOT` or `NDK_HOME`)

6. Compile eye-ai-core-rs-native-lib for Android:

    ```bash
    cargo build-android
    ```
