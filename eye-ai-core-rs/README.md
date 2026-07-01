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


## Profiling eye-ai-core-rs with tracy

1. Install tracy version `0.12.2` (not needed when using `nix develop`, see 4.)

2. Compile with `enable_tracy_profiling` feature on

    ```bash
    cargo build-android -- --features enable_tracy_profiling
    ```

    (Now upload the EyeAIApp to your phone)

3. Forward port 8086 for tracy over adb

    ```bash
    adb forward tcp:8086 tcp:8086
    ```

    This allows tracy to send the tracing packets over tcp through adb right back to your PC running the tracy gui app.

4. Launch tracy and connect to EyeAIApp while its running

    (the correct tracy version gets automatically installed when you use `nix develop`)

    ```bash
    tracy
    ```

    If you dont want to use the tracy GUI, there is also the `tracy-capture` headless cli command. Just run:

    ```bash
    tracy-capture -o output.tracy
    ```

    instead. You can view the `output.tracy` in the GUI app or in a browser by going to <https://tracy.nereid.pl/> and opening that file.
