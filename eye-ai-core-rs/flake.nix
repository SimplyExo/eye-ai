{
  description = "eye-ai-core-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.android_sdk.accept_license = true;
          overlays = [ rust-overlay.overlays.default ];
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "aarch64-linux-android" ];
          extensions = [ "rust-src" ];
        };
        androidComposition = pkgs.androidenv.composeAndroidPackages {
          includeNDK = true;
          ndkVersions = [ "29.0.14206865" ];
        };
      in
      {
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
          name = "eye-ai-core-rs";

          nativeBuildInputs = with pkgs; [
            rustToolchain
            rust-analyzer
            cargo-ndk

            androidComposition.androidsdk

            pkg-config
            cmake
            ninja
            ccache

            clang-tools # clang-tidy

            tracy_0_12 # for profling
          ];

          shellHook = ''
            export ANDROID_SDK_ROOT="${androidComposition.androidsdk}/libexec/android-sdk"
            export ANDROID_NDK_ROOT="$ANDROID_SDK_ROOT/ndk-bundle"
            export NDK_HOME="$ANDROID_NDK_ROOT"
          '';
        };
      }
    );
}
