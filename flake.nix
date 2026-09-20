{
  description = "dev shell to build eye-ai-core-rs and EyeAIApp";

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
        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
          targets = [ "aarch64-linux-android" ];
          extensions = [ "rust-src" ];
        };
        androidComposition = pkgs.androidenv.composeAndroidPackages {
          platformVersions = [ "37" ];
          buildToolsVersions = [ "36.0.0" ];
          includeNDK = true;
          ndkVersions = [ "29.0.14206865" ];
          cmakeVersions = [ "3.22.1" ];
        };
        androidSdk = androidComposition.androidsdk;
      in
      {
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
          name = "eye-ai";

          nativeBuildInputs = with pkgs; [
            rustToolchain
            rust-analyzer
            cargo-ndk

            androidSdk
            openjdk21

            pkg-config
            cmake
            ninja
            ccache

            clang-tools # clang-tidy

            tracy_0_12 # for profling
          ];

          ANDROID_HOME = "${androidSdk}/libexec/android-sdk";
          ANDROID_SDK_ROOT = "${androidSdk}/libexec/android-sdk";
          ANDROID_NDK_ROOT = "${androidSdk}/libexec/android-sdk/ndk-bundle/";
        };
      }
    );
}
