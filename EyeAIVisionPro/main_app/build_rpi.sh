#!/usr/bin/env bash
set -euo pipefail

SYSROOT_DIR="build/sysroot"
PROJECT_DIR="$(pwd)"

if [ ! -d "${SYSROOT_DIR}" ]; then
    echo "Error: Sysroot at '${SYSROOT_DIR}' not found! Please run ./setup_sysroot.sh first."
    exit 1
fi

echo "Starting build process inside ARMhf chroot environment..."

# Mount current project directory into chroot and run CMake build
sudo systemd-nspawn -D "${SYSROOT_DIR}" \
    --bind="${PROJECT_DIR}:/workspace" \
    --chdir=/workspace bash -c "
        echo '--> Configuring CMake...'
        cmake -B build_armhf -S . -DCMAKE_BUILD_TYPE=Release

        echo '--> Building application...'
        cmake --build build_armhf -j\$(nproc)
    "

echo "=== Build finished successfully! Output binaries are in ./build_armhf ==="
