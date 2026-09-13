#!/usr/bin/env bash
set -euo pipefail

# Configuration
BUILD_DIR="build"
SYSROOT_DIR="${BUILD_DIR}/sysroot"
IMAGE_XZ="${BUILD_DIR}/raspios.img.xz"
IMAGE_RAW="${BUILD_DIR}/raspios.img"
MOUNT_DIR="${BUILD_DIR}/mnt_rootfs"
IMAGE_URL="https://downloads.raspberrypi.com/raspios_armhf/images/raspios_armhf-2026-06-19/2026-06-18-raspios-trixie-armhf.img.xz"

# Packages to install in the sysroot
PACKAGES=(
    qt6-websockets-dev
    libqt6websockets6-dev
    build-essential
    cmake
    pkg-config
    qt6-base-dev
    qt6-connectivity-dev
    qt6-httpserver-dev
    libbluetooth-dev
)

echo "[1/6] Preparing directory structure..."
mkdir -p "${BUILD_DIR}"
mkdir -p "${SYSROOT_DIR}"
mkdir -p "${MOUNT_DIR}"

echo "[2/6] Installing required host tools..."
sudo apt-get update
sudo apt-get install -y qemu-user-static systemd-container xz-utils rsync

echo "[3/6] Downloading and extracting RaspiOS image..."
if [ ! -f "${IMAGE_RAW}" ]; then
    if [ ! -f "${IMAGE_XZ}" ]; then
        curl -L "${IMAGE_URL}" -o "${IMAGE_XZ}"
    fi
    xz -d -k "${IMAGE_XZ}"
fi

echo "[4/6] Setting up loop device and copying RootFS to sysroot..."
LOOP_DEV=$(sudo losetup -fP --show "${IMAGE_RAW}")

# Cleanup hook on script termination or error
cleanup() {
    sudo umount "${MOUNT_DIR}" 2>/dev/null || true
    sudo losetup -d "${LOOP_DEV}" 2>/dev/null || true
    rmdir "${MOUNT_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# Partition 2 is the RootFS partition on Raspberry Pi OS
sudo mount "${LOOP_DEV}p2" "${MOUNT_DIR}"

echo "Copying system files to ${SYSROOT_DIR}..."
sudo rsync -aHAX "${MOUNT_DIR}/" "${SYSROOT_DIR}/"

# Unmount before the chroot phase
sudo umount "${MOUNT_DIR}"
sudo losetup -d "${LOOP_DEV}"

echo "[5/6] Integrating QEMU ARM emulation into sysroot..."
sudo cp /usr/bin/qemu-arm-static "${SYSROOT_DIR}/usr/bin/"

echo "[6/6] Entering chroot and installing packages..."
# Bind mount /etc/resolv.conf for working DNS resolution during package installation
sudo systemd-nspawn -D "${SYSROOT_DIR}" --bind-ro /etc/resolv.conf bash -c "
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y ${PACKAGES[*]}
    apt-get clean
"

echo "=== Sysroot successfully set up at ${SYSROOT_DIR} ==="
