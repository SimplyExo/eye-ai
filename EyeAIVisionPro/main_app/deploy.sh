#!/usr/bin/env bash
set -euo pipefail

# Configuration
RASPI_USER="eyeai"
RASPI_HOST="192.168.4.1" 
TARGET_DIR="/home/eyeai/main_app"
BUILD_DIR="build_armhf"
EXECUTABLE_NAME="EyeAIVision_Main" # Replace with your actual binary name

# Check if the build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: '${BUILD_DIR}' not found. Please build the project first!"
    exit 1
fi

echo "--> Creating target directory on Raspberry Pi (if it doesn't exist)..."
ssh "${RASPI_USER}@${RASPI_HOST}" "mkdir -p ${TARGET_DIR}"

echo "--> Copying files to ${RASPI_USER}@${RASPI_HOST}:${TARGET_DIR}..."
rsync -avz --progress "${BUILD_DIR}/${EXECUTABLE_NAME}" "${RASPI_USER}@${RASPI_HOST}:${TARGET_DIR}/"

echo "--> Making binary executable..."
ssh "${RASPI_USER}@${RASPI_HOST}" "chmod +x ${TARGET_DIR}/${EXECUTABLE_NAME}"

echo "--> Executing application on Raspberry Pi..."
# Using -t allocates a pseudo-terminal for interactive output/logging
ssh -t "${RASPI_USER}@${RASPI_HOST}" "cd ${TARGET_DIR} && ./${EXECUTABLE_NAME}"

echo "=== Execution completed! ==="
