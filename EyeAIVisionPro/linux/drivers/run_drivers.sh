#!/bin/bash

set -e

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

BUILD_DIR="./build"

for ko in "$BUILD_DIR"/*.ko; do
    [ -e "$ko" ] || continue

    filename=$(basename "$ko" .ko)
    output="$OVERLAY_DST_DIR/${filename}.ko"

    echo "Starting $ko..."
    insmod $ko
done
