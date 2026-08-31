#!/bin/bash

set -e

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

echo "Building drivers, please wait..."
make

echo "Building device tree overlays..."

OVERLAY_SRC_DIR="overlays"
OVERLAY_DST_DIR="/boot/firmware/overlays"

sudo mkdir -p "$OVERLAY_DST_DIR"

for dts in "$OVERLAY_SRC_DIR"/*.dts; do
    [ -e "$dts" ] || continue

    filename=$(basename "$dts" .dts)
    output="$OVERLAY_DST_DIR/${filename}.dtbo"

    echo "  Building $dts -> $output"

    sudo dtc -@ -I dts -O dtb \
        -o "$output" \
        "$dts"
done

echo
echo "DONE!"
echo "All device tree overlays have been built and copied to:"
echo "$OVERLAY_DST_DIR"
echo
echo "Add following lines to /boot/firmware/config.txt:"
echo 
for dts in "$OVERLAY_SRC_DIR"/*.dts; do
    filename=$(basename "$dts" .dts)
    echo "dtoverlay=${filename}"
done
echo
echo "Reboot your system and start the drivers by running 'run_drivers.sh'!"
