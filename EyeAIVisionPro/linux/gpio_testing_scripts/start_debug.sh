#!/bin/bash

# Script to start a debug session on the remote device

cd /home/eyeai/gpio_testing_scripts || exit 1

pkill -f debugpy

nohup python3 -m debugpy \
    --listen 0.0.0.0:5678 \
    --wait-for-client \
    "$1" \
    >/tmp/debugpy.log 2>&1 </dev/null &

echo Started debugpy on port 5678

exit 0