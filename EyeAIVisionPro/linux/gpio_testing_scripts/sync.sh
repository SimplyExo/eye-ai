#!/bin/bash

rsync -avz --delete \
    --exclude ".vscode" \
    --exclude "__pycache__" \
    --exclude "sync.sh" \
    --exclude "README.md" \
    ./ \
    eyeai@192.168.4.1:/home/eyeai/gpio_testing_scripts/
