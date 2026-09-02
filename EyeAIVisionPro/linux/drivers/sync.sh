#!/bin/bash

rsync -avz --delete \
    --exclude ".vscode" \
    --exclude "__pycache__" \
    --exclude "sync.sh" \
    --exclude "README.md" \
    --exclude "compile_commands.json" \
    --exclude ".cache" \
    --exclude "build" \
    ./ \
    eyeai@192.168.4.1:/home/eyeai/drivers/
