#!/bin/bash

set -e

cmake -S . \
    -B build-armv6 \
    -DCMAKE_TOOLCHAIN_FILE=cmake/armv6.cmake

cmake --build build-armv6 -j$(nproc)