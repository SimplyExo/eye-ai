#!/bin/bash

set -e

cmake -S . -B build-x86 -DTARGET_ARCH=x86
cmake --build build-x86 -j
