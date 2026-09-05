#!/bin/bash

set -e

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

ARCH=$(uname -m)
echo "CPU arch: $ARCH"

# Download files
echo "Downloading mediamtx binaries..."
if [[ $ARCH == x86_64 ]] ; then
    wget -O /tmp/mediamtx.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.20.1/mediamtx_v1.20.1_linux_amd64.tar.gz
elif [[ $ARCH == armv6 ]] ; then
    wget -O /tmp/mediamtx.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.20.1/mediamtx_v1.20.1_linux_armv6.tar.gz
elif [[ $ARCH == armv7 ]] ; then
    wget -O /tmp/mediamtx.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.20.1/mediamtx_v1.20.1_linux_armv7.tar.gz
elif [[ $ARCH == arm64 ]] ; then
    wget -O /tmp/mediamtx.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.20.1/mediamtx_v1.20.1_linux_arm64.tar.gz
fi

# Install application to /usr/bin
echo Installing mediamtx to /usr/bin...
mkdir -p /usr/bin/mediamtx
tar -xvf /tmp/mediamtx.tar.gz -C /usr/bin/mediamtx
rm /tmp/mediamtx.tar.gz

# Copy configuration
echo Creating config file in /etc/mediamtx.yml...
cp -rf ./mediamtx.yml /etc/mediamtx.yml

echo Done! Now start create_service.sh as non-root!