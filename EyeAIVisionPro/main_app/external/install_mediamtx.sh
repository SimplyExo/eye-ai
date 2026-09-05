#!/bin/bash

set -e

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

ARCH=$(uname -m)
echo "CPU arch: $ARCH"

# Get init system
INIT_SYS=$(cat "/proc/1/comm")
echo "Init system: $INIT_SYS"

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

# Create service (systemd or runit)
if [[ $INIT_SYS == systemd ]] ; then
    echo Creating systemd service...
    cp ./mediamtx.service /etc/systemd/system/mediamtx.service
    systemctl daemon-reload

    echo Done!
    echo To start mediamtx for current session, run:
    echo "   sudo systemctl start mediamtx"
elif [[ $INIT_SYS == runit ]] ; then
    echo Creating runit service...
    mkdir -p /etc/sv/mediamtx
    cp ./run /etc/sv/mediamtx/run
    chmod +x /etc/sv/mediamtx/run
    ln -s /etc/sv/mediamtx /var/service/mediamtx

    echo Done!
    echo To start mediamtx for current session, run:
    echo "   sudo sv start mediamtx"
fi
