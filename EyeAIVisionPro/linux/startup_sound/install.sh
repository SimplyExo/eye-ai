#!/bin/bash

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

set -e

echo Copying wav-file...
mkdir -p /etc/sound && cp startup.wav /etc/sound

echo Copying script to /etc/script...
mkdir -p /etc/script && cp startup_sound.sh /etc/script

echo Creating systemd service...
cp hotspot-sound.service /etc/systemd/system/

systemctl daemon-reload
systemctl enable hotspot-sound.service

echo Done! To test the sound run following command:
echo
echo sudo systemctl start hotspot-sound.service
echo
