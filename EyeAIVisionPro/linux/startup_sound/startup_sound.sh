#!/bin/sh

# enable speaker, disable headphones
amixer -c 0 sset 'Headphone' 0%
amixer -c 0 sset 'Speaker' 85%

# route PCM to speaker
amixer -c 0 sset 'Left Output Mixer PCM' on
amixer -c 0 sset 'Right Output Mixer PCM' on

# play sound
aplay -Dhw:0 /etc/sound/startup.wav

# disable speaker, enable headphones (about 14% volume)
amixer -c 0 sset 'Speaker' 0%
amixer -c 0 sset 'Headphone' 60%
