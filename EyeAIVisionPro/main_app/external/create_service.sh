#!/bin/bash

set -e

if ! [ "$EUID" -ne 0 ]
  then echo "Please run as non-root"
  exit
fi

# Get init system
INIT_SYS=$(cat "/proc/1/comm")
echo "Init system: $INIT_SYS"

# Create service (systemd or runit)
if [[ $INIT_SYS == systemd ]] ; then
    echo Creating systemd service...
    mkdir -p /home/$USER/.config/systemd/user
    cp ./mediamtx.service "/home/$USER/.config/systemd/user/mediamtx.service"
    systemctl --user daemon-reload

    echo Done!
    echo To start mediamtx for current session, run:
    echo "   systemctl --user start mediamtx"
elif [[ $INIT_SYS == runit ]] ; then
    echo "Creating runit service..."

    # Find sv
    SV_BIN="$(command -v sv)"

    if [[ -z "$SV_BIN" ]]; then
        echo "ERROR: sv not found!"
        exit 1
    fi

    # Create runit service
    mkdir -p /etc/service/mediamtx
    cp ./run /etc/service/mediamtx/run
    chmod 755 /etc/service/mediamtx/run
    chown root:root /etc/service/mediamtx/run

    # Create dedicated service user
    if ! id -u mediamtx >/dev/null 2>&1; then
        useradd --system \
                --no-create-home \
                --shell /usr/sbin/nologin \
                mediamtx
    fi

    # Allow eyeaimain to control ONLY mediamtx
    cat > /etc/sudoers.d/mediamtx <<EOF
eyeaimain ALL=(root) NOPASSWD: $SV_BIN start /etc/service/mediamtx
eyeaimain ALL=(root) NOPASSWD: $SV_BIN stop /etc/service/mediamtx
eyeaimain ALL=(root) NOPASSWD: $SV_BIN restart /etc/service/mediamtx
eyeaimain ALL=(root) NOPASSWD: $SV_BIN status /etc/service/mediamtx
EOF

    # Validate sudo configuration
    if ! visudo -c -f /etc/sudoers.d/mediamtx >/dev/null; then
        echo "ERROR: Invalid sudoers configuration!"
        rm -f /etc/sudoers.d/mediamtx
        exit 1
    fi

    chmod 440 /etc/sudoers.d/mediamtx

    echo "Done!"
    echo
    echo "User eyeaimain can now run:"
    echo "  sudo $SV_BIN start /etc/service/mediamtx"
    echo "  sudo $SV_BIN stop /etc/service/mediamtx"
    echo "  sudo $SV_BIN restart /etc/service/mediamtx"
    echo "  sudo $SV_BIN status /etc/service/mediamtx"
fi
