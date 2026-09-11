#!/bin/sh

set -eu

########################################
# Einstellungen
########################################

RUNIT_SERVICE_DIR="/etc/service"

DBUS_SERVICE_DIR="/etc/service/dbus"
BLUETOOTH_SERVICE_DIR="/etc/service/bluetooth"
BLUEALSA_SERVICE_DIR="/etc/service/bluealsa"

########################################
# Hilfsfunktionen
########################################

die()
{
    echo "FEHLER: $*" >&2
    exit 1
}

info()
{
    echo
    echo "==> $*"
}

########################################
# Root prüfen
########################################

if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi

########################################
# Betriebssystem prüfen
########################################

verify_os()
{
    info "Prüfe Betriebssystem..."

    [ -f /etc/os-release ] || die "/etc/os-release nicht gefunden."

    . /etc/os-release

    if [ "$VERSION_ID" != "13" ]; then
        die "Dieses Skript erwartet Debian/Raspberry Pi OS 13 (Trixie). Gefunden: $VERSION_ID"
    fi

    case "$ID" in
        debian|raspbian)
            ;;
        *)
            die "Nicht unterstütztes Betriebssystem: $ID"
            ;;
    esac
}

########################################
# runit prüfen
########################################

check_runit()
{
    info "Prüfe runit..."

    command -v sv >/dev/null 2>&1 ||
        die "sv wurde nicht gefunden. Ist runit installiert?"

    [ -d "$RUNIT_SERVICE_DIR" ] ||
        die "$RUNIT_SERVICE_DIR existiert nicht."

    echo "runit OK"
}

########################################
# Alte systemd Services deaktivieren
########################################

disable_systemd_services()
{
    info "Deaktiviere eventuell vorhandene systemd-Dienste..."

    # Nur falls systemctl vorhanden ist.
    # Es wird NICHT benötigt und nicht gestartet.

    if command -v systemctl >/dev/null 2>&1; then
        systemctl disable --now bluetooth.service 2>/dev/null || true
        systemctl disable --now dbus.service 2>/dev/null || true
        systemctl disable --now bluealsa.service 2>/dev/null || true
        systemctl disable --now bluealsa-aplay.service 2>/dev/null || true
    fi
}

########################################
# Pakete installieren
########################################

install_packages()
{
    info "Installiere Bluetooth/ALSA-Pakete..."

    apt-get update

    apt-get install -y --no-install-recommends \
        bluez \
        bluez-alsa-utils \
        dbus \
        dbus-daemon \
        alsa-utils \
        udev

    echo
    echo "Installierte Komponenten:"
    echo "  BlueZ:"
    bluetoothd --version || true

    echo
    echo "  BlueALSA:"
    bluealsa --version || true
}

########################################
# Bluetooth konfigurieren
########################################

configure_bluetooth()
{
    info "Konfiguriere Bluetooth..."

    mkdir -p /etc/bluetooth

    cat > /etc/bluetooth/main.conf <<'EOF'
[General]
Name = Raspberry Pi
Class = 0x200414
DiscoverableTimeout = 0
PairableTimeout = 0
JustWorksRepairing = always

[Policy]
AutoEnable = true
EOF

    # Bluetooth-Gerät automatisch einschalten,
    # sobald es vom Kernel erkannt wird.
    mkdir -p /etc/udev/rules.d

    cat > /etc/udev/rules.d/99-bluetooth-runit.rules <<'EOF'
SUBSYSTEM=="bluetooth", ACTION=="add", RUN+="/usr/bin/rfkill unblock bluetooth"
EOF

    udevadm control --reload-rules || true
}

########################################
# D-Bus Service
########################################

create_dbus_service()
{
    info "Erstelle runit-Service: dbus"

    mkdir -p "$DBUS_SERVICE_DIR"

    cat > "$DBUS_SERVICE_DIR/run" <<'EOF'
#!/bin/sh

exec 2>&1

# D-Bus muss seinen System-Bus als root starten.
exec /usr/bin/dbus-daemon \
    --system \
    --nofork \
    --nopidfile
EOF

    chmod 755 "$DBUS_SERVICE_DIR/run"

    # D-Bus benötigt einen sauberen Logger.
    cat > "$DBUS_SERVICE_DIR/log-run" <<'EOF'
#!/bin/sh

mkdir -p /var/log/runit/dbus

exec svlogd /var/log/runit/dbus
EOF

    chmod 755 "$DBUS_SERVICE_DIR/log-run"

    mkdir -p "$DBUS_SERVICE_DIR/log"
}

########################################
# Bluetooth Service
########################################

create_bluetooth_service()
{
    info "Erstelle runit-Service: bluetooth"

    mkdir -p "$BLUETOOTH_SERVICE_DIR"

    cat > "$BLUETOOTH_SERVICE_DIR/run" <<'EOF'
#!/bin/sh

exec 2>&1

echo "Waiting for D-Bus..."

while ! dbus-send \
    --system \
    --dest=org.freedesktop.DBus \
    --type=method_call \
    --print-reply \
    /org/freedesktop/DBus \
    org.freedesktop.DBus.ListNames \
    >/dev/null 2>&1
do
    sleep 1
done

echo "D-Bus is ready."

# Bluetooth Controller entsperren.
rfkill unblock bluetooth 2>/dev/null || true

# bluetoothd im Vordergrund laufen lassen.
exec /usr/libexec/bluetooth/bluetoothd --nodetach
EOF

    chmod 755 "$BLUETOOTH_SERVICE_DIR/run"

    cat > "$BLUETOOTH_SERVICE_DIR/log-run" <<'EOF'
#!/bin/sh

mkdir -p /var/log/runit/bluetooth

exec svlogd /var/log/runit/bluetooth
EOF

    chmod 755 "$BLUETOOTH_SERVICE_DIR/log-run"

    mkdir -p "$BLUETOOTH_SERVICE_DIR/log"
}

########################################
# BlueALSA Service
########################################

create_bluealsa_service()
{
    info "Erstelle runit-Service: bluealsa"

    mkdir -p "$BLUEALSA_SERVICE_DIR"

    cat > "$BLUEALSA_SERVICE_DIR/run" <<'EOF'
#!/bin/sh

exec 2>&1

echo "Waiting for D-Bus..."

while ! dbus-send \
    --system \
    --dest=org.freedesktop.DBus \
    --type=method_call \
    --print-reply \
    /org/freedesktop/DBus \
    org.freedesktop.DBus.ListNames \
    >/dev/null 2>&1
do
    sleep 1
done

echo "Waiting for bluetoothd..."

while ! bluetoothctl show >/dev/null 2>&1
do
    sleep 1
done

echo "Bluetooth is ready."

#
# BlueALSA
#
# a2dp-source:
#   Bluetooth-Geräte können Audio an den Raspberry senden.
#
# a2dp-sink:
#   Raspberry kann Audio an Bluetooth-Lautsprecher/Kopfhörer senden.
#
# Für einen Bluetooth-Lautsprecher als Ausgang ist a2dp-source
# normalerweise nicht erforderlich.
#

exec /usr/bin/bluealsa \
    --profile=a2dp \
    --syslog
EOF

    chmod 755 "$BLUEALSA_SERVICE_DIR/run"

    cat > "$BLUEALSA_SERVICE_DIR/log-run" <<'EOF'
#!/bin/sh

mkdir -p /var/log/runit/bluealsa

exec svlogd /var/log/runit/bluealsa
EOF

    chmod 755 "$BLUEALSA_SERVICE_DIR/log-run"

    mkdir -p "$BLUEALSA_SERVICE_DIR/log"
}

########################################
# Bluetooth Agent
########################################

create_bluetooth_agent()
{
    info "Erstelle Bluetooth-Pairing-Agent..."

    mkdir -p /etc/service/bluetooth-agent

    cat > /etc/service/bluetooth-agent/run <<'EOF'
#!/bin/sh

exec 2>&1

echo "Waiting for bluetoothd..."

while ! bluetoothctl show >/dev/null 2>&1
do
    sleep 1
done

echo "Bluetooth ready."

#
# bluetoothctl übernimmt den Agent.
#
# NoInputNoOutput:
# Geräte ohne Display/Tastatur können gekoppelt werden.
#

exec bluetoothctl \
    --timeout 0 \
    agent NoInputNoOutput
EOF

    chmod 755 /etc/service/bluetooth-agent/run

    cat > /etc/service/bluetooth-agent/log-run <<'EOF'
#!/bin/sh

mkdir -p /var/log/runit/bluetooth-agent

exec svlogd /var/log/runit/bluetooth-agent
EOF

    chmod 755 /etc/service/bluetooth-agent/log-run

    mkdir -p /etc/service/bluetooth-agent/log
}

########################################
# Service starten
########################################

start_services()
{
    info "Starte runit Services..."

    # Reihenfolge:
    #
    # dbus
    #   ↓
    # bluetooth
    #   ↓
    # bluealsa
    #   ↓
    # bluetooth-agent

    sv up dbus

    echo "Warte auf D-Bus..."
    sleep 2

    sv up bluetooth

    echo "Warte auf Bluetooth..."
    sleep 2

    sv up bluealsa

    echo "Warte auf BlueALSA..."
    sleep 2

    sv up bluetooth-agent
}

########################################
# Status
########################################

show_status()
{
    info "Service-Status"

    echo
    echo "D-Bus:"
    sv status dbus || true

    echo
    echo "Bluetooth:"
    sv status bluetooth || true

    echo
    echo "BlueALSA:"
    sv status bluealsa || true

    echo
    echo "Bluetooth Agent:"
    sv status bluetooth-agent || true

    echo
    echo "Bluetooth Adapter:"
    bluetoothctl list || true

    echo
    echo "Bluetooth Status:"
    bluetoothctl show || true

    echo
    echo "ALSA BlueALSA Geräte:"
    aplay -L 2>/dev/null | grep -i bluealsa || true
}

########################################
# Main
########################################

echo
echo "=============================================="
echo " Raspberry Pi Bluetooth Audio / ALSA / runit"
echo "=============================================="
echo

verify_os
check_runit
disable_systemd_services
install_packages
configure_bluetooth

create_dbus_service
create_bluetooth_service
create_bluealsa_service
create_bluetooth_agent

start_services

show_status

echo
echo "=============================================="
echo " Installation abgeschlossen."
echo "=============================================="
echo
echo "Services:"
echo "  sv status dbus"
echo "  sv status bluetooth"
echo "  sv status bluealsa"
echo "  sv status bluetooth-agent"
echo
echo "Bluetooth:"
echo "  bluetoothctl"
echo
echo "ALSA:"
echo "  aplay -L"
echo
echo "Logs:"
echo "  /var/log/runit/dbus"
echo "  /var/log/runit/bluetooth"
echo "  /var/log/runit/bluealsa"
echo "  /var/log/runit/bluetooth-agent"
echo
