SUMMARY = "WLAN Hotspot Configuration using hostapd and dnsmasq with BusyBox init"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://hostapd-eyeai.conf \
    file://dnsmasq-eyeai.conf \
    file://hotspot-init \
"

S = "${UNPACKDIR}"

RDEPENDS:${PN} = "hostapd dnsmasq iptables iw"

do_install() {
    install -d ${D}${sysconfdir}
    install -m 0644 ${UNPACKDIR}/hostapd-eyeai.conf ${D}${sysconfdir}/hostapd-eyeai.conf
    install -m 0644 ${UNPACKDIR}/dnsmasq-eyeai.conf ${D}${sysconfdir}/dnsmasq-eyeai.conf

    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/hotspot-init ${D}${sysconfdir}/init.d/hotspot

    # force-create symlinks
    install -d ${D}${sysconfdir}/rcS.d
    ln -s ../init.d/hotspot ${D}${sysconfdir}/rcS.d/S90hotspot
}

FILES:${PN} += " \
    ${sysconfdir}/hostapd-eyeai.conf \
    ${sysconfdir}/dnsmasq-eyeai.conf \
    ${sysconfdir}/init.d/hotspot \
    ${sysconfdir}/rcS.d/S90hotspot \
"