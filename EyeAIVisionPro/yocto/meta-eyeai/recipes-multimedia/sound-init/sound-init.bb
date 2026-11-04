SUMMARY = "Initializes the alsa sound framework"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://sound-init \
    file://asound.conf \
    file://asound.state \
"

inherit update-rc.d

RDEPENDS:${PN} = "bash"

INITSCRIPT_NAME = "sound-init"

INITSCRIPT_PARAMS = "defaults 95"

S = "${UNPACKDIR}"

do_install() {
    install -d ${D}${localstatedir}/lib/alsa/
    install -m 0755 ${UNPACKDIR}/asound.state ${D}${localstatedir}/lib/alsa/asound.state

    install -d ${D}${sysconfdir}/
    install -m 0755 ${UNPACKDIR}/asound.conf ${D}${sysconfdir}/asound.conf

    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/sound-init ${D}${sysconfdir}/init.d/sound-init
}

FILES:${PN} += " \
    ${sysconfdir}/init.d/sound-init \
"
