SUMMARY = "Plays a wav file after finishing boot process"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://startup-sound \
    file://startup.wav \
"

inherit update-rc.d

RDEPENDS:${PN} = "bash"

INITSCRIPT_NAME = "startup-sound"

INITSCRIPT_PARAMS = "defaults 96"

S = "${UNPACKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/sound
    install -m 0555 ${UNPACKDIR}/startup.wav ${D}${sysconfdir}/sound/startup.wav

    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/startup-sound ${D}${sysconfdir}/init.d/startup-sound
}

FILES:${PN} += " \
    ${sysconfdir}/init.d/startup-sound \
"
