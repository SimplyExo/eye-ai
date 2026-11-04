SUMMARY = "Turns off system on holding power button"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://shutdown-button \
"

inherit update-rc.d

RDEPENDS:${PN} = "bash"

INITSCRIPT_NAME = "shutdown-button"

INITSCRIPT_PARAMS = "defaults 90"

S = "${UNPACKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/shutdown-button ${D}${sysconfdir}/init.d/shutdown-button
}

FILES:${PN} += " \
    ${sysconfdir}/init.d/shutdown-button \
"
