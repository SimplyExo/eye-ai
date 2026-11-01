SUMMARY = "Transistor disable via i2c on system shutdown"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://transistor-cutoff \
"

S = "${UNPACKDIR}"

RDEPENDS:${PN} += "bash"

do_install() {
    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/transistor-cutoff ${D}${sysconfdir}/init.d/transistor-cutoff

    install -d ${D}${sysconfdir}/rc0.d

    ln -s ../init.d/transistor-cutoff ${D}${sysconfdir}/rc0.d/K99transistor-cutoff
}

FILES:${PN} += " \
    ${sysconfdir}/init.d/transistor-cutoff \
    ${sysconfdir}/rc0.d/K99transistor-cutoff \
"