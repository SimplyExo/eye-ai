SUMMARY = "EyeAI Vision Main Application (C++20 / Qt6)"
HOMEPAGE = "https://github.com/SimplyExo/eye-ai"
LICENSE = "CLOSED"

SRC_URI = " \
    file://eyeaivision-main \
"

inherit cmake qt6-cmake externalsrc update-rc.d

# Path to CMAKE Project
EXTERNALSRC := "${@os.path.abspath(d.getVar('THISDIR') + '/../../../../main_app')}"
S = "${EXTERNALSRC}"

DEPENDS = "qtbase qthttpserver qtconnectivity"

EXTRA_OECMAKE += "-DTARGET_ARCH=armv6"

INITSCRIPT_NAME = "eyeaivision-main"
INITSCRIPT_PARAMS = "defaults 99"

do_install() {
    install -d ${D}${sysconfdir}/init.d
    install -m 0755 ${UNPACKDIR}/eyeaivision-main ${D}${sysconfdir}/init.d/eyeaivision-main

    install -d ${D}${bindir}
    install -m 0755 ${B}/EyeAIVision_Main ${D}${bindir}/EyeAIVision_Main
}

FILES:${PN} += " \
    ${bindir}/EyeAIVision_Main \
    ${sysconfdir}/init.d/eyeaivision-main \
"