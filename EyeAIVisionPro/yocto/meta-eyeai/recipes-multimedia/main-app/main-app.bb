SUMMARY = "EyeAI Vision Main Application (C++20 / Qt6)"
HOMEPAGE = "https://github.com/SimplyExo/eye-ai"
LICENSE = "CLOSED"

inherit cmake qt6-cmake externalsrc

# Path to CMAKE Project
# ${THISDIR} = meta-eyeai/recipes-apps/eyeai-vision/
# Mit ../../../../ gelangen wir zum Basisverzeichnis, wo main_app liegt.
EXTERNALSRC := "${@os.path.abspath(d.getVar('THISDIR') + '/../../../../main_app')}"

S = "${EXTERNALSRC}"

DEPENDS = "qtbase qthttpserver qtconnectivity"

EXTRA_OECMAKE += "-DTARGET_ARCH=armv6"

# Dateien, die ins Ziel-Image pakettiert werden sollen
FILES:${PN} += "${bindir}/EyeAIVision_Main"