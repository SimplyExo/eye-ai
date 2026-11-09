SUMMARY = "GPIO Device Tree Overlay for EyeAI HAT"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://button_driver_overlay.dts \
           file://i2c_overlay.dts"

S = "${UNPACKDIR}"

inherit deploy

DEPENDS = "dtc-native"

do_compile() {
    dtc -I dts -O dtb -o button_driver_overlay.dtbo button_driver_overlay.dts
    dtc -I dts -O dtb -o i2c_overlay.dtbo i2c_overlay.dts
}

do_deploy() {
    install -d ${DEPLOYDIR}/overlays
    install -m 0644 button_driver_overlay.dtbo ${DEPLOYDIR}/overlays/button_driver_overlay.dtbo
    install -m 0644 i2c_overlay.dtbo ${DEPLOYDIR}/overlays/i2c_overlay.dtbo
}

addtask deploy before do_build after do_compile
