SUMMARY = "EyeAI Button Kernel Driver"
DESCRIPTION = "Kernel module accessing the Button on the HAT"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit module

SRC_URI = "file://button_driver.c \
           file://Makefile"

S = "${UNPACKDIR}"

# Übergibt das Modul-Ziel explizit an Kbuild
EXTRA_OEMAKE = "KERNEL_SRC=${STAGING_KERNEL_DIR}"

do_configure[noexec] = "1"
do_clean[noexec] = "1"