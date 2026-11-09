SUMMARY = "EyeAI I2C Kernel Driver for ATtiny85"
DESCRIPTION = "Kernel module for controlling LEDs, transistor and reading battery voltage via ATtiny85 on I2C"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit module

SRC_URI = "file://i2c_driver.c \
           file://Makefile"

S = "${UNPACKDIR}"

EXTRA_OEMAKE = "KERNEL_SRC=${STAGING_KERNEL_DIR}"

do_configure[noexec] = "1"
do_clean[noexec] = "1"