set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

set(CMAKE_SYSROOT /opt/sysroot-armv6)

set(CMAKE_C_FLAGS_INIT "-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS_INIT "-march=armv6 -marm -mfpu=vfp -mfloat-abi=hard")

set(CMAKE_FIND_ROOT_PATH /opt/sysroot-armv6)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_PROGRAM_PATH
    /usr/lib/qt6/bin
    /usr/lib/qt6/libexec
)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(QT_HOST_PATH /usr/lib/qt6)
set(QT_HOST_MOC /usr/lib/qt6/libexec/moc)
set(QT_HOST_UIC /usr/lib/qt6/libexec/uic)
set(QT_HOST_RCC /usr/lib/qt6/libexec/rcc)

set(CMAKE_PREFIX_PATH
    /opt/sysroot-armv6/usr
    /opt/sysroot-armv6/usr/lib/arm-linux-gnueabihf/cmake
)
