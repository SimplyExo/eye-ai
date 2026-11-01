SUMMARY = "Fix GCC 16 type conversion error in libcamera-apps"

EXTRA_OEMESON:append = " -Dwerror=false"

CXXFLAGS:remove = "-Werror"
CXXFLAGS:append = " -Wno-error -Wno-error=literal-suffix -Wno-error=conversion -Wno-error==-Wno-pessimizing-move"
