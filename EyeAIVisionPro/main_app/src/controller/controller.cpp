#include "init_helper/init_helper.hpp"
#include <controller/controller.hpp>

Q_LOGGING_CATEGORY(logServices, "eyeaimain.services")
Q_LOGGING_CATEGORY(logPeripherial, "eyeaimain.peripherial")
Q_LOGGING_CATEGORY(logNetwork, "eyeaimain.network")

cmd_output controller::start_mediamtx() {
    return mediamtx_init.start_service();
}
