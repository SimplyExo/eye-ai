#include "button_sock/button_sock.hpp"
#include "init_helper/init_helper.hpp"
#include <controller/controller.hpp>

Q_LOGGING_CATEGORY(logGeneral, "eyeaimain")
Q_LOGGING_CATEGORY(logServices, "eyeaimain.services")
Q_LOGGING_CATEGORY(logPeripherial, "eyeaimain.peripherial")
Q_LOGGING_CATEGORY(logNetwork, "eyeaimain.network")

cmd_output controller::start_mediamtx() {
    return mediamtx_init.start_service();
}

void controller::on_quit() {
    qCInfo(logServices).noquote() << "Stopping mediamtx service";
    mediamtx_init.stop_service();

    qCInfo(logNetwork).noquote() << "Stopping button socket";
    socket_btn.stop_server();

    qCInfo(logGeneral).noquote() << "Quitting application...";
}

void controller::start_socket() {

}
