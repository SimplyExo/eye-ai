#include <init_helper/init_helper.hpp>

#include <format>
#include <fstream>
#include <stdexcept>

init_helper::init_helper(std::string service_name) {
    this->service_name = service_name;
    init_used = get_init_sys();

    if (init_used == UNKNOWN) 
        throw std::runtime_error("Your init system is not supported! Supported init systems are systemd and runit");
}

cmd_output init_helper::start_service() {
    std::string command_to_run;

    if (init_used == SYSTEMD)
        command_to_run = std::format(START_SERVICE_TEMPLATE_SYSTEMD, service_name);
    else if (init_used == RUNIT)
        command_to_run = std::format(START_SERVICE_TEMPLATE_RUNIT, service_name);
    
    return exec(command_to_run.c_str());
}

cmd_output init_helper::stop_service() {
    std::string command_to_run;

    if (init_used == SYSTEMD)
        command_to_run = std::format(STOP_SERVICE_TEMPLATE_SYSTEMD, service_name);
    else if (init_used == RUNIT)
        command_to_run = std::format(STOP_SERVICE_TEMPLATE_RUNIT, service_name);

    return exec(command_to_run.c_str());
}

cmd_output init_helper::restart_service() {
    std::string command_to_run;

    if (init_used == SYSTEMD)
        command_to_run = std::format(RESTART_SERVICE_TEMPLATE_SYSTEMD, service_name);
    else if (init_used == RUNIT)
        command_to_run = std::format(RESTART_SERVICE_TEMPLATE_RUNIT, service_name);

    return exec(command_to_run.c_str());
}

cmd_output init_helper::get_logs() {
        std::string command_to_run;

    if (init_used == SYSTEMD)
        command_to_run = std::format(LOGS_SERVICE_TEMPLATE_SYSTEMD, service_name);
    else if (init_used == RUNIT)
        command_to_run = std::format(LOGS_SERVICE_TEMPLATE_RUNIT, service_name);

    return exec(command_to_run.c_str());
}

INIT_SYSTEM init_helper::get_init_sys() {
    std::ifstream proc_comm("/proc/1/comm");

    std::string text;

    while (getline(proc_comm, text)) {
        if (text == "systemd") 
            return SYSTEMD;
        else if (text == "runit") 
            return RUNIT;
    }

    return UNKNOWN;
}

cmd_output init_helper::exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    FILE * pipe = popen(cmd, "r");

    cmd_output res_struct;

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }

    res_struct.text = result;
    res_struct.exit_code = WEXITSTATUS(pclose(pipe));

    return res_struct;
}
