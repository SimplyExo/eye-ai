#include <cstdlib>
#include <systemd/systemd.hpp>

#include <format>

systemd::systemd(std::string service_name) {
    this->service_name = service_name;
}

cmd_output systemd::start_service() {
    std::string command_to_run = std::format(START_SERVICE_TEMPLATE, service_name);
    
    return exec(command_to_run.c_str());
}

cmd_output systemd::stop_service() {
    std::string command_to_run = std::format(STOP_SERVICE_TEMPLATE, service_name);

    return exec(command_to_run.c_str());
}

cmd_output systemd::restart_service() {
    std::string command_to_run = std::format(RESTART_SERVICE_TEMPLATE, service_name);

    return exec(command_to_run.c_str());
}

cmd_output systemd::get_logs() {
    std::string command_to_run = std::format(LOGS_SERVICE_TEMPLATE, service_name);

    return exec(command_to_run.c_str());
}

cmd_output systemd::exec(const char* cmd) {
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
