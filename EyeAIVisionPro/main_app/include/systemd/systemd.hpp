#pragma once

#include <string>

#define START_SERVICE_TEMPLATE "systemctl start {} 2>&1"
#define STOP_SERVICE_TEMPLATE "systemctl stop {} 2>&1"
#define RESTART_SERVICE_TEMPLATE "systemctl restart {} 2>&1"
#define LOGS_SERVICE_TEMPLATE "systemctl -l status {} 2>&1"

struct cmd_output {
    int exit_code;
    std::string text;
};

class systemd {
    public:
        systemd(std::string service_name);

        cmd_output start_service();
        cmd_output stop_service();
        cmd_output restart_service();
        cmd_output get_logs();

    private:
        static cmd_output exec(const char* cmd);
        std::string service_name = "";
};
