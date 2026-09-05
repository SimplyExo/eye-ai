#pragma once

#include <string>

// systemd
#define START_SERVICE_TEMPLATE_SYSTEMD "systemctl start {} 2>&1"
#define STOP_SERVICE_TEMPLATE_SYSTEMD "systemctl stop {} 2>&1"
#define RESTART_SERVICE_TEMPLATE_SYSTEMD "systemctl restart {} 2>&1"
#define LOGS_SERVICE_TEMPLATE_SYSTEMD "systemctl -l status {} 2>&1"

//runit
#define START_SERVICE_TEMPLATE_RUNIT "sv up {} 2>&1"
#define STOP_SERVICE_TEMPLATE_RUNIT "sv down {} 2>&1"
#define RESTART_SERVICE_TEMPLATE_RUNIT "sv restart {} 2>&1"
#define LOGS_SERVICE_TEMPLATE_RUNIT "sv status {} 2>&1"

struct cmd_output {
    int exit_code;
    std::string text;
};

enum INIT_SYSTEM {
    SYSTEMD,
    RUNIT,
    UNKNOWN
};

class init_helper {
    public:
        init_helper(std::string service_name);

        cmd_output start_service();
        cmd_output stop_service();
        cmd_output restart_service();
        cmd_output get_logs();

    private:
        static cmd_output exec(const char* cmd);
        static INIT_SYSTEM get_init_sys();
        
        std::string service_name = "";
        INIT_SYSTEM init_used = UNKNOWN;
};
