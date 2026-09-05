#pragma once

// systemd
#include <qdebug.h>

#define START_SERVICE_TEMPLATE_SYSTEMD "systemctl --user start %1 2>&1"
#define STOP_SERVICE_TEMPLATE_SYSTEMD "systemctl --user stop %1 2>&1"
#define RESTART_SERVICE_TEMPLATE_SYSTEMD "systemctl --user restart %1 2>&1"
#define LOGS_SERVICE_TEMPLATE_SYSTEMD "systemctl -l --user status %1 2>&1"

//runit
#define START_SERVICE_TEMPLATE_RUNIT "sv start %1 2>&1"
#define STOP_SERVICE_TEMPLATE_RUNIT "sv stop %1 2>&1"
#define RESTART_SERVICE_TEMPLATE_RUNIT "sv restart %1 2>&1"
#define LOGS_SERVICE_TEMPLATE_RUNIT "sv status %1 2>&1"

struct cmd_output {
    int exit_code;
    QString text;
};

enum INIT_SYSTEM {
    SYSTEMD,
    RUNIT,
    UNKNOWN
};

class init_helper {
    public:
        init_helper(QString service_name);

        cmd_output start_service();
        cmd_output stop_service();
        cmd_output restart_service();
        cmd_output get_logs();

    private:
        static cmd_output exec(const char* cmd);
        static INIT_SYSTEM get_init_sys();
        
        QString service_name = "";
        INIT_SYSTEM init_used = UNKNOWN;
};
