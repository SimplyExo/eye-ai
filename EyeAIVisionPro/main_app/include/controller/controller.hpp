#pragma once

#include <init_helper/init_helper.hpp>
#include <QLoggingCategory>
#include <qloggingcategory.h>
#include <qobject.h>
#include <qtmetamacros.h>

Q_DECLARE_LOGGING_CATEGORY(logServices)
Q_DECLARE_LOGGING_CATEGORY(logPeripherial)
Q_DECLARE_LOGGING_CATEGORY(logNetwork)

class controller : public QObject {
    Q_OBJECT

    public:
        controller(QObject *parent = 0) : QObject(parent) {
            qCInfo(logServices) << "Starting mediamtx service";
            cmd_output mtx_result = start_mediamtx();

            if (mtx_result.exit_code != 0) {
                qCCritical(logServices) << "CRITICAL: Couldn't start mediamtx!";
                qCCritical(logServices) << mtx_result.text;
            }
        }
        
    private:
        init_helper mediamtx_init = init_helper("mediamtx");
        cmd_output start_mediamtx();
};
