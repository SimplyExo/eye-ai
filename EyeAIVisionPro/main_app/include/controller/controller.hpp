#pragma once

#include "status_led/status_led.hpp"
#include <init_helper/init_helper.hpp>
#include <button_sock/button_sock.hpp>
#include <webserver/webserver.hpp>
#include <QLoggingCategory>
#include <qdebug.h>
#include <qhostaddress.h>
#include <qloggingcategory.h>
#include <qobject.h>
#include <qtmetamacros.h>

Q_DECLARE_LOGGING_CATEGORY(logGeneral)
Q_DECLARE_LOGGING_CATEGORY(logServices)
Q_DECLARE_LOGGING_CATEGORY(logPeripherial)
Q_DECLARE_LOGGING_CATEGORY(logNetwork)

class controller : public QObject {
    Q_OBJECT

    public:
        controller(QObject *parent = 0) : QObject(parent) {
            this->setParent(parent);
            qCInfo(logServices).noquote() << "Starting mediamtx service";
            cmd_output mtx_result = start_mediamtx();

            if (mtx_result.exit_code != 0) {
                qCCritical(logServices) << "CRITICAL: Couldn't start mediamtx!";
                qCCritical(logServices) << mtx_result.text;
            }

            qCInfo(logNetwork).noquote() << QString("Starting button socket on port %1").arg(TCP_PORT);
            connect(&socket_btn, &button_sock::clientConnected,
                this, [this](const QHostAddress addr, const quint16 port) {
                        auto ipv4 = QHostAddress(addr.toIPv4Address());
                        qCInfo(logServices).noquote() << QString("New client connected: %1:%2")
                            .arg(ipv4.toString())
                            .arg(port);

                        if (!led_setter.set_led_state(status_led::GREEN))
                            qCWarning(logGeneral).noquote() << "Couldn't set LED to green!";
                     });

            connect(&socket_btn, &button_sock::clientDisconnected,
                this, [this](const QHostAddress addr, const quint16 port) {
                        auto ipv4 = QHostAddress(addr.toIPv4Address());
                        qCInfo(logServices).noquote() << QString("Client Disconnected: %1:%2")
                            .arg(ipv4.toString())
                            .arg(port);
                        
                            if (!led_setter.set_led_state(status_led::RED))
                                qCWarning(logGeneral).noquote() << "Couldn't set LED to red!";
                     });

            bool sock_result = socket_btn.start_server();
            
            if (!sock_result) 
                qCCritical(logNetwork).noquote() << "Couldn't start button tcp socket!";

            qCInfo(logNetwork).noquote() << "Starting HTTP server";
            
            if (!web.start_server()) {
                qCCritical(logNetwork).noquote() << "Couldn't start HTTP server!";
            }

            connect(&web, &webserver::requestReceived, [](const QHttpServerRequest &request) {
                qCInfo(logNetwork).noquote() << webserver::methodToString(request.method()) << " | "
                 << request.url().toDisplayString() << " | " 
                 << QHostAddress(request.remoteAddress().toIPv4Address()).toString();
            });

            qCInfo(logGeneral).noquote() << "Finished initialization!";
        }

        void on_quit();
        
    private:
        init_helper mediamtx_init = init_helper("mediamtx");
        button_sock socket_btn = button_sock();
        status_led led_setter = status_led();
        webserver web = webserver(&mediamtx_init);

        cmd_output start_mediamtx();
};
