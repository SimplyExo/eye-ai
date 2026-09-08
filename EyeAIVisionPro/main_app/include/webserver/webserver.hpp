#pragma once

#include <init_helper/init_helper.hpp>
#include <QHttpServer>
#include <qtmetamacros.h>

#define HTTP_PORT 8080

class webserver : public QObject {
    Q_OBJECT

    public:
        webserver(init_helper* mediamtx);

        int start_server();
        static QString methodToString(QHttpServerRequest::Method method);
        static QString get_bt_mac();

    private:
        QHttpServer server;
        quint16 port;

        init_helper* mediamtx;

        void register_url();

    signals: 
        void requestReceived(const QHttpServerRequest &request);
};
