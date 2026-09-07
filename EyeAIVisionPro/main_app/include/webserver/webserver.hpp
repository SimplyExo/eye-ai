#pragma once

#include <QHttpServer>
#include <qtmetamacros.h>

#define HTTP_PORT 8080

class webserver : public QObject {
    Q_OBJECT

    public:
        webserver();

        int start_server();

    private:
        QHttpServer server;
        quint16 port;

        void register_url();
};
