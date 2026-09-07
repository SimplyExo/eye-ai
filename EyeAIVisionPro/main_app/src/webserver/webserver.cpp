#include <qhttpserverresponse.h>
#include <QJsonObject>
#include <webserver/webserver.hpp>
#include <QHttpServerResponse>
#include <QJsonDocument>

webserver::webserver() {
    register_url();
}

void webserver::register_url() {
    server.route("/api/hello", []() {
        QJsonObject json;
        json["message"] = "Hallo Welt";
        json["success"] = true;
        
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(json).toJson(QJsonDocument::Compact)
        );
    });
}

int webserver::start_server() {
    port = server.listen(QHostAddress::Any, HTTP_PORT);

    return port;
}
