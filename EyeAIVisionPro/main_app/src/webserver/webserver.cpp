#include <qhttpserverresponse.h>
#include <QJsonObject>
#include <webserver/webserver.hpp>
#include <QHttpServerResponse>
#include <QJsonDocument>
#include <connectivity/bluetooth.hpp>

webserver::webserver() {
    register_url();
}

void webserver::register_url() {
    server.route("/api/bt_mac", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        QJsonObject json;
        json["mac_addr"] = bluetooth::get_bt_mac();
        
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

QString webserver::methodToString(QHttpServerRequest::Method method)
{
    switch (method) {
    case QHttpServerRequest::Method::Get:
        return "GET";

    case QHttpServerRequest::Method::Post:
        return "POST";

    case QHttpServerRequest::Method::Put:
        return "PUT";

    case QHttpServerRequest::Method::Delete:
        return "DELETE";

    case QHttpServerRequest::Method::Patch:
        return "PATCH";

    case QHttpServerRequest::Method::Head:
        return "HEAD";

    case QHttpServerRequest::Method::Options:
        return "OPTIONS";

    default:
        return "UNKNOWN";
    }
}
