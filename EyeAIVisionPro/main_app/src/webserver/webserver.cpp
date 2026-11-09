#include <hardware/battery_voltage.hpp>
#include <qhttpserverresponse.h>
#include <QJsonObject>
#include <QJsonArray>
#include <webserver/webserver.hpp>
#include <QHttpServerResponse>
#include <QJsonDocument>
#include <connectivity/bluetooth.hpp>
#include <QTcpServer>
#include <QHttpServer>
#include <QHostAddress>

#include <logger/logger.hpp>

webserver::webserver(init_helper* mediamtx, battery_voltage* battery) {
    this->mediamtx = mediamtx;
    this->battery = battery;
    
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

    server.route("/api/battery", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        QJsonObject json;

        double voltage = battery->get_battery_voltage();

        json["percentage"] = battery->get_battery_percentage(voltage);
        json["voltage"] = voltage;
        
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(json).toJson(QJsonDocument::Compact)
        );
    });

    server.route("/api/logs/eyeai", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        QJsonObject json;
        QJsonArray logsArray;

        for (const QString &log : Logger::logs()) {
            logsArray.append(log);
        }

        json["logs"] = logsArray;
        
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(json).toJson(QJsonDocument::Compact)
        );
    });

    server.route("/api/logs/mediamtx", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        QJsonObject json;
        QJsonArray logsArray;
        auto logs = mediamtx->get_logs();

        for (const QString &log : logs.text.split("\n")) {
            logsArray.append(log);
        }

        json["exit_code"] = logs.exit_code;
        json["logs"] = logsArray;
        
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(json).toJson(QJsonDocument::Compact)
        );
    });
}

int webserver::start_server() {
    auto tcpServer = new QTcpServer(this);
    if (!tcpServer->listen(QHostAddress::Any, HTTP_PORT)) {
        return -1;
    }

    server.bind(tcpServer);
    int port = tcpServer->serverPort();
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
