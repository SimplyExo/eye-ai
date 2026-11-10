#include <hardware/battery_voltage.hpp>
#include <hardware/volume_control.hpp>
#include <qhttpserverresponse.h>
#include <QJsonObject>
#include <QJsonArray>
#include <webserver/webserver.hpp>
#include <QHttpServerResponse>
#include <QJsonDocument>
#include <QUrlQuery>
#include <connectivity/bluetooth.hpp>
#include <QTcpServer>
#include <QHttpServer>
#include <QHostAddress>

#include <logger/logger.hpp>

static QJsonObject volume_state_to_json(const volume_state &state, volume_control *volume) {
    QJsonObject json;
    json["success"] = state.success;
    json["volume"] = state.volume;
    json["muted"] = state.muted;
    if (volume) {
        json["card"] = volume->get_card();
        json["control"] = volume->get_control();
    }
    if (!state.success && !state.error_message.isEmpty()) {
        json["error"] = state.error_message;
    }
    return json;
}

webserver::webserver(init_helper* mediamtx, battery_voltage* battery, volume_control* volume) {
    this->mediamtx = mediamtx;
    this->battery = battery;
    this->volume = volume;
    
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

    // ALSA Volume REST API: /api/volume
    server.route("/api/volume", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        if (!volume) {
            QJsonObject err;
            err["success"] = false;
            err["error"] = "Volume control not initialized";
            return QHttpServerResponse(
                "application/json",
                QJsonDocument(err).toJson(QJsonDocument::Compact)
            );
        }

        volume_state state;
        QUrlQuery query(request.url().query());

        if (request.method() == QHttpServerRequest::Method::Post || request.method() == QHttpServerRequest::Method::Put) {
            QJsonDocument doc = QJsonDocument::fromJson(request.body());
            if (doc.isObject()) {
                QJsonObject obj = doc.object();
                if (obj.contains("volume")) {
                    state = volume->set_volume(obj.value("volume").toInt());
                } else if (obj.contains("value")) {
                    state = volume->set_volume(obj.value("value").toInt());
                } else if (obj.value("action").toString() == "up" || obj.contains("step_up")) {
                    int step = obj.contains("step") ? obj.value("step").toInt() : 5;
                    state = volume->increase_volume(step);
                } else if (obj.value("action").toString() == "down" || obj.contains("step_down")) {
                    int step = obj.contains("step") ? obj.value("step").toInt() : 5;
                    state = volume->decrease_volume(step);
                } else if (obj.contains("step")) {
                    int step = obj.value("step").toInt();
                    if (step >= 0) state = volume->increase_volume(step);
                    else state = volume->decrease_volume(-step);
                } else if (obj.contains("mute")) {
                    state = volume->set_mute(obj.value("mute").toBool());
                } else if (obj.value("action").toString() == "toggle_mute") {
                    state = volume->toggle_mute();
                } else {
                    state = volume->get_volume();
                }
            } else if (query.hasQueryItem("volume") || query.hasQueryItem("value")) {
                int val = query.hasQueryItem("volume") ? query.queryItemValue("volume").toInt() : query.queryItemValue("value").toInt();
                state = volume->set_volume(val);
            } else {
                state = volume->get_volume();
            }
        } else {
            // GET
            if (query.hasQueryItem("volume") || query.hasQueryItem("value")) {
                int val = query.hasQueryItem("volume") ? query.queryItemValue("volume").toInt() : query.queryItemValue("value").toInt();
                state = volume->set_volume(val);
            } else if (query.queryItemValue("action") == "up" || query.hasQueryItem("up")) {
                int step = query.hasQueryItem("step") ? query.queryItemValue("step").toInt() : 5;
                state = volume->increase_volume(step);
            } else if (query.queryItemValue("action") == "down" || query.hasQueryItem("down")) {
                int step = query.hasQueryItem("step") ? query.queryItemValue("step").toInt() : 5;
                state = volume->decrease_volume(step);
            } else if (query.hasQueryItem("step")) {
                int step = query.queryItemValue("step").toInt();
                if (step >= 0) state = volume->increase_volume(step);
                else state = volume->decrease_volume(-step);
            } else if (query.hasQueryItem("mute")) {
                QString m = query.queryItemValue("mute").toLower();
                state = volume->set_mute(m == "1" || m == "true");
            } else if (query.queryItemValue("action") == "toggle_mute") {
                state = volume->toggle_mute();
            } else {
                state = volume->get_volume();
            }
        }

        return QHttpServerResponse(
            "application/json",
            QJsonDocument(volume_state_to_json(state, volume)).toJson(QJsonDocument::Compact)
        );
    });

    // Dedicated endpoint: increase volume
    server.route("/api/volume/up", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        if (!volume) {
            QJsonObject err;
            err["success"] = false;
            err["error"] = "Volume control not initialized";
            return QHttpServerResponse("application/json", QJsonDocument(err).toJson(QJsonDocument::Compact));
        }

        int step = 5;
        QUrlQuery query(request.url().query());
        if (query.hasQueryItem("step")) {
            step = query.queryItemValue("step").toInt();
        } else if (!request.body().isEmpty()) {
            QJsonDocument doc = QJsonDocument::fromJson(request.body());
            if (doc.isObject() && doc.object().contains("step")) {
                step = doc.object().value("step").toInt();
            }
        }

        volume_state state = volume->increase_volume(step);
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(volume_state_to_json(state, volume)).toJson(QJsonDocument::Compact)
        );
    });

    // Dedicated endpoint: decrease volume
    server.route("/api/volume/down", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        if (!volume) {
            QJsonObject err;
            err["success"] = false;
            err["error"] = "Volume control not initialized";
            return QHttpServerResponse("application/json", QJsonDocument(err).toJson(QJsonDocument::Compact));
        }

        int step = 5;
        QUrlQuery query(request.url().query());
        if (query.hasQueryItem("step")) {
            step = query.queryItemValue("step").toInt();
        } else if (!request.body().isEmpty()) {
            QJsonDocument doc = QJsonDocument::fromJson(request.body());
            if (doc.isObject() && doc.object().contains("step")) {
                step = doc.object().value("step").toInt();
            }
        }

        volume_state state = volume->decrease_volume(step);
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(volume_state_to_json(state, volume)).toJson(QJsonDocument::Compact)
        );
    });

    // Dedicated endpoint: mute
    server.route("/api/volume/mute", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        if (!volume) {
            QJsonObject err;
            err["success"] = false;
            err["error"] = "Volume control not initialized";
            return QHttpServerResponse("application/json", QJsonDocument(err).toJson(QJsonDocument::Compact));
        }

        volume_state state = volume->set_mute(true);
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(volume_state_to_json(state, volume)).toJson(QJsonDocument::Compact)
        );
    });

    // Dedicated endpoint: unmute
    server.route("/api/volume/unmute", [this](const QHttpServerRequest &request) {
        emit requestReceived(request);

        if (!volume) {
            QJsonObject err;
            err["success"] = false;
            err["error"] = "Volume control not initialized";
            return QHttpServerResponse("application/json", QJsonDocument(err).toJson(QJsonDocument::Compact));
        }

        volume_state state = volume->set_mute(false);
        return QHttpServerResponse(
            "application/json",
            QJsonDocument(volume_state_to_json(state, volume)).toJson(QJsonDocument::Compact)
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
