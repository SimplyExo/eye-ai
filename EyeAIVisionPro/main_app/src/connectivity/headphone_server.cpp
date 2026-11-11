#include <connectivity/headphone_server.hpp>
#include <qobject.h>

headphone_server::headphone_server(QObject *parent)
    : QTcpServer(parent), vol_ctrl(DEFAULT_ALSA_CARD, DEFAULT_ALSA_CONTROL) {
}

headphone_server::~headphone_server() {
    stop_server();
}

void headphone_server::start_server() {
    if (this->isListening()) {
        emit onWarning(QString("Headphone server already running on port %1").arg(this->serverPort()));
        return;
    }

    // start aplay as audio sink output
    setup_audio();

    // start tcp server on port 4041
    if (this->listen(QHostAddress::Any, HEADPHONE_PORT)) {
        _running = true;
        emit onLog(QString("Started headphone server on port %1 successfully!").arg(HEADPHONE_PORT));
    } else {
        emit onError(QString("Error on starting headphone server: %1").arg(this->errorString()));
        stop_server();
    }
}

void headphone_server::stop_server() {
    if (_running) {
        _running = false;
    } else {
        return;
    }

    // stop tcp server
    if (this->isListening()) {
        this->close();
    }

    // disconnect all clients
    for (QTcpSocket *client : clients) {
        client->disconnectFromHost();
        client->deleteLater();
    }
    clients.clear();

    // stop aplay gracefully
    if (audio_process) {
        audio_process->closeWriteChannel(); // Schließt stdin, damit aplay verbleibende Puffer abspielt
        audio_process->kill();
        audio_process->waitForFinished(1000);
        audio_process->deleteLater();
        audio_process = nullptr;
    }

    emit onLog("Stopped headphone server");
}

void headphone_server::setup_audio() {
    audio_process = new QProcess(this);

    QString program = "aplay";
    QStringList arguments;
    arguments << "-D" << "plughw:CARD=wm8960soundcard" 
              << "-f" << "S16_LE" 
              << "-r" << "48000" 
              << "-c" << "2" 
              << "-t" << "raw"
              << "--buffer-size=4096"
              << "--period-size=1024";

    connect(audio_process, &QProcess::errorOccurred, this, &headphone_server::process_error);

    audio_process->start(program, arguments);

    if (!audio_process->waitForStarted(3000)) {
        emit onError("Couldn't start aplay!");
    } else {
        emit onLog("aplay started successfully as audio sink!");
    }
}

void headphone_server::incomingConnection(qintptr socketDescriptor) {
    QTcpSocket *client = new QTcpSocket(this);
    
    if (client->setSocketDescriptor(socketDescriptor)) {
        client->setSocketOption(QAbstractSocket::LowDelayOption, 1);

        clients.append(client);
        emit clientConnected(client->peerAddress(), client->peerPort());

        connect(client, &QTcpSocket::readyRead, this, &headphone_server::read_client_data);
        connect(client, &QTcpSocket::disconnected, this, &headphone_server::client_disconnected);
    } else {
        delete client;
    }
}

void headphone_server::read_client_data() {
    QTcpSocket *client = qobject_cast<QTcpSocket*>(sender());
    if (!client || !audio_process) return;

    QByteArray data = client->readAll();
    if (data.isEmpty()) return;

    // Netzwerk-Audiodaten direkt an die Standardeingabe von aplay schreiben
    audio_process->write(data);
}

void headphone_server::client_disconnected() {
    QTcpSocket *client = qobject_cast<QTcpSocket*>(sender());
    if (client) {
        emit clientDisconnected(client->peerAddress(), client->peerPort());
        clients.removeOne(client);
        client->deleteLater();

        if (clients.isEmpty() && audio_process) {
            audio_process->kill();
            audio_process->waitForFinished(500);
            audio_process->deleteLater();
            audio_process = nullptr;

            setup_audio();
            emit onLog("Audio stream stopped, aplay reset.");
        }
    }
}

void headphone_server::process_error(QProcess::ProcessError error) {
    emit onError(QString("aplay error: %1 %2").arg(error).arg(audio_process->errorString()));
}

volume_state headphone_server::get_volume() {
    return vol_ctrl.get_volume();
}

volume_state headphone_server::set_volume(int percentage) {
    return vol_ctrl.set_volume(percentage);
}

volume_state headphone_server::increase_volume(int step) {
    return vol_ctrl.increase_volume(step);
}

volume_state headphone_server::decrease_volume(int step) {
    return vol_ctrl.decrease_volume(step);
}

volume_state headphone_server::set_mute(bool mute) {
    return vol_ctrl.set_mute(mute);
}

volume_state headphone_server::toggle_mute() {
    return vol_ctrl.toggle_mute();
}

volume_control* headphone_server::get_volume_control() {
    return &vol_ctrl;
}
