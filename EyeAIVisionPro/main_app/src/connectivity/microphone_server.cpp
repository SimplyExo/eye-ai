#include <connectivity/microphone_server.hpp>

#include <qobject.h>

microphone_server::microphone_server(QObject *parent)
    : QTcpServer(parent) {
}

microphone_server::~microphone_server() {
    stop_server();
}

void microphone_server::start_server() {
    if (this->isListening()) {
        emit onWarning(QString("Microphone server already running on port %1").arg(this->serverPort()));
        return;
    }

    // start arecord
    setup_audio();

    // start tcp server
    if (this->listen(QHostAddress::Any, MIC_PORT)) {
        _running = true;
        emit onLog(QString("Started microphone server on port %1 successfully!").arg(MIC_PORT));
    } else {
        emit onError(QString("Error on starting microphone server: %1").arg(this->errorString()));
        stop_server();
    }
}

void microphone_server::stop_server() {
    if (_running) {
        _running = false;
    } else {
        return;
    }

    // stop tcp server
    if (this->isListening()) {
        this->close();
    }

    // stop arecord gracefully
    if (audio_process) {
        audio_process->kill();
        audio_process->waitForFinished(1000);
        audio_process->deleteLater();
        audio_process = nullptr;
    }

    // disconnect all clients
    for (QTcpSocket *client : clients) {
        client->disconnectFromHost();
        client->deleteLater();
    }
    clients.clear();

    emit onLog("Stopped microphone server");
}

void microphone_server::setup_audio() {
    audio_process = new QProcess(this);

    // reading input via arecord command
    QString program = "arecord";
    QStringList arguments;
    arguments << "-D" << "plughw:CARD=wm8960soundcard" 
              << "-f" << "S16_LE" 
              << "-r" << "44100" 
              << "-c" << "1" 
              << "-t" << "raw"
              << "--buffer-size=2048"
              << "--period-size=512";

    connect(audio_process, &QProcess::readyReadStandardOutput, this, &microphone_server::read_audio_data);
    connect(audio_process, &QProcess::errorOccurred, this, &microphone_server::process_error);

    audio_process->start(program, arguments);

    if (!audio_process->waitForStarted(3000)) {
        emit onError("Couldn't start arecord!");
    } else {
        emit onError("arecord started successfully!");
    }
}

void microphone_server::incomingConnection(qintptr socketDescriptor) {
    QTcpSocket *client = new QTcpSocket(this);
    
    if (client->setSocketDescriptor(socketDescriptor)) {
        client->setSocketOption(QAbstractSocket::LowDelayOption, 1);

        clients.append(client);
        emit clientConnected(client->peerAddress(), client->peerPort());

        connect(client, &QTcpSocket::disconnected, this, &microphone_server::client_disconnected);
    } else {
        delete client;
    }
}

void microphone_server::read_audio_data() {
    if (!audio_process) return;

    // get data from arecord
    QByteArray data = audio_process->readAllStandardOutput();

    if (data.isEmpty()) return;

    // broadcast data to all tcp clients
    for (int i = clients.size() - 1; i >= 0; --i) {
        QTcpSocket *client = clients.at(i);
        if (client->state() == QAbstractSocket::ConnectedState) {
            client->write(data);
        }
    }
}

void microphone_server::client_disconnected() {
    QTcpSocket *client = qobject_cast<QTcpSocket*>(sender());
    if (client) {
        emit clientDisconnected(client->peerAddress(), client->peerPort());
        clients.removeOne(client);
        client->deleteLater();
    }
}

void microphone_server::process_error(QProcess::ProcessError error) {
    emit onError(QString("arecord error: %1 %2").arg(error).arg(audio_process->errorString()));
}
