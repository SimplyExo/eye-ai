#include <button_sock/button_sock.hpp>

#include <QFile>
#include <QTimer>
#include <QDebug>

bool button_sock::start_server()
{
    disconnect(this, &QTcpServer::newConnection,
               this, &button_sock::newConnection);

    connect(this, &QTcpServer::newConnection,
            this, &button_sock::newConnection);

    if (!listen(QHostAddress::Any, TCP_PORT)) {
        qDebug() << "Server konnte nicht gestartet werden:"
                 << errorString();
        return false;
    }

    qDebug() << "Server läuft auf Port" << TCP_PORT;
    return true;
}

void button_sock::newConnection()
{
    while (hasPendingConnections()) {
        QTcpSocket *socket = nextPendingConnection();

        if (!socket) {
            continue;
        }

        qDebug() << "Client verbunden:"
                 << socket->peerAddress()
                 << socket->peerPort();

        emit clientConnected(socket->peerAddress(),
                             socket->peerPort());

        socket->write("Hello client\r\n");

        QTimer *timer = new QTimer(socket);
        timer->setInterval(10);

        connect(timer, &QTimer::timeout,
                this,
                [this, socket]() {
                    if (!socket ||
                        socket->state() != QAbstractSocket::ConnectedState) {
                        return;
                    }

                    update_click_detection();

                    CLICK_TYPE click = take_detected_click();

                    QByteArray data;

                    switch (click) {
                    case SINGLE:
                        qDebug() << "Single";
                        data.append('a');
                        break;

                    case DOUBLE:
                        qDebug() << "Double";
                        data.append('b');
                        break;

                    case TRIPLE:
                        qDebug() << "Triple";
                        data.append('c');
                        break;

                    case NONE:
                        return;
                    }

                    socket->write(data);
                });

        connect(socket, &QTcpSocket::disconnected,
                timer, &QTimer::stop);

        connect(socket, &QTcpSocket::disconnected,
                this,
                [this, socket]() {
                    qDebug() << "Client getrennt:"
                             << socket->peerAddress()
                             << socket->peerPort();

                    emit clientDisconnected(socket->peerAddress(),
                                            socket->peerPort());
                });

        connect(socket, &QTcpSocket::disconnected,
                socket, &QTcpSocket::deleteLater);

        timer->start();
    }
}

void button_sock::stop_server()
{
    qDebug() << "Stoppe Server...";

    close();

    const auto sockets = findChildren<QTcpSocket *>();

    for (QTcpSocket *socket : sockets) {
        if (!socket) {
            continue;
        }

        qDebug() << "Trenne Client:"
                 << socket->peerAddress()
                 << socket->peerPort();

        socket->disconnectFromHost();
    }

    qDebug() << "Server gestoppt.";
}

button_sock::BUTTON_STATE button_sock::get_button_state()
{
    QFile f(BUTTON_DEV);

    if (!f.open(QIODevice::ReadOnly)) {
        return UNKNOWN;
    }

    QByteArray data = f.read(1);

    f.close();

    if (data.isEmpty()) {
        return UNKNOWN;
    }

    return data.at(0) == '0'
               ? RELEASED
               : PRESSED;
}

void button_sock::update_click_detection()
{
    const BUTTON_STATE state = get_button_state();

    if (state == UNKNOWN) {
        return;
    }

    const qint64 now = clickTimer.elapsed();

    if (state == PRESSED && !buttonWasPressed) {
        buttonWasPressed = true;

        if (clickCount == 0) {
            clickCount = 1;
            clickTimer.restart();

            qDebug() << "Click 1";
        }
        else if (clickCount == 1 &&
                 now <= CLICK_THRESHOLD) {
            clickCount = 2;

            qDebug() << "Click 2";
        }
        else if (clickCount == 2 &&
                 now <= CLICK_THRESHOLD) {
            clickCount = 3;

            qDebug() << "Click 3";
        }
    }

    if (state == RELEASED && buttonWasPressed) {
        buttonWasPressed = false;
    }

    if (clickCount > 0 &&
        now > CLICK_THRESHOLD) {

        switch (clickCount) {
        case 1:
            detectedClick = SINGLE;
            break;

        case 2:
            detectedClick = DOUBLE;
            break;

        case 3:
        default:
            detectedClick = TRIPLE;
            break;
        }

        clickCount = 0;
    }
}

button_sock::CLICK_TYPE button_sock::take_detected_click()
{
    const CLICK_TYPE result = detectedClick;

    detectedClick = NONE;

    return result;
}
