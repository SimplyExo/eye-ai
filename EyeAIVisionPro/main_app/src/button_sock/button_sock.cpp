#include <button_sock/button_sock.hpp>

#include <QFile>
#include <QTimer>

bool button_sock::start_server()
{
    disconnect(this, &QTcpServer::newConnection,
               this, &button_sock::newConnection);

    connect(this, &QTcpServer::newConnection,
            this, &button_sock::newConnection);

    if (!listen(QHostAddress::Any, TCP_PORT)) {
        return false;
    }

    return true;
}

void button_sock::newConnection()
{
    while (hasPendingConnections()) {
        QTcpSocket *socket = nextPendingConnection();

        if (!socket) {
            continue;
        }

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
                        data.append('a');
                        break;

                    case DOUBLE:
                        data.append('b');
                        break;

                    case TRIPLE:
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
    close();

    const auto sockets = findChildren<QTcpSocket *>();

    for (QTcpSocket *socket : sockets) {
        if (!socket) {
            continue;
        }

        socket->disconnectFromHost();
    }
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
        }
        else if (clickCount == 1 &&
                 now <= CLICK_THRESHOLD) {
            clickCount = 2;
        }
        else if (clickCount == 2 &&
                 now <= CLICK_THRESHOLD) {
            clickCount = 3;
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
