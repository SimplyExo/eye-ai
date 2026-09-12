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

    // Zentraler Timer für die Button-Abfrage (10 ms Interval)
    QTimer *pollTimer = new QTimer(this);
    connect(pollTimer, &QTimer::timeout, this, [this]() {
        update_click_detection();
        
        CLICK_TYPE click = take_detected_click();
        if (click != NONE) {
            broadcast_click(click);
        }
    });
    pollTimer->start(10);

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

        connect(socket, &QTcpSocket::disconnected,
                this,
                [this, socket]() {
                    emit clientDisconnected(socket->peerAddress(),
                                            socket->peerPort());
                });

        connect(socket, &QTcpSocket::disconnected,
                socket, &QTcpSocket::deleteLater);
    }
}

void button_sock::broadcast_click(CLICK_TYPE click)
{
    QByteArray data;

    switch (click) {
    case SINGLE:
        data.append('1');
        break;
    case DOUBLE:
        data.append('2');
        break;
    case TRIPLE:
        data.append('3');
        break;
    case NONE:
        return;
    }

    const auto sockets = findChildren<QTcpSocket *>();
    for (QTcpSocket *socket : sockets) {
        if (socket && socket->state() == QAbstractSocket::ConnectedState) {
            socket->write(data);
        }
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

    // Flankenerkennung für Tasterdruck
    if (state == PRESSED && !buttonWasPressed) {
        buttonWasPressed = true;

        if (clickCount == 0) {
            clickCount = 1;
            clickTimer.restart();
        }
        else if (clickCount == 1 && clickTimer.elapsed() <= CLICK_THRESHOLD) {
            clickCount = 2;
            clickTimer.restart(); // Timer-Reset für das Fenster zum 3. Klick
        }
        else if (clickCount == 2 && clickTimer.elapsed() <= CLICK_THRESHOLD) {
            clickCount = 3;
            // Dreifachklick ist das Maximum – direkt auslösen!
            detectedClick = TRIPLE;
            clickCount = 0;
            return;
        }
        else {
            // Zeit abgelaufen: Als neuen Einzelklick werten
            clickCount = 1;
            clickTimer.restart();
        }
    }

    if (state == RELEASED && buttonWasPressed) {
        buttonWasPressed = false;
    }

    // Auswertung bei Zeitüberschreitung nach 1 oder 2 Klicks
    if (clickCount > 0 && clickTimer.elapsed() > CLICK_THRESHOLD) {
        if (clickCount == 1) {
            detectedClick = SINGLE;
        }
        else if (clickCount == 2) {
            detectedClick = DOUBLE;
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