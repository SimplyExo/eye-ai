#pragma once

#include <QTcpServer>
#include <QTcpSocket>
#include <QElapsedTimer>

#define TCP_PORT 3333
#define BUTTON_DEV "/dev/button"
#define CLICK_THRESHOLD 300ULL // ms

class button_sock : public QTcpServer
{
    Q_OBJECT

    public:
        enum BUTTON_STATE {
            RELEASED,
            PRESSED,
            UNKNOWN
        };

        enum CLICK_TYPE {
            NONE,
            SINGLE,
            DOUBLE,
            TRIPLE
        };

        bool start_server();
        void stop_server();

    signals:
        void clientConnected(QHostAddress address, quint16 port);
        void clientDisconnected(QHostAddress address, quint16 port);

    private slots:
        void newConnection();

    private:
        BUTTON_STATE get_button_state();

        void broadcast_click(CLICK_TYPE click);
        void update_click_detection();
        CLICK_TYPE take_detected_click();

        bool buttonWasPressed = false;
        int clickCount = 0;
        CLICK_TYPE detectedClick = NONE;

        QElapsedTimer clickTimer;
};