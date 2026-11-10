#pragma once

#include <QTcpServer>
#include <QTcpSocket>
#include <QProcess>
#include <QList>
#include <QHostAddress>
#include <hardware/volume_control.hpp>

#define HEADPHONE_PORT 4041

class headphone_server : public QTcpServer {
    Q_OBJECT

public:
    explicit headphone_server(QObject *parent = nullptr);
    ~headphone_server() override;

    void start_server();
    void stop_server();

    // ALSA Volume Control for WM8960 Headphone
    volume_state get_volume();
    volume_state set_volume(int percentage);
    volume_state increase_volume(int step = 5);
    volume_state decrease_volume(int step = 5);
    volume_state set_mute(bool mute);
    volume_state toggle_mute();
    volume_control* get_volume_control();

protected:
    void incomingConnection(qintptr socketDescriptor) override;

private slots:
    void read_client_data();
    void client_disconnected();
    void process_error(QProcess::ProcessError error);

signals:
    void clientConnected(QHostAddress address, quint16 port);
    void clientDisconnected(QHostAddress address, quint16 port);
    void onError(QString message);
    void onLog(QString message);
    void onWarning(QString message);

private:
    bool _running = false;
    QProcess *audio_process = nullptr;
    QList<QTcpSocket*> clients;
    volume_control vol_ctrl;

    void setup_audio();
};
