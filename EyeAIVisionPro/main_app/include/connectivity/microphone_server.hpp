#pragma once

#include <QTcpServer>
#include <QTcpSocket>
#include <QProcess>
#include <QList>

#define MIC_PORT 4040

class microphone_server : public QTcpServer {
    Q_OBJECT

public:
    explicit microphone_server(QObject *parent = nullptr);
    ~microphone_server() override;

    void start_server();
    void stop_server();

protected:
    void incomingConnection(qintptr socketDescriptor) override;

private slots:
    void read_audio_data();
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

    void setup_audio();
};