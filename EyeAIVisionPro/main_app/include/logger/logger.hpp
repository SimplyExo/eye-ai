#pragma once

#include <QString>
#include <QStringList>

class Logger
{
public:
    static void install();

    static QStringList logs();
    static QString allLogs();

private:
    static void messageHandler(
        QtMsgType type,
        const QMessageLogContext &context,
        const QString &message
    );

    static QStringList m_logs;
};
