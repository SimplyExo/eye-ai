#include <logger/logger.hpp>

#include <QDebug>
#include <QLoggingCategory>

QStringList Logger::m_logs;

void Logger::install()
{
    qInstallMessageHandler(Logger::messageHandler);
}

void Logger::messageHandler(
    QtMsgType type,
    const QMessageLogContext &context,
    const QString &message)
{
    Q_UNUSED(type);

    QString log;

    if (context.category && *context.category) {
        log = QString("%1: %2")
                  .arg(context.category)
                  .arg(message);
    } else {
        log = message;
    }

    // Im Speicher speichern
    m_logs.append(log);

    // Gleichzeitig weiterhin in der Konsole ausgeben
    fprintf(stderr, "%s\n", log.toUtf8().constData());
}

QStringList Logger::logs()
{
    return m_logs;
}

QString Logger::allLogs()
{
    return m_logs.join("\n");
}