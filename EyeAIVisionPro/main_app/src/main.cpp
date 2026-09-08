#include <controller/controller.hpp>
#include <logger/logger.hpp>

#include <QCoreApplication>
#include <QSocketNotifier>

#include <csignal>
#include <unistd.h>

static int signalPipe[2];

static void signalHandler(int signal)
{
    ::write(signalPipe[1], &signal, sizeof(signal));
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    Logger::install();

    controller main_control(&app);

    // Pipe für Unix-Signale
    if (::pipe(signalPipe) != 0)
        return 1;

    std::signal(SIGINT, signalHandler);   // Ctrl+C
    std::signal(SIGTERM, signalHandler);  // systemd stop

    QSocketNotifier signalNotifier(
        signalPipe[0],
        QSocketNotifier::Read,
        &app
    );

    QObject::connect(
        &signalNotifier,
        &QSocketNotifier::activated,
        &app,
        [&](int) {
            int signal = 0;

            if (::read(signalPipe[0], &signal, sizeof(signal)) > 0) {
                app.quit();
            }
        }
    );

    QObject::connect(
        &app,
        &QCoreApplication::aboutToQuit,
        &main_control,
        [&main_control] {
            main_control.on_quit();
        }
    );

    return app.exec();
}
