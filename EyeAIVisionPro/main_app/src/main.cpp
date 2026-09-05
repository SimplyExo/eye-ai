#include "controller/controller.hpp"
#include <init_helper/init_helper.hpp>

#include <QCoreApplication>

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    controller * main_control = new controller(&app);

    return app.exec();
}