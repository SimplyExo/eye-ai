#include <status_led/status_led.hpp>
#include <QFile>

status_led::status_led() {
    set_led_state(RED);     // Indicates no Client
}

bool status_led::set_led_state(LED_STATE new_state) {
    QFile f(LED_DEV);

    if (!f.open(QIODevice::WriteOnly)) {
        return false;
    }

    const QByteArray data(1, static_cast<char>(new_state));

    if (f.write(data) != 1) {
        return false;
    }

    led_state = new_state;
    return true;
}