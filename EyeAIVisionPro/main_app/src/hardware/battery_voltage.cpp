#include <hardware/battery_voltage.hpp>
#include <QFile>

uint8_t battery_voltage::get_raw()
{
    QFile f(BATTERY_DEV);

    if (!f.open(QIODevice::ReadOnly)) {
        return 0;
    }

    QByteArray data = f.readAll().trimmed();
    f.close();

    bool ok = false;
    uint value = data.toUInt(&ok);

    if (!ok || value > 255) {
        return 0;
    }

    return static_cast<uint8_t>(value);
}

double battery_voltage::get_battery_voltage() {
    float u12 = digit_to_volt(get_raw());
    return u12 * (R_12 + R_13) / R_12;
}

double battery_voltage::digit_to_volt(uint8_t digit) {
    return (3.3/256) * digit;
}

double battery_voltage::get_battery_percentage() {
    double v = get_battery_voltage();

    // Voltage less than 4 Volts --> below 80%
    if (v < 4) {
        return f0_80(v);
    } else {
        return f80_100(v);
    }
}

double battery_voltage::get_battery_percentage(double v) {
    if (v < 4) {
        return f0_80(v);
    } else {
        return f80_100(v);
    }
}

double battery_voltage::f0_80(double voltage) {
    return 2 * (voltage - 3.6);
}

double battery_voltage::f80_100(double voltage) {
    return voltage - 3.2;
}
