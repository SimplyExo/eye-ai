#include <hardware/volume_control.hpp>
#include <QProcess>
#include <QRegularExpression>

volume_control::volume_control(const QString &card, const QString &control)
    : card_name(card), control_name(control) {
}

QString volume_control::get_card() const {
    return card_name;
}

QString volume_control::get_control() const {
    return control_name;
}

void volume_control::set_card(const QString &card) {
    card_name = card;
}

void volume_control::set_control(const QString &control) {
    control_name = control;
}

volume_state volume_control::get_volume() {
    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sget" << control_name;
    return execute_amixer(args);
}

volume_state volume_control::set_volume(int percentage) {
    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;

    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sset" << control_name << QString("%1%").arg(percentage);
    return execute_amixer(args);
}

volume_state volume_control::increase_volume(int step) {
    if (step <= 0) step = 5;

    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sset" << control_name << QString("%1%+").arg(step);
    return execute_amixer(args);
}

volume_state volume_control::decrease_volume(int step) {
    if (step <= 0) step = 5;

    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sset" << control_name << QString("%1%-").arg(step);
    return execute_amixer(args);
}

volume_state volume_control::set_mute(bool mute) {
    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sset" << control_name << (mute ? "mute" : "unmute");
    return execute_amixer(args);
}

volume_state volume_control::toggle_mute() {
    QStringList args;
    if (!card_name.isEmpty()) {
        args << "-c" << card_name;
    }
    args << "sset" << control_name << "toggle";
    return execute_amixer(args);
}

volume_state volume_control::execute_amixer(const QStringList &args) {
    QProcess proc;
    proc.start("amixer", args);

    if (!proc.waitForFinished(2000)) {
        if (proc.error() == QProcess::FailedToStart) {
            volume_state state;
            state.card = card_name;
            state.control = control_name;
            state.success = false;
            state.error_message = "amixer binary not found";
            return state;
        }
        proc.kill();
        proc.waitForFinished(500);
        volume_state state;
        state.card = card_name;
        state.control = control_name;
        state.success = false;
        state.error_message = "amixer process timed out";
        return state;
    }

    QString output = QString::fromUtf8(proc.readAllStandardOutput());
    QString errOutput = QString::fromUtf8(proc.readAllStandardError());
    if (proc.exitCode() != 0) {
        if (!errOutput.trimmed().isEmpty()) {
            output = errOutput;
        }
    }

    return parse_amixer_output(output, proc.exitCode());
}

volume_state volume_control::parse_amixer_output(const QString &output, int exit_code) {
    volume_state state;
    state.card = card_name;
    state.control = control_name;

    if (exit_code != 0) {
        state.success = false;
        state.error_message = output.trimmed();
        return state;
    }

    // Match volume percentage [XX%]
    QRegularExpression rxVol(R"(\[(\d+)%\])");
    auto itVol = rxVol.globalMatch(output);
    int total = 0;
    int count = 0;
    while (itVol.hasNext()) {
        auto m = itVol.next();
        total += m.captured(1).toInt();
        count++;
    }

    // Match mute status [on] / [off]
    QRegularExpression rxMute(R"(\[(on|off)\])");
    auto itMute = rxMute.globalMatch(output);
    int offCount = 0;
    int onCount = 0;
    while (itMute.hasNext()) {
        auto m = itMute.next();
        if (m.captured(1) == "off") {
            offCount++;
        } else if (m.captured(1) == "on") {
            onCount++;
        }
    }

    if (offCount > 0) {
        state.muted = true;
    } else if (onCount > 0) {
        state.muted = false;
    }

    if (count > 0) {
        state.volume = total / count;
        state.success = true;
    } else if (offCount > 0 || onCount > 0) {
        state.success = true;
    } else {
        state.success = false;
        state.error_message = "Could not parse volume information from amixer output";
    }

    return state;
}
