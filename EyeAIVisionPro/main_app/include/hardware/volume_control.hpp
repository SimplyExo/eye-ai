#pragma once

#include <QString>
#include <QStringList>
#include <QJsonObject>

#define DEFAULT_ALSA_CARD "wm8960soundcard"
#define DEFAULT_ALSA_CONTROL "Headphone"

struct volume_state {
    int volume = 0;             // 0 - 100%
    bool muted = false;
    bool success = false;
    QString error_message = "";
    QString card = DEFAULT_ALSA_CARD;
    QString control = DEFAULT_ALSA_CONTROL;

    QJsonObject toJson() const {
        QJsonObject json;
        json["card"] = card;
        json["control"] = control;
        json["volume"] = volume;
        json["muted"] = muted;
        json["success"] = success;
        if (!success && !error_message.isEmpty()) {
            json["error"] = error_message;
        }
        return json;
    }
};

class volume_control {
    public:
        explicit volume_control(const QString &card = DEFAULT_ALSA_CARD, 
                               const QString &control = DEFAULT_ALSA_CONTROL);

        volume_state get_volume();
        volume_state set_volume(int percentage);
        volume_state increase_volume(int step = 5);
        volume_state decrease_volume(int step = 5);
        volume_state set_mute(bool mute);
        volume_state toggle_mute();

        QString get_card() const;
        QString get_control() const;
        void set_card(const QString &card);
        void set_control(const QString &control);

    private:
        QString card_name;
        QString control_name;

        volume_state execute_amixer(const QStringList &args);
        volume_state parse_amixer_output(const QString &output, int exit_code);
};
