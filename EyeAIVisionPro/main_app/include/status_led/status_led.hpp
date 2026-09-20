#pragma once

#define LED_DEV "/dev/led"

class status_led {
    public:
        enum LED_STATE {
            RED = 'R',
            GREEN = 'G'
        };

        status_led();

        bool set_led_state(LED_STATE new_state);
    private:
        LED_STATE led_state = RED;
};
