#pragma once

#include <configuration.h>

class LEDController {
    public: 
        enum LED_STATE {
            RED_ON,
            GREEN_ON,
            OFF
        };
        LEDController();
       
        void set_led(LED_STATE new_state);

    private: 
        LED_STATE current_state = RED_ON;   // standard value (eyeaivision not connected to phone)
        
        void init();
};
