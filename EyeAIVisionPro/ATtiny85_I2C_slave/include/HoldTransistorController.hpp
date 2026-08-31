#pragma once

/* 
   Class for managing the BC547 BJT, which keeps the device running after the user releases the 
   push-button on the EyeAIVision HAT 
*/

class HoldTransistorController {
    public:
        enum TRANSISTOR_STATE {
            TRANSISTOR_OFF,
            TRANSISTOR_ON
        };

        HoldTransistorController();

        void init();
        void set_transistor(TRANSISTOR_STATE new_state);

    private:
        TRANSISTOR_STATE current_state = TRANSISTOR_ON;
};