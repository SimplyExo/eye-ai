package com.algorithmic_alliance.eyeaiapp.llm.statemachine

/**
 * Conversation states belong to the EyeAI runtime, not to an Activity.
 * Keeping this type in the state-machine package prevents UI recreation from
 * becoming part of the conversation lifecycle.
 */
enum class EyeAIState {
    IDLE,
    SETTINGS_MENU,
    SETTINGS_CHOICE,
    SETTINGS_ACTION,
    SETTINGS_EXTERNAL_CONFIRMATION,
}
