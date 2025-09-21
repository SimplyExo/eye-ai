package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.preference.PreferenceManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.content.edit

class TutorialScreen : AppCompatActivity() {
	enum class State {
		CameraPermissionExplanation,
		MicrophonePermissionExplanation,
		BluetoothPermissionExplanation
	}

	private var currentState = State.CameraPermissionExplanation

	private var permissionManager =
		PermissionManager(this, ::onCameraPermissionResult, ::onMicrophonePermissionResult, ::onBluetoothPermissionsResult)

	private var acceptBtn: Button? = null
	private var skipBtn: Button? = null

	private var explanationIcon: ImageView? = null

	private var explanationText: TextView? = null

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
		val tutorialCompleted = sharedPreferences.getBoolean(
			getString(R.string.tutorial_completed_setting),
			false
		)
		if (tutorialCompleted) {
			exitTutorial()
		}

		enableEdgeToEdge()
		setContentView(R.layout.activity_tutorial_screen)
		ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
			val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
			v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
			insets
		}

		acceptBtn = findViewById(R.id.tutorial_screen_accept_btn)
		acceptBtn?.setOnClickListener {
			continueTutorial(false)
		}

		skipBtn = findViewById(R.id.tutorial_screen_skip_btn)
		skipBtn?.setOnClickListener {
			when (currentState) {
				State.MicrophonePermissionExplanation -> continueTutorial(true)
				State.BluetoothPermissionExplanation -> continueTutorial(true)
				else -> {}
			}
		}

		explanationIcon = findViewById(R.id.tutorial_screen_explanation_icon)

		explanationText = findViewById(R.id.tutorial_screen_explanation_text)

		changeState(State.CameraPermissionExplanation)
	}

	private fun continueTutorial(skip: Boolean) {
		when (currentState) {
			State.CameraPermissionExplanation -> {
				permissionManager.requestCameraPermission()
				changeState(State.MicrophonePermissionExplanation)
			}

			State.MicrophonePermissionExplanation -> {
				if (skip)
					PreferenceManager.getDefaultSharedPreferences(this).edit(true) {
						putBoolean(getString(R.string.enable_speech_recognition_setting), false)
					}
				else
					permissionManager.requestMicrophonePermission()
				changeState(State.BluetoothPermissionExplanation)
			}
			State.BluetoothPermissionExplanation -> {
				if (skip) {
					exitTutorial()
				}
				else {
					// will exit with callback
					permissionManager.requestBluetoothPermissions()
				}
			}
		}
	}

	private fun changeState(newState: State) {
		currentState = newState

		when (currentState) {
			State.CameraPermissionExplanation -> {
				explanationIcon?.setImageDrawable(
					AppCompatResources.getDrawable(
						this,
						R.drawable.photo_camera_24px
					)
				)
				explanationText?.text = getString(R.string.tutorial_camera_permission_explanation)
				skipBtn?.visibility = GONE
			}

			State.MicrophonePermissionExplanation -> {
				explanationIcon?.setImageDrawable(
					AppCompatResources.getDrawable(
						this,
						R.drawable.mic_24px
					)
				)
				explanationText?.text =
					getString(R.string.tutorial_microphone_permission_explanation)
				skipBtn?.visibility = VISIBLE
			}

			State.BluetoothPermissionExplanation -> {
				explanationIcon?.setImageDrawable(AppCompatResources.getDrawable(this, R.drawable.headphones_24px))
				explanationText?.text = getString(R.string.tutorial_bluetooth_permission_explanation)
				skipBtn?.visibility = VISIBLE
			}
		}
	}

	private fun exitTutorial() {
		val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
		sharedPreferences.edit(commit = true) {
			putBoolean(getString(R.string.tutorial_completed_setting), true)
		}
		startActivity(Intent(this, MainActivity::class.java))
		overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
		finish()
	}

	private fun onCameraPermissionResult(isGranted: Boolean) {
		if (!isGranted)
			Log.w(EyeAIApp.APP_LOG_TAG, "Camera permission declined!")

		// we cannot ask the user again, so proceed and display notice after tutorial again
		changeState(State.MicrophonePermissionExplanation)
	}

	@Suppress("unused")
	private fun onMicrophonePermissionResult(isGranted: Boolean) {
		// nothing to do
	}

	private fun onBluetoothPermissionsResult(isGranted: Boolean) {
		exitTutorial()
	}
}