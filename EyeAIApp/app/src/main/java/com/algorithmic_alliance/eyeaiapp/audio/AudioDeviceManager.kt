package com.algorithmic_alliance.eyeaiapp.audio

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothHeadset
import android.bluetooth.BluetoothProfile
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.NativeLib

class AudioDeviceManager(private val context: Context) : BroadcastReceiver() {


	companion object {
		private const val TAG = "AudioDeviceManager"
	}

	private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

	override fun onReceive(context: Context, intent: Intent) {
		when (intent.action) {
			AudioManager.ACTION_HEADSET_PLUG -> {
				when (intent.getIntExtra("state", -1)) {
					0 -> {
						Log.d(TAG, "Removed Headset")
						restartSpatialAudio()
					}

					1 -> {
						Log.d(TAG, "Added Headset")
						restartSpatialAudio()
					}

					else -> Log.d(TAG, "Headset state unknown")
				}


			}

			BluetoothHeadset.ACTION_CONNECTION_STATE_CHANGED -> {
				when (intent.getIntExtra(BluetoothProfile.EXTRA_STATE, -1)) {
					BluetoothProfile.STATE_CONNECTED -> {
						Log.d(TAG, "Bluetooth headset connected")
						restartSpatialAudio()
					}

					BluetoothProfile.STATE_DISCONNECTED -> {
						Log.d(TAG, "Bluetooth headset disconnected")
						restartSpatialAudio()
					}
				}
			}
		}

	}

	fun restartSpatialAudio() {
		NativeLib.destroySpatialAudio()
	}

	fun register() {
		val filter = IntentFilter().apply {
			addAction(AudioManager.ACTION_HEADSET_PLUG)
			addAction(BluetoothHeadset.ACTION_CONNECTION_STATE_CHANGED)
		}
		context.registerReceiver(this, filter)
		Log.d(TAG, "AudioDeviceManager registriert")
	}

	fun unregister() {
		try {
			context.unregisterReceiver(this)
			Log.d(TAG, "AudioDeviceManager deregistriert")
		} catch (e: IllegalArgumentException) {
			Log.w(TAG, "Receiver war bereits deregistriert", e)
		}
	}

}