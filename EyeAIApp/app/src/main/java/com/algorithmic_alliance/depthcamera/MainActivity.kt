package com.algorithmic_alliance.eyeaiapp

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.camera.view.PreviewView
import androidx.appcompat.app.AlertDialog
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class MainActivity : ComponentActivity() {
	private var permissionManager = PermissionManager(this, ::onCameraPermissionResult)

	private var cameraFrameAnalyzer: CameraFrameAnalyzer? = null

	private var cameraPreviewView: PreviewView? = null
	private var cameraPermissionNotice: LinearLayout? = null
	private var allowCameraPermission: Button? = null
	private var enableFlashlightCheckbox: CheckBox? = null
	private var switchModelButton: Button? = null

	private var depthPreviewImage: ImageView? = null

	private var performanceText: TextView? = null


	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		enableEdgeToEdge()
		setContentView(R.layout.activity_main)

		cameraPreviewView = findViewById(R.id.camera_view)

		depthPreviewImage = findViewById(R.id.depth_preview_image)

		performanceText = findViewById(R.id.performance_text)

		cameraPermissionNotice = findViewById(R.id.camera_permission_notice)

		allowCameraPermission = findViewById(R.id.allow_camera_permission_btn)
		allowCameraPermission!!.setOnClickListener {
			permissionManager.openCameraPermissionSettings()
		}

		enableFlashlightCheckbox = findViewById(R.id.enable_flashlight)
		enableFlashlightCheckbox!!.isChecked =
			eyeAIApp().cameraManager.isCameraFlashlightOn()
		enableFlashlightCheckbox!!.setOnClickListener {
			val flashlightOn =
				eyeAIApp().cameraManager.toggleCameraFlashlight()
			enableFlashlightCheckbox!!.isChecked = flashlightOn
		}

		switchModelButton = findViewById(R.id.switch_model_button)
		switchModelButton!!.text = EyeAIApp.MODELS[eyeAIApp().selectedModelIndex].name
		switchModelButton!!.setOnClickListener {
			val modelNames = EyeAIApp.MODELS.map { it.name }.toTypedArray()

			MaterialAlertDialogBuilder(this)
				.setTitle("Select Depth Model")
				.setSingleChoiceItems(
					modelNames,
					eyeAIApp().selectedModelIndex
				) { dialog, which ->
					val selectedPosition = (dialog as AlertDialog).listView.checkedItemPosition
					if (selectedPosition >= 0) {
						eyeAIApp().switchModel(selectedPosition)
						switchModelButton!!.text =
							EyeAIApp.MODELS[eyeAIApp().selectedModelIndex].name
					}

					// close popup after small delay
					Handler(Looper.getMainLooper()).postDelayed({
						dialog.dismiss()
					}, 200)
				}
				.show()
		}

		cameraFrameAnalyzer =
			CameraFrameAnalyzer(
				eyeAIApp(),
				depthPreviewImage!!,
				performanceText!!
			)

		requestCameraPermission()
		initCamera()

		window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
	}

	override fun onResume() {
		super.onResume()

		enableFlashlightCheckbox!!.isChecked =
			eyeAIApp().cameraManager.isCameraFlashlightOn()

		val cameraPermissionGranted = permissionManager.isCameraPermissionGranted()

		if (eyeAIApp().cameraManager.cameraPreview == null && cameraPermissionGranted)
			initCamera()
	}

	private fun onCameraPermissionResult(isGranted: Boolean) {
		if (isGranted) {
			cameraPermissionNotice!!.visibility = View.GONE
			initCamera()
		} else {
			cameraPermissionNotice!!.visibility = View.VISIBLE
		}
	}

	private fun eyeAIApp(): EyeAIApp {
		return application as EyeAIApp
	}

	private fun initCamera() {
		if (permissionManager.isCameraPermissionGranted()) {
			cameraPermissionNotice!!.visibility = View.GONE
			eyeAIApp().cameraManager.init(
				this,
				eyeAIApp().depthModel.getInputSize(),
				cameraPreviewView,
				cameraFrameAnalyzer!!
			)
		} else {
			cameraPermissionNotice!!.visibility = View.VISIBLE
		}
	}

	private fun requestCameraPermission() {
		if (permissionManager.isCameraPermissionGranted()) {
			cameraPermissionNotice!!.visibility = View.GONE
		} else {
			permissionManager.requestCameraPermission()
		}
	}
}
