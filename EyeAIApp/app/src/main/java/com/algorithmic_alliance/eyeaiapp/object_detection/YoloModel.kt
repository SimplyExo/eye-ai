package com.algorithmic_alliance.eyeaiapp.object_detection

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import com.algorithmic_alliance.eyeaiapp.NativeLib
import java.io.File

class YoloModel(var info: YoloModelInfo) {
	fun create(context: Context)
	{
		// Erstellen einer Yolo-Instanz
		NativeLib.initYoloRuntime(info.getAsBytes(context),
			createSerializedGpuDelegateCacheDirectory(context).path,
			getModelToken(context, info.filename))
	}

	/*
	fun run(input: ByteArray): ByteArray
	{
		NativeLib
	}*/

	fun createSerializedGpuDelegateCacheDirectory(context: Context): File {
		val gpuDelegateCacheDirectory = File(context.cacheDir, "gpu_delegate_cache")
		if (!gpuDelegateCacheDirectory.exists()) gpuDelegateCacheDirectory.mkdirs()
		return gpuDelegateCacheDirectory
	}

	private fun getLastAppUpdateTime(context: Context): Long {
		try {
			val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
			return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
				packageInfo.lastUpdateTime
			} else {
				// Fallback
				File(context.packageCodePath).lastModified()
			}
		} catch (e: PackageManager.NameNotFoundException) {
			e.printStackTrace()
			return 0L
		}
	}

	private fun getModelToken(context: Context, modelFilename: String): String {
		val lastUpdateTime = getLastAppUpdateTime(context)
		return "${modelFilename}_${lastUpdateTime}"
	}

}