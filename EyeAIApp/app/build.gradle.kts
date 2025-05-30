plugins {
	alias(libs.plugins.android.application)
	alias(libs.plugins.kotlin.android)
	alias(libs.plugins.kotlin.compose)
}

android {
	namespace = "com.algorithmic_alliance.eyeaiapp"
	compileSdk = 35

	defaultConfig {
		applicationId = "com.algorithmic_alliance.eyeaiapp"
		minSdk = 26
		targetSdk = 35
		versionCode = 1
		versionName = "1.0"

		ndk {
			abiFilters += "arm64-v8a"
		}

		externalNativeBuild {
			cmake {
				targets("NativeLib")
				abiFilters("arm64-v8a")
			}
		}
	}

	buildTypes {
		release {
			isMinifyEnabled = false
			proguardFiles(
				getDefaultProguardFile("proguard-android-optimize.txt"),
				"proguard-rules.pro"
			)
			signingConfig = signingConfigs.getByName("debug")
		}
	}
	compileOptions {
		sourceCompatibility = JavaVersion.VERSION_11
		targetCompatibility = JavaVersion.VERSION_11
	}
	kotlinOptions {
		jvmTarget = "11"
	}
	buildFeatures {
		compose = true
	}
	externalNativeBuild {
		cmake {
			path = file("src/main/cpp/CMakeLists.txt")
		}
	}
	androidResources {
		noCompress.add("tflite")
		noCompress.add("onnx")
	}
}

dependencies {
	implementation(libs.androidx.core.ktx)
	implementation(libs.androidx.lifecycle.runtime.ktx)
	implementation(libs.androidx.activity.compose)
	implementation(libs.androidx.constraintlayout)
	implementation(platform(libs.androidx.compose.bom))
	implementation(libs.androidx.ui)
	implementation(libs.androidx.ui.graphics)
	implementation(libs.androidx.ui.tooling.preview)
	implementation(libs.androidx.material3)
	implementation(libs.androidx.preference.ktx)

	// Camera
	implementation(libs.androidx.camera.camera2)
	implementation(libs.androidx.camera.view)
	implementation(libs.androidx.camera.lifecycle)
	implementation(libs.androidx.camera.extensions)
	implementation(libs.material)

	// Vosk
	implementation(libs.vosk)
	implementation(libs.androidx.preference)
	implementation(libs.androidx.appcompat)

	testImplementation(libs.junit)
	androidTestImplementation(libs.androidx.junit)
	androidTestImplementation(libs.androidx.espresso.core)
	androidTestImplementation(platform(libs.androidx.compose.bom))
	androidTestImplementation(libs.androidx.ui.test.junit4)
	debugImplementation(libs.androidx.ui.tooling)
	debugImplementation(libs.androidx.ui.test.manifest)
}