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
				arguments("-DANDROID_STL=c++_shared")
			}
		}

		buildConfigField("String", "BUILD_TIME", "\"${System.currentTimeMillis()}\"")
		buildConfigField("String", "GIT_BRANCH", "\"${getGitBranch()}\"")
		buildConfigField("String", "GIT_TAG", "\"${getGitTag()}\"")
		buildConfigField("String", "GIT_COMMIT", "\"${getGitCommitHash()}\"")
	}

	buildTypes {
		debug {
			buildConfigField("String", "BUILD_VARIANT", "\"Debug\"")

			applicationIdSuffix = ".dev"
			versionNameSuffix = "-dev"
		}
		release {
			isMinifyEnabled = false
			proguardFiles(
				getDefaultProguardFile("proguard-android-optimize.txt"),
				"proguard-rules.pro"
			)
			signingConfig = signingConfigs.getByName("debug")

			buildConfigField("String", "BUILD_VARIANT", "\"Release\"")

			applicationIdSuffix = ".dev"
			versionNameSuffix = "-dev"
		}
		create("profiling") {
			initWith(getByName("release"))
			matchingFallbacks += listOf("release")

			buildConfigField("String", "BUILD_VARIANT", "\"Profiling\"")

			applicationIdSuffix = ".dev"
			versionNameSuffix = "-dev"

			externalNativeBuild {
				cmake {
					arguments += "-DEYE_AI_CORE_ENABLE_TRACY_PROFILER=ON"
				}
			}
		}

		create("production") {
			initWith(getByName("release"))
			matchingFallbacks += listOf("release")

			buildConfigField("String", "BUILD_VARIANT", "\"Production\"")

			applicationIdSuffix = ""
			versionNameSuffix = ""

			// TODO: signingConfig
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
		prefab = true
		compose = true
		buildConfig = true
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
	packaging {
		jniLibs.useLegacyPackaging = true
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
	implementation("com.google.android.material:material:1.13.0")

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
	implementation(libs.androidx.activity)

	// TFLite Select Ops for NLU
	implementation("org.tensorflow:tensorflow-lite:2.16.1")
	implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")

	//implementation(libs.play.services.mlkit.text.recognition.common)
	//implementation(libs.play.services.mlkit.text.recognition)

	testImplementation(libs.junit)
	androidTestImplementation(libs.androidx.junit)
	androidTestImplementation(libs.androidx.espresso.core)
	androidTestImplementation(platform(libs.androidx.compose.bom))
	androidTestImplementation(libs.androidx.ui.test.junit4)
	debugImplementation(libs.androidx.ui.tooling)
	debugImplementation(libs.androidx.ui.test.manifest)

	// OCR
	implementation(libs.text.recognition)
	implementation(libs.oboe)
}


fun getGitBranch(): String {
	return providers.exec {
		commandLine("git", "rev-parse", "--abbrev-ref", "HEAD")
	}.standardOutput.asText.get().trim()
}

fun getGitTag(): String {
	return providers.exec {
		commandLine("git", "describe", "--abbrev=0", "--tags")
	}.standardOutput.asText.get().trim()
}

fun getGitCommitHash(): String {
	return providers.exec {
		commandLine("git", "rev-parse", "--short", "HEAD")
	}.standardOutput.asText.get().trim()
}


// eye-ai-core-rs-native-lib
abstract class VerifyEyeAICoreRSBuildTask : DefaultTask() {
	@get:OutputFile
    val requiredFile = project.layout.projectDirectory.file("src/main/jniLibs/arm64-v8a/libeye_ai_core_rs_native_lib.so")

	init {
		group = "verification"
		description =
			"Verifies that the eye-ai-core-rs-native-lib rust library was build and the needed eye_ai_core_rs_native_lib.so library exists in jniLibs."
	}

	@TaskAction
	fun verify() {
		if (requiredFile.asFile.exists()) {
			println("Verified: eye-ai-core-rs-native-lib-native-lib was built and $requiredFile is present in jniLibs!")
		} else {
			throw GradleException(
				"\nERROR: eye-ai-core-rs-native-lib rust library has not been build yet!\n" +
					"Please build eye-ai-core-rs-native-lib before building EyeAIApp.\n" +
					"First follow the build instructions in `eye-ai-core-rs/README.md`.\n" +
					"After running the `eye-ai-core-rs/native-lib/build_android.sh` script, you are good to go!"
			)
		}
	}
}

val verifyEyeAICoreRSBuild by tasks.registering(VerifyEyeAICoreRSBuildTask::class)
tasks.named("preBuild") {
	dependsOn(verifyEyeAICoreRSBuild)
}