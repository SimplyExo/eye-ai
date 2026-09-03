import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.api.file.DirectoryProperty
import java.security.MessageDigest

plugins {
	alias(libs.plugins.android.application)
	alias(libs.plugins.kotlin.compose)
	id("org.jetbrains.kotlin.plugin.serialization") version "2.4.0"
}

android {
	namespace = "com.algorithmic_alliance.eyeaiapp"
	compileSdk = 37

	defaultConfig {
		applicationId = "com.algorithmic_alliance.eyeaiapp"
		minSdk = 26
		targetSdk = 38
		versionCode = 1
		versionName = "1.0"
		testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

		ndk {
			//noinspection ChromeOsAbiSupport
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
	//noinspection WrongGradleMethod
	kotlin {
		compilerOptions {
			jvmTarget = JvmTarget.fromTarget("11")
		}
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
	ndkVersion = "29.0.14206865"
	androidResources {
		noCompress.add("tflite")
		noCompress.add("onnx")
	}
	packaging {
		jniLibs.useLegacyPackaging = true
	}
	sourceSets.getByName("test").resources.directories.add("../../settings_parser/spec")
	// Golden commands are test-only assets for the on-device TFLite parity test;
	// they are never merged into the production APK assets.
sourceSets.getByName("androidTest").assets.directories.add("src/test/resources")
}

dependencies {
	implementation(libs.androidx.benchmark.common)
    implementation(libs.androidx.compose.foundation.layout)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.remote.creation.core)
    implementation(libs.androidx.core.ktx)
	implementation(libs.androidx.lifecycle.runtime.ktx)
	implementation(libs.androidx.lifecycle.service)
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
	implementation(libs.androidx.activity)

	// TFLite runtime for NLP V2 BaselineCNN
	implementation(libs.tensorflow.lite)

	// runtime only libs for tflite gpu/npu delegates
	runtimeOnly(libs.litert.gpu)
	runtimeOnly(libs.qnn.litert.delegate)

	// OCR
	implementation(libs.text.recognition)

	testImplementation(libs.junit)
	testImplementation(libs.org.json)
	androidTestImplementation(libs.androidx.test.runner)
	androidTestImplementation(libs.androidx.test.ext.junit)

	//UI
	implementation(libs.androidx.navigation.compose)
	implementation(libs.kotlinx.serialization.json)

	debugImplementation(libs.androidx.ui.tooling)
	debugImplementation(libs.androidx.ui.test.manifest)
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
					"After you ran `cargo build-android`, you are good to go!"
			)
		}
	}
}

val verifyEyeAICoreRSBuild = tasks.register<VerifyEyeAICoreRSBuildTask>("verifyEyeAICoreRSBuild") {
	description = "Checks whether eye-ai-core-rs-native-lib has been built"
}
tasks.named("preBuild").configure {
	dependsOn(verifyEyeAICoreRSBuild)
}

/** Fails both test and APK builds if a Clean-v2 production asset is altered or added accidentally. */
abstract class VerifySettingsParserAssetsTask : DefaultTask() {
	@get:InputDirectory
	abstract val assetDirectory: DirectoryProperty

	init {
		group = "verification"
		description = "Verifies the immutable production Settings Parser TFLite/tokenizer assets"
	}

	@TaskAction
	fun verify() {
		val directory = assetDirectory.get().asFile
		val expected = linkedMapOf(
			"word_operation_seed_20260812.tflite" to
				"0b992d94767c87629d4e1044d097638bcc2a85a9c4050ea3719e7c55009f0519",
			"character_speaker_seed_20260814.tflite" to
				"fd61e69b450378cf91991c3900dd966fd412492ff9e5be10db82e231989b4a79",
			"word_tokenizer.json" to
				"6f87b77a9609b82c7bec09c4450d98b892a84549edd1086e8d03419c9da64405",
			"character_tokenizer.json" to
				"6b7a7b71f686a07eb14e45c37bb99653d1855e5a26e2e8c41a5cdef5285067d0"
		)
		val contractName = "settings_parser_contract.json"
		val actualNames = directory.listFiles()?.map { it.name }?.sorted()
			?: throw GradleException("Missing Settings Parser asset directory: $directory")
		if (actualNames != (expected.keys + contractName).sorted()) {
			throw GradleException("Settings Parser asset directory contains unexpected files: $actualNames")
		}
		expected.forEach { (name, expectedSha) ->
			val file = directory.resolve(name)
			val actualSha = MessageDigest.getInstance("SHA-256")
				.digest(file.readBytes())
				.joinToString("") { "%02x".format(it.toInt() and 0xff) }
			if (actualSha != expectedSha) {
				throw GradleException("Frozen Settings Parser SHA mismatch for $name: $actualSha")
			}
		}
		val contract = directory.resolve(contractName).readText()
		if (
			!contract.contains("\"architecture\": \"SPECIALIZED_WORD_OPERATION_CHAR_SPEAKER\"") ||
			expected.values.any { hash -> !contract.contains(hash) }
		) {
			throw GradleException("Settings Parser production contract does not match frozen assets")
		}
		println("Verified 2 frozen Settings Parser TFLite models and 2 tokenizer contracts.")
	}
}

val verifySettingsParserAssets = tasks.register<VerifySettingsParserAssetsTask>("verifySettingsParserAssets") {
	assetDirectory.set(layout.projectDirectory.dir("src/main/assets/nlp-v2/settings-parser"))
}
tasks.named("preBuild").configure {
	dependsOn(verifySettingsParserAssets)
}
tasks.configureEach {
	if (name.endsWith("UnitTest")) {
		dependsOn(verifySettingsParserAssets)
	}
}
tasks.configureEach {
	if (name.startsWith("assemble") && name.endsWith("Debug")) {
		dependsOn(verifySettingsParserAssets)
	}
}
