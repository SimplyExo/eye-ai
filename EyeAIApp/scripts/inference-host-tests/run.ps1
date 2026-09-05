param(
    [string]$JdkRoot = 'C:\Program Files\Android\Android Studio\jbr',
    [string]$KotlinLib = 'C:\Program Files\Android\Android Studio\plugins\Kotlin\kotlinc\lib',
    [ValidateRange(1, 100)][int]$Repetitions = 1
)
$ErrorActionPreference = 'Stop'
$appRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$moduleCache = Join-Path $appRoot '.gradle/caches/modules-2/files-2.1'
function CachedJar([string]$relative) {
    $jar = Get-ChildItem -LiteralPath (Join-Path $moduleCache $relative) -Recurse -Filter '*.jar' |
        Select-Object -First 1
    if (!$jar) { throw "Missing cached dependency: $relative" }
    return $jar.FullName
}
$junit = CachedJar 'junit/junit/4.13.2'
$hamcrest = CachedJar 'org.hamcrest/hamcrest-core/1.3'
$coroutines = CachedJar 'org.jetbrains.kotlinx/kotlinx-coroutines-core-jvm/1.9.0'
$classpath = "$junit;$hamcrest;$coroutines;$KotlinLib/kotlin-stdlib.jar"
$output = Join-Path $appRoot ('build/inference-host-tests/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$main = Join-Path $appRoot 'app/src/main/java/com/algorithmic_alliance/eyeaiapp'
$test = Join-Path $appRoot 'app/src/test/java/com/algorithmic_alliance/eyeaiapp'
$sources = @(Get-ChildItem -LiteralPath "$main/inference" -Filter '*.kt' | ForEach-Object FullName)
$sources += @('AIModelData.kt', 'camera/AnalysisFrame.kt', 'camera/AnalysisResults.kt',
    'camera/FrameAnalyzer.kt', 'ocr/TextBoundingBox.kt', 'runtime/EyeAIRuntimeState.kt') |
    ForEach-Object { Join-Path $main $_ }
$sources += @(Get-ChildItem -LiteralPath "$test/inference" -Filter '*.kt' | ForEach-Object FullName)
$sources += @(Get-ChildItem -LiteralPath "$test/camera" -Filter '*.kt' | ForEach-Object FullName)
$sources += Join-Path $appRoot 'app/src/androidTest/java/com/algorithmic_alliance/eyeaiapp/camera/AdaptivePipelineInstrumentationTest.kt'
$sources += @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.kt' | ForEach-Object FullName)
& "$JdkRoot/bin/java.exe" -cp "$KotlinLib/*" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler `
    -no-stdlib -no-reflect -classpath $classpath -d $output @sources
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'HOST SIMULATION: real Kotlin pipeline with Bitmap/native type doubles, NOT an Android device pass.'
$classes = @('inference.InferenceSchedulerTest', 'inference.SceneChangeMonitorTest',
    'inference.LumaSceneChangeScorerTest', 'inference.PhoneMotionMonitorTest',
    'inference.PhoneMotionScoreLogicTest', 'inference.ObjectDetectionV1PolicyTest',
    'inference.PhoneMotionLifecycleTest', 'camera.AnalysisResultsTest',
    'camera.AdaptivePipelineInstrumentationTest') |
    ForEach-Object { 'com.algorithmic_alliance.eyeaiapp.' + $_ }
for ($iteration = 1; $iteration -le $Repetitions; $iteration++) {
    Write-Output "Run $iteration/$Repetitions"
    & "$JdkRoot/bin/java.exe" -cp "$output;$classpath" org.junit.runner.JUnitCore @classes
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0
