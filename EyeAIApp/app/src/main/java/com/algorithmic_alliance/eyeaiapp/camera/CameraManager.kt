package com.algorithmic_alliance.eyeaiapp.camera

import android.content.Context
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.lang.ref.WeakReference
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Owns the one active local CameraX source for [EyeAIApp].
 *
 * The lifecycle owner is supplied by the foreground service, never by an
 * Activity. Preview surfaces are optional and weakly referenced so the UI can
 * disappear while ImageAnalysis continues headlessly.
 */
class CameraManager(
    private val onStateChanged: (running: Boolean, error: Throwable?) -> Unit = { _, _ -> },
) {
    private val lock = Any()
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var appContext: Context? = null
    private var mainExecutor: java.util.concurrent.Executor? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    private var previewProvider = WeakReference<Preview.SurfaceProvider>(null)
    private var previewView = WeakReference<PreviewView>(null)
    private var lifecycleOwner = WeakReference<LifecycleOwner>(null)
    private var bindingGeneration = 0L
    private var bindInFlight = false
    private var shutdown = false

    /**
     * Starts one CameraX binding. Repeated calls update only the preview
     * surface when the same service-owned binding is already active.
     */
    fun start(
        context: Context,
        owner: LifecycleOwner,
        preferredImageSize: Size,
        cameraPreviewView: PreviewView?,
        sourceSession: AnalysisSourceSession,
        onStartFailure: (Throwable) -> Unit = {},
    ) {
        // A null preview means headless operation; preserve a surface already
        // attached by a visible Activity in that case.
        if (cameraPreviewView != null) attachPreview(cameraPreviewView)

        val applicationContext = context.applicationContext
        val executor = ContextCompat.getMainExecutor(applicationContext)
        val generation: Long
        synchronized(lock) {
            if (shutdown) return
            appContext = applicationContext
            mainExecutor = executor
            lifecycleOwner = WeakReference(owner)
            if (cameraProvider != null || bindInFlight) return
            bindInFlight = true
            generation = ++bindingGeneration
        }

        val future = ProcessCameraProvider.getInstance(applicationContext)
        future.addListener(
            {
                try {
                    val provider = future.get()
                    val stillRequested = synchronized(lock) {
                        generation == bindingGeneration &&
                            sourceSession.isCurrent() &&
                            !shutdown &&
                            lifecycleOwner.get() === owner
                    }
                    if (!stillRequested) return@addListener

                    val preview = Preview.Builder()
                        .setTargetFrameRate(Range(60, 120))
                        .build()
                    val analysis = ImageAnalysis.Builder()
                        .setImageQueueDepth(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setResolutionSelector(performanceResolutionSelector(preferredImageSize))
                        .build()
                    analysis.setAnalyzer(cameraExecutor, CameraXFrameAdapter(sourceSession))

                    // CameraManager is the sole local CameraX owner. Unbind
                    // only at a new binding boundary, never on UI recreation.
                    provider.unbindAll()
                    provider.bindToLifecycle(
                        owner,
                        mostWideCameraSelector(provider),
                        analysis,
                        preview,
                    )

                    val accepted = synchronized(lock) {
                        if (generation != bindingGeneration || shutdown || !sourceSession.isCurrent()) {
                            false
                        } else {
                            cameraProvider = provider
                            imageAnalysis = analysis
                            previewUseCase = preview
                            true
                        }
                    }
                    if (!accepted) {
                        provider.unbindAll()
                        return@addListener
                    }

                    val surfaceProvider = previewProvider.get()
                    preview.surfaceProvider = surfaceProvider
                    onStateChanged(true, null)
                } catch (error: Throwable) {
                    Log.e(EyeAIApp.APP_LOG_TAG, "CameraX binding failed", error)
                    if (sourceSession.isCurrent()) {
                        onStateChanged(false, error)
                        onStartFailure(error)
                    }
                } finally {
                    synchronized(lock) {
                        if (generation == bindingGeneration) bindInFlight = false
                    }
                }
            },
            executor,
        )
    }

    /** Attaches a UI preview without changing the analysis binding. */
    fun attachPreview(cameraPreviewView: PreviewView?) {
        if (cameraPreviewView == null) return
        val provider = cameraPreviewView.surfaceProvider
        synchronized(lock) {
            previewView = WeakReference(cameraPreviewView)
            previewProvider = WeakReference(provider)
        }
        val executor = synchronized(lock) { mainExecutor }
        executor?.execute {
            val useCase = synchronized(lock) {
                previewUseCase.takeIf {
                    previewView.get() === cameraPreviewView && previewProvider.get() === provider
                }
            }
            useCase?.surfaceProvider = provider
        }
    }

    /**
     * Detaches only the requested UI surface while leaving ImageAnalysis
     * running. An old composable must not detach a newer PreviewView.
     */
    fun detachPreview(cameraPreviewView: PreviewView? = null) {
        synchronized(lock) {
            if (cameraPreviewView != null && previewView.get() !== cameraPreviewView) return
            previewView.clear()
            previewProvider.clear()
        }
        val executor = synchronized(lock) { mainExecutor }
        executor?.execute {
            val useCase = synchronized(lock) {
                previewUseCase.takeIf { previewView.get() == null }
            }
            useCase?.surfaceProvider = null
        }
    }

    /** Stops the source and unbinds CameraX, but keeps the reusable executor. */
    fun stop() {
        val provider: ProcessCameraProvider?
        synchronized(lock) {
            ++bindingGeneration
            bindInFlight = false
            provider = cameraProvider
            cameraProvider = null
            imageAnalysis = null
            previewUseCase = null
            previewView.clear()
            previewProvider.clear()
            lifecycleOwner.clear()
        }
        val executor = synchronized(lock) { mainExecutor }
        executor?.execute {
            try {
                provider?.unbindAll()
            } catch (error: Throwable) {
                Log.w(EyeAIApp.APP_LOG_TAG, "CameraX unbind failed", error)
            }
        }
        onStateChanged(false, null)
    }

    /** Stops and permanently releases the CameraX executor. */
    fun shutdown() {
        synchronized(lock) {
            if (shutdown) return
            shutdown = true
        }
        stop()
        cameraExecutor.shutdownNow()
    }
}
