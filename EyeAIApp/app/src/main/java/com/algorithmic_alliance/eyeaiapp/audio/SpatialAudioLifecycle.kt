package com.algorithmic_alliance.eyeaiapp.audio

import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

internal interface SpatialAudioSessionBackend {
    /** Identity/configuration only; must not perform device work. */
    fun begin(): ULong
    fun invalidate(session: ULong)
    fun create(session: ULong)
    fun destroy(session: ULong)
}

/**
 * One worker per run. Stop revokes native identity synchronously, but never waits
 * for device work. Completion always queues final destruction on that run's worker,
 * including cancellation before the coroutine body starts and exceptional exits.
 */
internal class SpatialAudioLifecycle(
    private val backend: SpatialAudioSessionBackend,
    private val updates: suspend CoroutineScope.(ULong) -> Unit,
    private val onError: (Throwable) -> Unit,
    private val executorFactory: () -> ExecutorService = {
        Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "SpatialAudio session") }
    },
) {
    private class Run(
        val id: ULong,
        val executor: ExecutorService,
        val dispatcher: ExecutorCoroutineDispatcher,
        val scope: CoroutineScope,
        val job: Job,
    )

    private val lock = Any()
    private var current: Run? = null

    fun currentSessionId(): ULong? = synchronized(lock) { current?.id }

    fun start() = synchronized(lock) {
        if (current != null) return@synchronized
        val executor = executorFactory()
        val dispatcher = executor.asCoroutineDispatcher()
        val scope = CoroutineScope(SupervisorJob() + dispatcher)
        val id = try {
            backend.begin()
        } catch (error: Throwable) {
            scope.cancel()
            dispatcher.close()
            throw error
        }
        val job = scope.launch(start = CoroutineStart.LAZY) {
            try {
                backend.create(id)
                ensureActive() // A synchronous create may have outlived stop.
                updates(id)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                onError(error)
            }
        }
        val run = Run(id, executor, dispatcher, scope, job)
        current = run
        job.invokeOnCompletion { finish(run) }
        job.start()
    }

    fun stop() = synchronized(lock) {
        val run = current ?: return@synchronized
        backend.invalidate(run.id) // Revoke BEFORE cancellation or a new begin.
        current = null
        run.job.cancel()
    }

    private fun finish(run: Run) {
        synchronized(lock) {
            if (current === run) {
                backend.invalidate(run.id)
                current = null
            }
        }
        // invokeOnCompletion can run on the stop caller for a not-yet-started job.
        // Always dispatch cleanup; never destroy/join native audio on that caller.
        run.executor.execute {
            try {
                backend.destroy(run.id)
            } catch (error: Throwable) {
                onError(error)
            } finally {
                run.scope.cancel()
                run.dispatcher.close()
            }
        }
    }
}
