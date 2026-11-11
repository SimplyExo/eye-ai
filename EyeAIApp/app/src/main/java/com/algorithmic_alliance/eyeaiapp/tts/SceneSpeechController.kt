package com.algorithmic_alliance.eyeaiapp.tts

import android.content.Context
import android.media.AudioFormat
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import com.algorithmic_alliance.eyeaiapp.scenenarrator.NarrationSegment
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import uniffi.NativeLib.SpatialSpeechListener
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/** Bounded synthesis: one playing section and at most one prepared successor. */
internal class SceneSpeechController(private val context: Context) {
    private data class Engine(val tts: TextToSpeech, val initialVoice: Voice?)
    private data class Clip(val pcm: ByteArray, val rate: Int) {
        val durationMs get() = pcm.size.toLong() * 1000 / (Short.SIZE_BYTES * rate)
    }
    private enum class PlaybackEvent { STARTED, COMPLETED, INTERRUPTED }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val playback = Mutex()
    private var engine: Engine? = null
    private var job: Job? = null

    @Synchronized
    fun start(
        segments: List<NarrationSegment>,
        voice: String?,
        rate: Float,
        valid: () -> Boolean,
        invalidated: Flow<Unit>,
        started: () -> Unit,
        finished: () -> Unit,
        failed: (List<NarrationSegment>) -> Unit,
        cancelled: () -> Unit,
    ) {
        job = scope.launch {
            playback.withLock {
                var index = 0
                fun checkCurrent() {
                    if (!valid()) throw CancellationException("Stale announcement")
                }

                fun fallback() = failed(segments.drop(index))
                val outcome: () -> Unit = try {
                    checkCurrent()
                    val (tts, initialVoice) = readyEngine()
                    val selected = if (voice == null) initialVoice else
                        checkNotNull(tts.voices?.firstOrNull { it.name == voice })
                    if (selected != null && tts.voice != selected) {
                        check(tts.setVoice(selected) == TextToSpeech.SUCCESS)
                    }
                    check(tts.setSpeechRate(rate) == TextToSpeech.SUCCESS)

                    supervisorScope {
                        val announcement = coroutineContext.job
                        val watch = launch {
                            invalidated.first()
                            announcement.cancel(CancellationException("Stale announcement"))
                        }
                        try {
                            var next = async { synthesize(tts, segments.first().text) }
                            for ((segmentIndex, segment) in segments.withIndex()) {
                                val clip = next.await()
                                checkCurrent()
                                val events = Channel<PlaybackEvent>(2)
                                try {
                                    val listener = object : SpatialSpeechListener {
                                        override fun onStarted() {
                                            events.trySend(PlaybackEvent.STARTED)
                                        }

                                        override fun onEnded(completed: Boolean) {
                                            events.trySend(if (completed) PlaybackEvent.COMPLETED else PlaybackEvent.INTERRUPTED)
                                        }
                                    }
                                    withContext(Dispatchers.IO) {
                                        uniffi.NativeLib.playSpatialSpeech(
                                            clip.pcm, clip.rate, segment.horizontalCenter,
                                            segment.distanceMeters ?: 2f, listener,
                                        )
                                    }
                                    if (segmentIndex + 1 < segments.size) {
                                        next = async { synthesize(tts, segments[segmentIndex + 1].text) }
                                    }
                                    withTimeout(clip.durationMs + 5_000) {
                                        while (true) {
                                            val event = events.receive()
                                            checkCurrent()
                                            when (event) {
                                                PlaybackEvent.STARTED -> if (segmentIndex == 0) started()
                                                PlaybackEvent.COMPLETED -> break
                                                PlaybackEvent.INTERRUPTED -> error("Spatial playback failed")
                                            }
                                        }
                                    }
                                } finally {
                                    events.close()
                                }
                                index++
                            }
                        } finally {
                            watch.cancel()
                        }
                    }
                    finished
                } catch (e: TimeoutCancellationException) {
                    ::fallback
                } catch (e: CancellationException) {
                    cancelled
                } catch (e: Exception) {
                    android.util.Log.w("SceneSpeech", "Spatial speech fallback", e)
                    ::fallback
                } finally {
                    withContext(NonCancellable + Dispatchers.IO) { uniffi.NativeLib.stopSpatialSpeech() }
                }
                if (isActive && valid()) outcome() else cancelled()
            }
        }
    }

    @Synchronized
    fun stop() {
        job?.cancel()
    }

    @Synchronized
    fun shutdown() {
        scope.cancel()
        CoroutineScope(Dispatchers.Main).launch {
            playback.withLock { releaseEngine() }
        }
    }

    private suspend fun readyEngine(): Engine {
        engine?.let { return it }
        withContext(Dispatchers.IO) {
            context.cacheDir.listFiles()?.forEach {
                if (it.name.startsWith("scene-speech-") && it.extension == "wav") it.delete()
            }
        }
        val ready = CompletableDeferred<Boolean>()
        val tts = TextToSpeech(context) { ready.complete(it == TextToSpeech.SUCCESS) }
        try {
            check(withTimeout(30_000) { ready.await() })
            return Engine(tts, tts.voice).also { engine = it }
        } catch (e: Exception) {
            tts.shutdown()
            throw e
        }
    }

    private fun releaseEngine() {
        engine?.tts?.let {
            it.stop()
            it.shutdown()
        }
        engine = null
    }

    private suspend fun synthesize(tts: TextToSpeech,text:String):Clip {
        val result=CompletableDeferred<Clip>()
        val bytes=ByteArrayOutputStream()
        var sampleRate=0;var format=0;var channels=0
        val id=UUID.randomUUID().toString()
        var file: File? = null
        var completed = false
        val listener = object : UtteranceProgressListener() {
            override fun onStart(utteranceId:String?) {}
            override fun onBeginSynthesis(utteranceId:String?,sampleRateInHz:Int,audioFormat:Int,channelCount:Int) {
                if(utteranceId==id)synchronized(bytes){sampleRate=sampleRateInHz;format=audioFormat;channels=channelCount}
            }
            override fun onAudioAvailable(utteranceId:String?,audio:ByteArray?) {
                if(utteranceId==id&&audio!=null)synchronized(bytes){
                    if (result.isCompleted) return
                    if(bytes.size()+audio.size>16*1024*1024)result.completeExceptionally(IllegalStateException("Speech clip too large"))
                    else bytes.write(audio)
                }
            }
            override fun onDone(utteranceId:String?) {
                if(utteranceId!=id || result.isCompleted)return
                try {synchronized(bytes){result.complete(Clip(decodePcm(bytes.toByteArray(),format,channels),sampleRate))}}
                catch(e:Exception){result.completeExceptionally(e)}
            }
            @Deprecated("Android callback") override fun onError(utteranceId:String?) {
                if(utteranceId==id)result.completeExceptionally(IllegalStateException("TTS synthesis failed"))
            }
        }
        var timedOut = false
        try {
            val target = withContext(Dispatchers.IO) {
                File.createTempFile("scene-speech-", ".wav", context.cacheDir).also { file = it }
            }
            tts.setOnUtteranceProgressListener(listener)
            check(tts.synthesizeToFile(text, Bundle(), target, id) == TextToSpeech.SUCCESS)
            val clip = withTimeout(30_000) { result.await() }
            check(clip.rate in 8000..192000 && clip.pcm.isNotEmpty())
            completed = true
            return clip
        } catch (e: TimeoutCancellationException) {
            timedOut = true
            throw e
        } finally {
            result.cancel()
            tts.setOnUtteranceProgressListener(null)
            if (timedOut) releaseEngine() else if (!completed) tts.stop()
            withContext(NonCancellable + Dispatchers.IO) { file?.delete() }
        }
    }
}

internal fun decodePcm(bytes:ByteArray,format:Int,channels:Int): ByteArray {
    require(channels in 1..2)
    val width=when(format){AudioFormat.ENCODING_PCM_8BIT->1;AudioFormat.ENCODING_PCM_16BIT->2;AudioFormat.ENCODING_PCM_FLOAT->4;else->error("Unsupported PCM")}
    require(bytes.size%(width*channels)==0)
    val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val output = ByteBuffer.allocate(bytes.size / width / channels * Short.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
    while (output.hasRemaining()) {
        var value=0.0
        repeat(channels){value+=when(width){1->((buffer.get().toInt() and 255)-128)*256.0;2->buffer.short.toDouble();else->buffer.float.toDouble().coerceIn(-1.0,1.0)*32767}}
        output.putShort((value / channels).toInt().coerceIn(-32768, 32767).toShort())
    }
    return output.array()
}
