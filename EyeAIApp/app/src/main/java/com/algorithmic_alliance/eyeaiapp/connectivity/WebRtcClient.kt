package com.algorithmic_alliance.eyeaiapp.connectivity

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.webrtc.DataChannel
import org.webrtc.DefaultVideoDecoderFactory
import org.webrtc.DefaultVideoEncoderFactory
import org.webrtc.EglBase
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.MediaStream
import org.webrtc.MediaStreamTrack
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RtpReceiver
import org.webrtc.RtpTransceiver
import org.webrtc.SdpObserver
import org.webrtc.SessionDescription
import org.webrtc.VideoFrame
import org.webrtc.VideoTrack
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicBoolean

class WebRtcClient(
	private val context: Context,
	private val onStreamError: (String) -> Unit = {},
	private val onFrameReceived: (Bitmap) -> Unit
) {
	companion object {
		private const val TAG = "WebRtcClient"
		private const val RECONNECT_DELAY_MS = 2000L
		private const val FRAME_TIMEOUT_SECONDS = 15L
	}

	private val executor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
	private val httpClient = OkHttpClient.Builder()
		.connectTimeout(5, TimeUnit.SECONDS)
		.readTimeout(10, TimeUnit.SECONDS)
		.build()
	private var eglBase: EglBase? = null
	private var peerConnectionFactory: PeerConnectionFactory? = null
	private var peerConnection: PeerConnection? = null
	private var videoTrack: VideoTrack? = null
	private var targetUrl: String? = null
	private val isStopped = AtomicBoolean(false)
	private val attemptState = WhepAttemptState()
	private var reconnect: ScheduledFuture<*>? = null
	private var watchdog: ScheduledFuture<*>? = null
	private var offerCall: Call? = null
	private val lastFrameAt = AtomicLong(0)

	// All signaling and lifecycle changes run on one executor. Native/HTTP callbacks
	// carry their attempt ID so a late answer can never mutate a replacement peer.
	private fun dispatch(action: () -> Unit) {
		if (isStopped.get()) return
		try {
			executor.execute { if (!isStopped.get()) action() }
		} catch (_: RejectedExecutionException) {
			// A callback raced with stop().
		}
	}

	fun start(url: String) = dispatch {
		targetUrl = url
		reconnect?.cancel(false)
		reconnect = null
		connectWebRtc()
	}

	private fun connectWebRtc() {
		val attempt = attemptState.begin()
		closeCurrentPeerConnection()
		lastFrameAt.set(System.nanoTime())
		watchdog = executor.scheduleAtFixedRate({
			if (isCurrent(attempt) &&
				System.nanoTime() - lastFrameAt.get() > TimeUnit.SECONDS.toNanos(FRAME_TIMEOUT_SECONDS)) {
				fail(attempt, "Kein Kamerabild von EyeAI-Vision empfangen. Verbindung wird erneut aufgebaut.")
			}
		}, FRAME_TIMEOUT_SECONDS, 2, TimeUnit.SECONDS)
		try {
			if (eglBase == null) eglBase = EglBase.create()
			if (peerConnectionFactory == null) {
				PeerConnectionFactory.initialize(PeerConnectionFactory.InitializationOptions
					.builder(context.applicationContext).createInitializationOptions())
				val eglContext = eglBase?.eglBaseContext
				val options = PeerConnectionFactory.Options().apply {
					networkIgnoreMask = PeerConnectionFactory.Options.ADAPTER_TYPE_CELLULAR
					// Android's default network callback omits this local-only hotspot.
					// Enumerate interfaces directly under the process's WiFi binding.
					disableNetworkMonitor = true
				}
				peerConnectionFactory = PeerConnectionFactory.builder()
					.setOptions(options)
					.setVideoEncoderFactory(DefaultVideoEncoderFactory(eglContext, true, true))
					.setVideoDecoderFactory(DefaultVideoDecoderFactory(eglContext))
					.createPeerConnectionFactory()
			}
			val config = PeerConnection.RTCConfiguration(emptyList()).apply {
				sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
				continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_ONCE
			}
			val pc = peerConnectionFactory?.createPeerConnection(config, observer(attempt))
				?: error("Failed to create PeerConnection")
			peerConnection = pc
			pc.addTransceiver(MediaStreamTrack.MediaType.MEDIA_TYPE_VIDEO,
				RtpTransceiver.RtpTransceiverInit(RtpTransceiver.RtpTransceiverDirection.RECV_ONLY))
			pc.createOffer(object : SdpObserverAdapter(attempt, "createOffer") {
				override fun onCreateSuccess(description: SessionDescription) = dispatch {
					if (isCurrent(attempt)) {
						pc.setLocalDescription(object : SdpObserverAdapter(attempt, "setLocalDescription") {
							override fun onSetSuccess() = dispatch { sendOffer(attempt) }
						}, description)
					}
				}
			}, MediaConstraints())
		} catch (e: Exception) {
			fail(attempt, "EyeAI-Vision: WebRTC konnte nicht gestartet werden: ${e.message}")
		}
	}

	private fun isCurrent(attempt: Long) = !isStopped.get() && attemptState.isCurrent(attempt)

	private fun observer(attempt: Long) = object : PeerConnection.Observer {
		override fun onSignalingChange(state: PeerConnection.SignalingState) {}
		override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) = dispatch {
			Log.i(TAG, "[WHEP] ICE: $state")
			if (state == PeerConnection.IceConnectionState.FAILED ||
				state == PeerConnection.IceConnectionState.DISCONNECTED) {
				fail(attempt, "EyeAI-Vision: Videoverbindung unterbrochen ($state).")
			}
		}
		override fun onConnectionChange(state: PeerConnection.PeerConnectionState) = dispatch {
			if (state == PeerConnection.PeerConnectionState.FAILED) {
				fail(attempt, "EyeAI-Vision: Videoverbindung fehlgeschlagen.")
			}
		}
		override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) = dispatch {
			if (state == PeerConnection.IceGatheringState.COMPLETE) sendOffer(attempt)
		}
		override fun onIceCandidate(candidate: IceCandidate) {
			Log.d(TAG, "[WHEP] ICE candidate: ${candidate.sdp}")
		}
		override fun onTrack(transceiver: RtpTransceiver) = dispatch {
			if (isCurrent(attempt)) attachVideoTrack(attempt, transceiver.receiver.track() as? VideoTrack)
		}
		// Unified Plan uses onTrack; attaching here as well would deliver frames twice.
		override fun onAddStream(stream: MediaStream) {}
		override fun onRemoveStream(stream: MediaStream) {}
		override fun onDataChannel(dataChannel: DataChannel) {}
		override fun onRenegotiationNeeded() {}
		override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}
		override fun onIceConnectionReceivingChange(receiving: Boolean) {}
		override fun onAddTrack(receiver: RtpReceiver, mediaStreams: Array<out MediaStream>) {}
	}

	private fun attachVideoTrack(attempt: Long, track: VideoTrack?) {
		if (track == null || videoTrack != null) return
		videoTrack = track
		track.setEnabled(true)
		track.addSink { frame ->
			if (isCurrent(attempt)) {
				val bitmap = videoFrameToBitmap(frame)
				if (bitmap != null && isCurrent(attempt)) {
					lastFrameAt.set(System.nanoTime())
					onFrameReceived(bitmap)
				}
			}
		}
	}

	private fun sendOffer(attempt: Long) {
		if (!isCurrent(attempt)) return
		val pc = peerConnection ?: return
		// No Trickle ICE PATCH support: the first POST must contain ALL candidates.
		if (!attemptState.claimOffer(attempt,
			pc.iceGatheringState() == PeerConnection.IceGatheringState.COMPLETE,
			pc.localDescription != null)) return
		val request = Request.Builder().url(requireNotNull(targetUrl))
			.post(pc.localDescription.description.toRequestBody("application/sdp".toMediaType()))
			.header("Accept", "application/sdp").build()
		offerCall = httpClient.newCall(request).also { call ->
			call.enqueue(object : Callback {
				override fun onFailure(call: Call, e: IOException) = dispatch {
					fail(attempt, "EyeAI-Vision: Kamerastream nicht erreichbar: ${e.message}")
				}
				override fun onResponse(call: Call, response: Response) {
					response.use { res ->
						val answer = res.body.string()
						dispatch {
							if (isCurrent(attempt)) {
								if (!res.isSuccessful || answer.isBlank()) {
									fail(attempt, "EyeAI-Vision: Kamerastream meldet HTTP ${res.code}.")
								} else {
									pc.setRemoteDescription(SdpObserverAdapter(attempt, "setRemoteDescription"),
										SessionDescription(SessionDescription.Type.ANSWER, answer))
								}
							}
						}
					}
				}
			})
		}
	}

	private fun fail(attempt: Long, message: String) {
		if (!isCurrent(attempt)) return
		attemptState.invalidate()
		Log.e(TAG, "[WHEP] $message")
		onStreamError(message)
		closeCurrentPeerConnection()
		reconnect = executor.schedule({
			reconnect = null
			if (!isStopped.get()) connectWebRtc()
		}, RECONNECT_DELAY_MS, TimeUnit.MILLISECONDS)
	}

	private fun videoFrameToBitmap(frame: VideoFrame): Bitmap? {
		val i420 = frame.buffer.toI420() ?: return null
		return try {
			val width = i420.width
			val height = i420.height
			val yuv = ByteArray(width * height * 3 / 2)

			val y = i420.dataY
			val u = i420.dataU
			val v = i420.dataV

			val strideY = i420.strideY
			val strideU = i420.strideU
			val strideV = i420.strideV

			for (row in 0 until height) {
				y.position(row * strideY)
				y.get(yuv, row * width, width)
			}

			val uvOffset = width * height
			for (row in 0 until height / 2) {
				for (col in 0 until width / 2) {
					yuv[uvOffset + row * width + col * 2] = v.get(row * strideV + col)
					yuv[uvOffset + row * width + col * 2 + 1] = u.get(row * strideU + col)
				}
			}

			val out = ByteArrayOutputStream()
			val yuvImage = YuvImage(yuv, ImageFormat.NV21, width, height, null)
			val success = yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
			if (!success) return null

			val jpegBytes = out.toByteArray()
			val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null

			if (frame.rotation == 0) return bitmap

			val matrix = Matrix().apply { postRotate(frame.rotation.toFloat()) }
			Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
		} catch (e: Exception) {
			Log.e(TAG, "[WHEP] Video frame conversion failed", e)
			null
		} finally {
			i420.release()
		}
	}

	private fun closeCurrentPeerConnection() {
		watchdog?.cancel(false)
		watchdog = null
		offerCall?.cancel()
		offerCall = null
		videoTrack = null
		peerConnection?.close()
		peerConnection?.dispose()
		peerConnection = null
	}

	fun stop() {
		if (!isStopped.compareAndSet(false, true)) return
		executor.execute {
			reconnect?.cancel(false)
			closeCurrentPeerConnection()
			peerConnectionFactory?.dispose()
			peerConnectionFactory = null
			eglBase?.release()
			eglBase = null
			executor.shutdown()
		}
	}

	private open inner class SdpObserverAdapter(private val attempt: Long, private val operation: String) : SdpObserver {
		override fun onCreateSuccess(description: SessionDescription) {}
		override fun onSetSuccess() {}
		override fun onCreateFailure(error: String?) = dispatch {
			fail(attempt, "EyeAI-Vision: $operation fehlgeschlagen: $error")
		}
		override fun onSetFailure(error: String?) = dispatch {
			fail(attempt, "EyeAI-Vision: $operation fehlgeschlagen: $error")
		}
	}
}
