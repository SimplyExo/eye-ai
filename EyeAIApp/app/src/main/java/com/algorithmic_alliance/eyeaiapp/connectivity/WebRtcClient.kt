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
import java.util.concurrent.atomic.AtomicBoolean

class WebRtcClient(
	private val context: Context,
	private val onFrameReceived: (Bitmap) -> Unit
) {

	companion object {
		private const val TAG = "WebRtcClient"
		private const val RECONNECT_DELAY_MS = 2000L
	}

	private val executor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()

	private val httpClient: OkHttpClient = OkHttpClient.Builder()
		.hostnameVerifier { _, _ -> true }
		.connectTimeout(5, TimeUnit.SECONDS)
		.readTimeout(5, TimeUnit.SECONDS)
		.build()

	private var eglBase: EglBase? = null
	private var peerConnectionFactory: PeerConnectionFactory? = null
	private var peerConnection: PeerConnection? = null
	private var videoTrack: VideoTrack? = null

	private var targetUrl: String? = null
	private val isStopped = AtomicBoolean(false)
	private val isOfferSent = AtomicBoolean(false)
	private var frameCount = 0L

	fun start(url: String) {
		Log.i(TAG, "[WHEP] Starting WebRTC client for URL: $url")
		targetUrl = url
		isStopped.set(false)

		executor.execute {
			connectWebRtc()
		}
	}

	private fun connectWebRtc() {
		if (isStopped.get()) return
		Log.i(TAG, "[WHEP] Initializing WebRTC connection")

		try {
			closeCurrentPeerConnection()

			if (eglBase == null) {
				eglBase = EglBase.create()
			}

			if (peerConnectionFactory == null) {
				PeerConnectionFactory.initialize(
					PeerConnectionFactory.InitializationOptions
						.builder(context.applicationContext)
						.createInitializationOptions()
				)

				val eglContext = eglBase?.eglBaseContext
				val encoderFactory = DefaultVideoEncoderFactory(eglContext, true, true)
				val decoderFactory = DefaultVideoDecoderFactory(eglContext)

				val options = PeerConnectionFactory.Options().apply {
					networkIgnoreMask = PeerConnectionFactory.Options.ADAPTER_TYPE_CELLULAR
					disableNetworkMonitor = true
				}

				peerConnectionFactory = PeerConnectionFactory.builder()
					.setOptions(options)
					.setVideoEncoderFactory(encoderFactory)
					.setVideoDecoderFactory(decoderFactory)
					.createPeerConnectionFactory()
			}

			val rtcConfig = PeerConnection.RTCConfiguration(emptyList()).apply {
				sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
				continualGatheringPolicy = PeerConnection.ContinualGatheringPolicy.GATHER_ONCE
			}

			val pc = peerConnectionFactory?.createPeerConnection(rtcConfig, createPeerConnectionObserver())
				?: throw IllegalStateException("Failed to create PeerConnection")

			peerConnection = pc
			isOfferSent.set(false)

			pc.addTransceiver(
				MediaStreamTrack.MediaType.MEDIA_TYPE_VIDEO,
				RtpTransceiver.RtpTransceiverInit(RtpTransceiver.RtpTransceiverDirection.RECV_ONLY)
			)

			val constraints = MediaConstraints().apply {
				mandatory.add(MediaConstraints.KeyValuePair("OfferToReceiveVideo", "true"))
			}

			pc.createOffer(object : SdpObserverAdapter("createOffer") {
				override fun onCreateSuccess(description: SessionDescription) {
					Log.i(TAG, "[WHEP] Local offer created; setting local description")
					pc.setLocalDescription(object : SdpObserverAdapter("setLocalDescription") {
						override fun onSetSuccess() {
							Log.i(TAG, "[WHEP] Local description successfully set")
							scheduleOfferDispatch(150L)
						}
					}, description)
				}
			}, constraints)

		} catch (e: Exception) {
			Log.e(TAG, "[WHEP] Exception creating WebRTC offer", e)
			scheduleReconnect()
		}
	}

	private fun createPeerConnectionObserver(): PeerConnection.Observer {
		return object : PeerConnection.Observer {
			override fun onSignalingChange(state: PeerConnection.SignalingState) {
				Log.d(TAG, "[WHEP] SignalingState: $state")
			}

			override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
				Log.i(TAG, "[WHEP] IceConnectionState: $state")
				if (state == PeerConnection.IceConnectionState.FAILED ||
					state == PeerConnection.IceConnectionState.DISCONNECTED
				) {
					Log.w(TAG, "[WHEP] ICE disconnected or failed; scheduling reconnect")
					scheduleReconnect()
				}
			}

			override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
				Log.d(TAG, "[WHEP] IceGatheringState: $state")
				if (state == PeerConnection.IceGatheringState.COMPLETE) {
					scheduleOfferDispatch(0L)
				}
			}

			override fun onConnectionChange(state: PeerConnection.PeerConnectionState) {
				Log.i(TAG, "[WHEP] PeerConnectionState: $state")
				if (state == PeerConnection.PeerConnectionState.FAILED) {
					scheduleReconnect()
				}
			}

			override fun onIceCandidate(candidate: IceCandidate) {
				Log.d(TAG, "[WHEP] ICE candidate gathered: ${candidate.sdp}")
			}

			override fun onTrack(transceiver: RtpTransceiver) {
				val track = transceiver.receiver.track()
				Log.i(TAG, "[WHEP] Received track: ${track?.kind()}")

				if (track is VideoTrack) {
					videoTrack = track
					track.setEnabled(true)
					track.addSink { frame ->
						frameCount++
						if (frameCount % 30L == 0L) {
							Log.i(TAG, "[WHEP] Received video frame #$frameCount (${frame.buffer.width}x${frame.buffer.height})")
						}
						val bitmap = videoFrameToBitmap(frame)
						if (bitmap != null) {
							onFrameReceived(bitmap)
						}
					}
				}
			}

			override fun onAddStream(stream: MediaStream) {
				for (track in stream.videoTracks) {
					videoTrack = track
					track.setEnabled(true)
					track.addSink { frame ->
						val bitmap = videoFrameToBitmap(frame)
						if (bitmap != null) {
							onFrameReceived(bitmap)
						}
					}
				}
			}

			override fun onRemoveStream(stream: MediaStream) {}
			override fun onDataChannel(dataChannel: DataChannel) {}
			override fun onRenegotiationNeeded() {}
			override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}
			override fun onIceConnectionReceivingChange(receiving: Boolean) {}
			override fun onAddTrack(receiver: RtpReceiver, mediaStreams: Array<out MediaStream>) {}
		}
	}

	private fun scheduleOfferDispatch(delayMs: Long) {
		executor.schedule({
			sendOfferToWhepServer()
		}, delayMs, TimeUnit.MILLISECONDS)
	}

	private fun sendOfferToWhepServer() {
		if (isStopped.get()) return
		if (!isOfferSent.compareAndSet(false, true)) return

		val url = targetUrl ?: run {
			Log.e(TAG, "[WHEP] Target WHEP URL is null")
			return
		}

		val sdp = peerConnection?.localDescription?.description ?: run {
			Log.e(TAG, "[WHEP] No local SDP available to send")
			return
		}

		Log.i(TAG, "[WHEP] Sending HTTP POST SDP offer to $url")

		val requestBody = sdp.toRequestBody("application/sdp".toMediaType())
		val request = Request.Builder()
			.url(url)
			.post(requestBody)
			.header("Content-Type", "application/sdp")
			.header("Accept", "application/sdp")
			.build()

		httpClient.newCall(request).enqueue(object : Callback {
			override fun onFailure(call: Call, e: IOException) {
				Log.e(TAG, "[WHEP] HTTP POST request to MediaMTX failed", e)
				isOfferSent.set(false)
				scheduleReconnect()
			}

			override fun onResponse(call: Call, response: Response) {
				response.use { res ->
					val answerSdp = res.body.string()
					Log.i(TAG, "[WHEP] Received HTTP ${res.code} response from MediaMTX")

					if (!res.isSuccessful) {
						Log.e(TAG, "[WHEP] MediaMTX returned error HTTP ${res.code}: $answerSdp")
						isOfferSent.set(false)
						scheduleReconnect()
						return
					}

					Log.d(TAG, "[WHEP] Setting remote SDP answer from MediaMTX")
					val remoteDesc = SessionDescription(SessionDescription.Type.ANSWER, answerSdp)
					executor.execute {
						peerConnection?.setRemoteDescription(
							object : SdpObserverAdapter("setRemoteDescription") {
								override fun onSetSuccess() {
									Log.i(TAG, "[WHEP] Remote SDP answer successfully applied")
								}

								override fun onSetFailure(error: String?) {
									Log.e(TAG, "[WHEP] Failed to set remote SDP answer: $error")
									isOfferSent.set(false)
									scheduleReconnect()
								}
							},
							remoteDesc
						)
					}
				}
			}
		})
	}

	private fun scheduleReconnect() {
		if (isStopped.get()) return
		Log.i(TAG, "[WHEP] Scheduling reconnect in ${RECONNECT_DELAY_MS}ms")
		executor.schedule({
			if (!isStopped.get()) {
				connectWebRtc()
			}
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
		try {
			videoTrack?.setEnabled(false)
			videoTrack = null
			peerConnection?.close()
			peerConnection?.dispose()
			peerConnection = null
		} catch (e: Exception) {
			Log.w(TAG, "[WHEP] Error closing PeerConnection: ${e.message}")
		}
	}

	fun stop() {
		Log.i(TAG, "[WHEP] Stopping WebRTC client")
		isStopped.set(true)

		executor.execute {
			try {
				closeCurrentPeerConnection()

				peerConnectionFactory?.dispose()
				peerConnectionFactory = null

				eglBase?.release()
				eglBase = null
			} catch (e: Exception) {
				Log.e(TAG, "[WHEP] Error disposing WebRTC resources", e)
			} finally {
				executor.shutdown()
			}
		}
	}

	private open class SdpObserverAdapter(private val tag: String) : SdpObserver {
		override fun onCreateSuccess(description: SessionDescription) {}
		override fun onSetSuccess() {}
		override fun onCreateFailure(error: String?) {
			Log.e(TAG, "[WHEP] $tag onCreateFailure: $error")
		}
		override fun onSetFailure(error: String?) {
			Log.e(TAG, "[WHEP] $tag onSetFailure: $error")
		}
	}
}
