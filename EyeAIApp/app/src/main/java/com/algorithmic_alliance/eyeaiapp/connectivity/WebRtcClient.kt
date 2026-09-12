package com.algorithmic_alliance.eyeaiapp.connectivity

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.webrtc.*
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class WebRtcClient(
    private val context: Context,
    private val onFrameReceived: (Bitmap) -> Unit
) {

    companion object {
        private const val TAG = "WebRtcClient"
    }

    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor()

    val httpClient = OkHttpClient.Builder()
        .hostnameVerifier { hostname, session ->
            hostname == "192.168.4.1"
        }
        .build()

    val eglBase: EglBase = EglBase.create()

    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var videoTrack: VideoTrack? = null

    private var pendingUrl: String? = null
    private var frameCount = 0

    // ------------------------------------------------------------
    // START
    // ------------------------------------------------------------

    fun start(url: String) {
        Log.i(TAG, "Starting WHEP connection")
        Log.i(TAG, "URL: $url")

        pendingUrl = url

        executor.execute {
            try {
                initializeFactory()
                createPeerConnection()

                val pc = peerConnection
                    ?: throw IllegalStateException("PeerConnection is null")

                addReceiveTransceiver(pc)

                createOffer(pc)

            } catch (e: Exception) {
                Log.e(TAG, "Failed to start WebRTC", e)
            }
        }
    }

    // ------------------------------------------------------------
    // FACTORY
    // ------------------------------------------------------------

    private fun initializeFactory() {
        if (peerConnectionFactory != null) {
            return
        }

        Log.i(TAG, "Initializing PeerConnectionFactory")

        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions
                .builder(context.applicationContext)
                .createInitializationOptions()
        )

        val encoderFactory =
            DefaultVideoEncoderFactory(
                eglBase.eglBaseContext,
                true,
                true
            )

        val decoderFactory =
            DefaultVideoDecoderFactory(
                eglBase.eglBaseContext
            )

        peerConnectionFactory =
            PeerConnectionFactory.builder()
                .setVideoEncoderFactory(encoderFactory)
                .setVideoDecoderFactory(decoderFactory)
                .createPeerConnectionFactory()

        Log.i(TAG, "PeerConnectionFactory initialized")
    }

    // ------------------------------------------------------------
    // PEER CONNECTION
    // ------------------------------------------------------------

    private fun createPeerConnection() {
        if (peerConnection != null) {
            return
        }

        val config =
            PeerConnection.RTCConfiguration(
                emptyList()
            ).apply {

                sdpSemantics =
                    PeerConnection.SdpSemantics.UNIFIED_PLAN

                continualGatheringPolicy =
                    PeerConnection.ContinualGatheringPolicy.GATHER_ONCE

                // MediaMTX normally works without a public STUN
                // server when the client can reach the server directly.
                //
                // If MediaMTX is behind NAT, configure TURN/STUN
                // according to your deployment.
            }

        peerConnection =
            peerConnectionFactory?.createPeerConnection(
                config,
                object : PeerConnection.Observer {

                    override fun onSignalingChange(
                        state: PeerConnection.SignalingState
                    ) {
                        Log.i(TAG, "Signaling state: $state")
                    }

                    override fun onIceConnectionChange(
                        state: PeerConnection.IceConnectionState
                    ) {
                        Log.i(TAG, "ICE connection state: $state")
                    }

                    override fun onIceGatheringChange(
                        state: PeerConnection.IceGatheringState
                    ) {
                        Log.i(TAG, "ICE gathering state: $state")

                        if (
                            state ==
                            PeerConnection.IceGatheringState.COMPLETE
                        ) {
                            sendOffer()
                        }
                    }

                    override fun onConnectionChange(
                        state: PeerConnection.PeerConnectionState
                    ) {
                        Log.i(TAG, "Peer connection state: $state")
                    }

                    override fun onIceCandidate(
                        candidate: IceCandidate
                    ) {
                        Log.d(
                            TAG,
                            "ICE candidate: ${candidate.sdp}"
                        )

                        // We intentionally wait for ICE gathering
                        // to complete and send the complete SDP
                        // to MediaMTX.
                    }

                    override fun onTrack(
                        transceiver: RtpTransceiver
                    ) {
                        Log.i(TAG, "========== onTrack ==========")

                        val receiver = transceiver.receiver
                        val track = receiver.track()

                        Log.i(TAG, "Track: $track")
                        Log.i(TAG, "Kind: ${track?.kind()}")

                        if (track is VideoTrack) {
                            Log.i(TAG, "VIDEO TRACK RECEIVED")

                            videoTrack = track

                            track.setEnabled(true)

                            track.addSink { frame ->

                                frameCount++

                                if (frameCount % 30 == 0) {
                                    Log.i(
                                        TAG,
                                        "Frames received: $frameCount " +
                                                "${frame.buffer.width}x${frame.buffer.height}"
                                    )
                                }

                                val bitmap =
                                    videoFrameToBitmap(frame)

                                if (bitmap != null) {
                                    onFrameReceived(bitmap)
                                }
                            }
                        }
                    }

                    override fun onAddStream(
                        stream: MediaStream
                    ) {
                        Log.i(
                            TAG,
                            "onAddStream: ${stream.id}"
                        )

                        for (track in stream.videoTracks) {
                            Log.i(
                                TAG,
                                "Video track from stream: $track"
                            )

                            videoTrack = track
                            track.setEnabled(true)

                            track.addSink { frame ->
                                val bitmap =
                                    videoFrameToBitmap(frame)

                                if (bitmap != null) {
                                    onFrameReceived(bitmap)
                                }
                            }
                        }
                    }

                    override fun onRemoveStream(
                        stream: MediaStream
                    ) {
                        Log.i(
                            TAG,
                            "onRemoveStream: ${stream.id}"
                        )
                    }

                    override fun onDataChannel(
                        dataChannel: DataChannel
                    ) {
                        Log.d(TAG, "Data channel received")
                    }

                    override fun onRenegotiationNeeded() {
                        Log.d(
                            TAG,
                            "Renegotiation needed"
                        )
                    }

                    override fun onIceCandidatesRemoved(
                        candidates: Array<out IceCandidate>
                    ) {
                    }

                    override fun onIceConnectionReceivingChange(
                        receiving: Boolean
                    ) {
                        Log.i(
                            TAG,
                            "ICE receiving: $receiving"
                        )
                    }

                    override fun onAddTrack(
                        receiver: RtpReceiver,
                        mediaStreams: Array<out MediaStream>
                    ) {
                        Log.i(
                            TAG,
                            "onAddTrack: ${receiver.track()?.kind()}"
                        )
                    }
                }
            )

        if (peerConnection == null) {
            throw IllegalStateException(
                "Failed to create PeerConnection"
            )
        }

        Log.i(TAG, "PeerConnection created")
    }

    // ------------------------------------------------------------
    // TRANSCEIVER
    // ------------------------------------------------------------

    private fun addReceiveTransceiver(
        pc: PeerConnection
    ) {
        Log.i(TAG, "Adding RECV_ONLY video transceiver")

        pc.addTransceiver(
            MediaStreamTrack.MediaType.MEDIA_TYPE_VIDEO,
            RtpTransceiver.RtpTransceiverInit(
                RtpTransceiver.RtpTransceiverDirection.RECV_ONLY
            )
        )
    }

    // ------------------------------------------------------------
    // OFFER
    // ------------------------------------------------------------

    private fun createOffer(
        pc: PeerConnection
    ) {
        Log.i(TAG, "Creating SDP offer")

        val constraints =
            MediaConstraints().apply {

                mandatory.add(
                    MediaConstraints.KeyValuePair(
                        "OfferToReceiveVideo",
                        "true"
                    )
                )
            }

        pc.createOffer(
            object : SdpObserver {

                override fun onCreateSuccess(
                    description: SessionDescription
                ) {
                    Log.i(
                        TAG,
                        "Offer created"
                    )

                    Log.d(
                        TAG,
                        "========== LOCAL SDP ==========\n" +
                                description.description +
                                "\n==============================="
                    )

                    pc.setLocalDescription(
                        object : SdpObserver {

                            override fun onSetSuccess() {
                                Log.i(
                                    TAG,
                                    "Local SDP set"
                                )
                            }

                            override fun onSetFailure(
                                error: String?
                            ) {
                                Log.e(
                                    TAG,
                                    "setLocalDescription failed: $error"
                                )
                            }

                            override fun onCreateSuccess(
                                description: SessionDescription?
                            ) {
                            }

                            override fun onCreateFailure(
                                error: String?
                            ) {
                            }
                        },
                        description
                    )
                }

                override fun onCreateFailure(
                    error: String?
                ) {
                    Log.e(
                        TAG,
                        "createOffer failed: $error"
                    )
                }

                override fun onSetSuccess() {
                }

                override fun onSetFailure(
                    error: String?
                ) {
                    Log.e(
                        TAG,
                        "Offer set failed: $error"
                    )
                }
            },
            constraints
        )
    }

    // ------------------------------------------------------------
    // WHEP
    // ------------------------------------------------------------

    private fun sendOffer() {

        val url = pendingUrl
            ?: run {
                Log.e(TAG, "No WHEP URL")
                return
            }

        val sdp =
            peerConnection
                ?.localDescription
                ?.description
                ?: run {
                    Log.e(
                        TAG,
                        "No local SDP available"
                    )
                    return
                }

        Log.i(TAG, "ICE gathering complete")
        Log.i(TAG, "Sending WHEP offer to MediaMTX")

        val body =
            sdp.toRequestBody(
                "application/sdp".toMediaType()
            )

        val request =
            Request.Builder()
                .url(url)
                .post(body)
                .header(
                    "Content-Type",
                    "application/sdp"
                )
                .header(
                    "Accept",
                    "application/sdp"
                )
                .build()

        httpClient
            .newCall(request)
            .enqueue(
                object : Callback {

                    override fun onFailure(
                        call: Call,
                        e: IOException
                    ) {
                        Log.e(
                            TAG,
                            "WHEP request failed",
                            e
                        )
                    }

                    override fun onResponse(
                        call: Call,
                        response: Response
                    ) {
                        response.use {

                            val answer =
                                it.body?.string()
                                    ?: ""

                            Log.i(
                                TAG,
                                "WHEP HTTP ${it.code}"
                            )

                            if (!it.isSuccessful) {
                                Log.e(
                                    TAG,
                                    "WHEP error: $answer"
                                )
                                return
                            }

                            Log.i(
                                TAG,
                                "========== REMOTE SDP ==========\n" +
                                        answer +
                                        "\n================================="
                            )

                            val pc =
                                peerConnection
                                    ?: return

                            val remoteDescription =
                                SessionDescription(
                                    SessionDescription.Type.ANSWER,
                                    answer
                                )

                            pc.setRemoteDescription(
                                object : SdpObserver {

                                    override fun onSetSuccess() {
                                        Log.i(
                                            TAG,
                                            "Remote SDP successfully set"
                                        )
                                    }

                                    override fun onSetFailure(
                                        error: String?
                                    ) {
                                        Log.e(
                                            TAG,
                                            "setRemoteDescription failed: $error"
                                        )
                                    }

                                    override fun onCreateSuccess(
                                        description: SessionDescription?
                                    ) {
                                    }

                                    override fun onCreateFailure(
                                        error: String?
                                    ) {
                                    }
                                },
                                remoteDescription
                            )
                        }
                    }
                }
            )
    }

    // ------------------------------------------------------------
    // VIDEO -> BITMAP
    // ------------------------------------------------------------

    private fun videoFrameToBitmap(
        frame: VideoFrame
    ): Bitmap? {

        val i420 = frame.buffer.toI420()
            ?: run {
                Log.e(TAG, "toI420() returned null")
                return null
            }

        try {
            val width = i420.width
            val height = i420.height

            val yuv = ByteArray(width * height * 3 / 2)

            val y = i420.dataY
            val u = i420.dataU
            val v = i420.dataV

            val strideY = i420.strideY
            val strideU = i420.strideU
            val strideV = i420.strideV

            // Y plane
            for (row in 0 until height) {
                y.position(row * strideY)
                y.get(
                    yuv,
                    row * width,
                    width
                )
            }

            // NV21 = Y + VU
            val uvOffset = width * height

            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {

                    yuv[
                        uvOffset +
                                row * width +
                                col * 2
                    ] = v.get(
                        row * strideV + col
                    )

                    yuv[
                        uvOffset +
                                row * width +
                                col * 2 + 1
                    ] = u.get(
                        row * strideU + col
                    )
                }
            }

            val jpeg = ByteArrayOutputStream()

            val success = YuvImage(
                yuv,
                ImageFormat.NV21,
                width,
                height,
                null
            ).compressToJpeg(
                Rect(
                    0,
                    0,
                    width,
                    height
                ),
                90,
                jpeg
            )

            if (!success) {
                Log.e(TAG, "YuvImage.compressToJpeg() failed")
                return null
            }

            val bitmap = BitmapFactory.decodeByteArray(
                jpeg.toByteArray(),
                0,
                jpeg.size()
            ) ?: run {
                Log.e(TAG, "BitmapFactory.decodeByteArray() failed")
                return null
            }

            if (frame.rotation == 0) {
                return bitmap
            }

            val matrix = Matrix().apply {
                postRotate(frame.rotation.toFloat())
            }

            return Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                matrix,
                true
            )

        } catch (e: Exception) {
            Log.e(TAG, "Video frame conversion failed", e)
            return null
        } finally {
            i420.release()
        }
    }

    // ------------------------------------------------------------
    // STOP
    // ------------------------------------------------------------

    fun stop() {

        Log.i(TAG, "Stopping WebRTC")

        executor.execute {

            try {
                videoTrack?.let {
                    it.setEnabled(false)
                }

                videoTrack = null

                peerConnection?.close()
                peerConnection?.dispose()
                peerConnection = null

                peerConnectionFactory?.dispose()
                peerConnectionFactory = null

            } catch (e: Exception) {
                Log.e(
                    TAG,
                    "Error stopping WebRTC",
                    e
                )
            } finally {

                if (!executor.isShutdown) {
                    executor.shutdown()
                }

                try {
                    eglBase.release()
                } catch (_: Exception) {
                }
            }
        }
    }
}