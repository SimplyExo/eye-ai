package com.algorithmic_alliance.eyeaiapp

import java.nio.ByteBuffer
import java.nio.ByteOrder

object NativeLib {
    class NativeFloatBuffer(length: Int) {
        val byteBuffer = ByteBuffer.allocateDirect(length * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
    }
}
