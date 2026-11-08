package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import java.security.MessageDigest
import kotlin.math.*

// Shared R4 app projection: current object boxes, confidence, age and box pair geometry.
internal object R4Tensorizer {
    const val PROJECTION_ID = "r4-app-shared-box-projection-1"
    const val NODES = 32
    const val NODE_FEATURES = 57
    const val PAIR_FEATURES = 28

    data class Node(val id: Int, val cls: Int, val source: Int, val confidence: Float, val ageMs: Int,
        val box: FloatArray)
    data class Tensor(val cats: IntArray, val nums: FloatArray, val pairs: FloatArray,
        val valid: ByteArray, val nodes: List<Node>, val snapshot: String, val omitted: Int)

    fun snapshot(input: TinyWorldInput): String = MessageDigest.getInstance("SHA-256")
        .digest(TinyWorldCodec.encode(input).toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it.toInt() and 255) }

    private fun ageMs(input: TinyWorldInput, objectId: Int): Int = when (input.sources[input.objects[objectId].source].age) {
        TinyAge.LE_100_MS -> 100
        TinyAge.LE_500_MS -> 500
        TinyAge.LE_1_S -> 1000
        TinyAge.LE_10_S -> 10000
        TinyAge.OLDER -> 10001
        TinyAge.INVALID -> Int.MAX_VALUE
    }

    fun encode(input: TinyWorldInput): Tensor {
        TinyWorldCodec.validate(input)
        require(input.channels[0].status == Availability.AVAILABLE)
        val all = input.objects.asSequence()
            .filter { it.track == TrackStatus.CONFIRMED && it.freshness == Availability.AVAILABLE }
            .filter { it.source == input.channels[0].source }
            .filter { it.box[0] < it.box[2] && it.box[1] < it.box[3] }
            .sortedWith(compareByDescending<TinyObject> { it.confidenceDecile }.thenBy { it.id.toString() })
            .toList()
        val chosen = all.take(NODES).map { o -> Node(o.id, o.cls + 1, o.source,
            o.confidenceDecile / 10f, ageMs(input, o.id), FloatArray(4) { o.box[it] / 1000f }) }
        val cats = IntArray(NODES)
        val nums = FloatArray(NODES * NODE_FEATURES)
        val pairs = FloatArray(NODES * NODES * PAIR_FEATURES)
        val valid = ByteArray(NODES)
        chosen.forEachIndexed { i, n ->
            require(n.cls in 1..80 && n.box.all { it.isFinite() && it in 0f..1f })
            cats[i] = n.cls
            valid[i] = 1
            val o = input.objects[n.id]
            val offset = i * NODE_FEATURES
            for (j in 0..3) nums[offset + j] = n.box[j]
            nums[offset + 4] = n.confidence
            // R4 projection 1 deliberately has no per-pixel depth or semantic map.
            // The omitted feature groups stay zero exactly as specified by R4.
            nums[offset + 11] = min(n.ageMs / 1000f, 10f)
            nums[offset + 12] = 0f
            check(o.id == n.id)
        }
        for (i in chosen.indices) for (j in chosen.indices) {
            if (i == j) continue
            val a = chosen[i].box
            val b = chosen[j].box
            val aw = a[2] - a[0]
            val ah = a[3] - a[1]
            val bw = b[2] - b[0]
            val bh = b[3] - b[1]
            val overlapW = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
            val overlapH = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
            val intersection = overlapW * overlapH
            val areaA = aw * ah
            val areaB = bw * bh
            val dx = (a[0] + a[2] - b[0] - b[2]) / 2f
            val dy = (a[1] + a[3] - b[1] - b[3]) / 2f
            val offset = (i * NODES + j) * PAIR_FEATURES
            pairs[offset] = dx
            pairs[offset + 1] = dy
            pairs[offset + 2] = ln((aw / bw).toDouble()).toFloat().coerceIn(-5f, 5f)
            pairs[offset + 3] = ln((ah / bh).toDouble()).toFloat().coerceIn(-5f, 5f)
            pairs[offset + 4] = intersection / max(1e-8f, areaA + areaB - intersection)
            pairs[offset + 5] = intersection / max(1e-8f, areaA)
            pairs[offset + 6] = intersection / max(1e-8f, areaB)
            pairs[offset + 7] = max(0f, max(a[0] - b[2], b[0] - a[2]))
            pairs[offset + 8] = max(0f, max(a[1] - b[3], b[1] - a[3]))
            pairs[offset + 9] = 0f
            pairs[offset + 10] = 0f
            pairs[offset + 11] = 0f
            pairs[offset + 12] = 0f
            pairs[offset + 13] = hypot(dx, dy)
            pairs[offset + 14] = if (chosen[i].cls == chosen[j].cls) 1f else 0f
            pairs[offset + 15] = 0f
            // R4 pair features 1..8, with absent image maps represented by zeros.
            pairs[offset + 16] = a[3] - b[1]
            pairs[offset + 17] = a[3] - b[3]
            pairs[offset + 18] = overlapW / aw
            pairs[offset + 19] = overlapW / bw
        }
        require(nums.all(Float::isFinite) && pairs.all(Float::isFinite))
        return Tensor(cats, nums, pairs, valid, chosen, snapshot(input), input.objects.size - chosen.size)
    }
}
