package com.algorithmic_alliance.eyeaiapp.context

import com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp
import com.algorithmic_alliance.eyeaiapp.worldstate.TimeOrigin
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.random.Random

class ContextEfficiencyTest {
    private fun observation(id: Int, value: Float = .2f) = ContextObservation(
        FrameStamp(1, id.toLong(), id * 200_000_000L, id * 200_000_000L, 640, 480, 0, TimeOrigin.LOCAL_RECEIPT),
        id * 200_000_000L, objects = mapOf((if (value < .4f) "chair" else "person") to 1f),
        visual = List(800) { value }, depth = List(32) { value }, semantics = List(19) { value },
        featureVersion = "context-features-2", quality = listOf("objects", "visual", "depth", "semantics").associateWith { 1.0 },
    )

    @Test fun `vector distances retain finite filtering and image slice semantics`() {
        val random = Random(42)
        fun average(a: List<Float?>, b: List<Float?>): Double? = a.indices.mapNotNull { i ->
            val x = a[i]
            val y = b[i]
            if (x != null && y != null && x.isFinite() && y.isFinite()) abs(x-y).toDouble() else null
        }.takeIf { it.size >= 4 }?.average()?.coerceIn(0.0, 1.0)
        repeat(100) { index ->
            val a = observation(1).copy(visual = List(800) { random.nextFloat() },
                depth = List(32) { if (it % 3 == 0) null else random.nextFloat() })
            val b = observation(2).copy(visual = List(800) { if (it % 7 == 0) Float.NaN else random.nextFloat() },
                depth = List(32) { if (it < index % 33) Float.POSITIVE_INFINITY else random.nextFloat() })
            val result = ContextEngine.distance(a, b)
            assertEquals(maxOf(average(a.visual!!.take(768), b.visual!!.take(768))!! * 3,
                average(a.visual.drop(768), b.visual.drop(768))!! * 2).coerceAtMost(1.0), result.visual)
            assertEquals(average(a.depth!!, b.depth!!), result.depth)
        }
        assertNull(ContextEngine.distance(observation(1).copy(depth = listOf(1f, null, 2f)),
            observation(2).copy(depth = listOf(2f, 3f, 4f))).depth)
        assertNull(ContextEngine.distance(observation(1).copy(depth = listOf(1f)), observation(2)).depth)
    }

    @Test fun `cached decisions match replay after window shifts reset restore and speech`() {
        for (generation in 1..2) for (settle in listOf(0L, 60_000_000_000L)) {
            val config = ContextPolicyConfig(generation = generation, settleNanos = settle, visualSettleNanos = settle)
            val cached = ContextEngine(config)
            cached.restore(ContextEngineState(observation(0), observation(0), null, 0, emptyList(), false))
            repeat(130) { i ->
                if (i == 45) cached.speechStarted(observation(i, .1f))
                if (i == 80) cached.restore(cached.snapshot())
                if (i == 100) cached.reset()
                val replay = ContextEngine(config).also { it.restore(cached.snapshot()) }
                val next = observation(i + 1, if (i % 11 == 0) .55f else .6f)
                assertEquals(replay.observe(next), cached.observe(next))
                assertEquals(replay.reason, cached.reason)
                assertEquals(replay.scores, cached.scores)
                assertEquals(replay.snapshot(), cached.snapshot())
            }
        }
    }

    @Test fun `unchanged candidates are not rescanned while reference stays fixed`() {
        var reads = 0
        val pixels = object : AbstractList<Float>() {
            override val size = 800
            override fun get(index: Int): Float { reads++
            return .2f }
        }
        val config = ContextPolicyConfig(settleNanos = 60_000_000_000L, visualSettleNanos = 60_000_000_000L)
        val engine = ContextEngine(config)
        repeat(6) { engine.observe(observation(it + 1).copy(visual = pixels)) }
        reads = 0
        engine.observe(observation(7).copy(visual = pixels))
        // One new candidate and the current-observation check: two pairs of 800 samples.
        assertEquals(3200, reads)
    }
}
