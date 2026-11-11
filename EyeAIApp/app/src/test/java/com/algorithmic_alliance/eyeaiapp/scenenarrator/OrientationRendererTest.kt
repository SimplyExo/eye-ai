package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Edge
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Item
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Placement
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Plan
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.RangeOrder
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Side
import com.algorithmic_alliance.eyeaiapp.worldstate.Availability
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyDepth
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyLimits
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyLoss
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyObject
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyPolicy
import com.algorithmic_alliance.eyeaiapp.worldstate.TinySemantics
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyWorldInput
import com.algorithmic_alliance.eyeaiapp.worldstate.TrackStatus
import com.algorithmic_alliance.eyeaiapp.worldstate.WorldTask
import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OrientationRendererTest {
    private val language = listOf("src/main", "app/src/main", "EyeAIApp/app/src/main")
        .map { File(it, "assets/scene-narrator/language.json") }.first { it.isFile }
        .let { Json.parseToJsonElement(it.readText()).jsonObject }
    private val renderer = OrientationRenderer(language)

    @Test
    fun tableAndBookAreIntroducedTogetherAndChairUsesRelativePlacement() {
        val plan = plan(
            items = listOf(item(0, "Esstisch"), item(1, "Buch"), item(2, "Stuhl", direction = 2)),
            edges = listOf(Edge(1, 0, "SUPPORTED_BY", 0), Edge(2, 0, "BESIDE", 1)),
            placements = mapOf(2 to Placement(Side.RIGHT, null, beside = true)),
        )
        val normal = verify(plan, "STANDARD")
        assertEquals("Vor dir ist ein Esstisch mit einem Buch darauf.", normal[0].text)
        assertEquals("Rechts daneben ist ein Stuhl.", normal[1].text)
        assertEquals(.8f, normal[1].horizontalCenter, .001f)
        val short = verify(plan, "SHORT")
        assertEquals("Voraus ein Esstisch mit einem Buch darauf.", short[0].text)
        assertEquals("Rechts daneben ein Stuhl.", short[1].text)
    }

    @Test
    fun samePredicateSharesSubjectAndExtraRelationUsesUnambiguousPronoun() {
        val plan = plan(
            items = listOf(item(0, "Person"), item(1, "Buch"), item(2, "Mobiltelefon"), item(3, "Esstisch")),
            edges = listOf(Edge(0, 1, "HOLDS", 0), Edge(0, 2, "HOLDS", 1), Edge(1, 3, "SUPPORTED_BY", 2)),
        )
        for (detail in details) {
            val segments = verify(plan, detail)
            assertEquals(listOf(0, 1, 2), segments.first().objectIds)
            assertTrue(segments.first().text.contains("ein Buch und ein Mobiltelefon"))
            assertFalse(segments.last().text.startsWith("Es "))
        }
        val chain = plan.copy(
            items = listOf(plan.items[0], plan.items[1], item(2, "Esstisch")),
            edges = listOf(Edge(0, 1, "HOLDS", 0), Edge(1, 2, "SUPPORTED_BY", 1)),
        )
        for (detail in details) {
            assertEquals("Es ist auf einem Esstisch.", verify(chain, detail).last().text)
        }
    }

    @Test
    fun relationOrderKeepsDifferentPredicatesWithClearPronoun() {
        val plan = plan(
            items = listOf(item(0, "Person"), item(1, "Tasse"), item(2, "Buch")),
            edges = listOf(Edge(0, 1, "HOLDS", 0), Edge(0, 2, "CARRIES", 1)),
        )
        for (detail in details) {
            val segments = verify(plan, detail)
            assertEquals(listOf(0, 2), segments.first().objectIds)
            assertEquals("Sie hält eine Tasse.", segments.last().text)
        }
    }

    @Test
    fun cyclicEdgesCannotReintroduceAnObjectAndNearbyItemsComeFirst() {
        val plan = plan(
            items = listOf(item(0, "Person", priority = true), item(1, "Stuhl", priority = true),
                item(2, "Buch"), item(3, "Esstisch")),
            edges = listOf(Edge(0, 2, "HOLDS", 0), Edge(2, 3, "SUPPORTED_BY", 1), Edge(3, 0, "OCCLUDES", 2)),
            ranges = mapOf(0 to 10, 1 to 15),
        )
        for (detail in details) {
            val segments = verify(plan, detail)
            assertEquals(listOf(0), segments[0].objectIds)
            assertEquals(1, segments[1].objectIds.first())
            assertEquals(1f, segments[0].distanceMeters)
            assertTrue(segments.first().text.contains(if (detail == "SHORT") "ein Meter" else "einen Meter"))
        }
    }

    @Test
    fun sharedLocationUsesGroupCenterAndIncompleteGroupStaysCentered() {
        val group = Item(listOf(3, 4), cls("Stuhl"), 4, 3, complete = false, priority = false)
        val plan = plan(listOf(item(0, "Esstisch", direction = 2), item(1, "Buch", direction = 2),
            item(2, "Stuhl", direction = 0), group))
        for (detail in details) {
            val segments = verify(plan, detail)
            assertEquals(listOf(0, 1), segments.first().objectIds)
            assertEquals(.8f, segments.first().horizontalCenter, .001f)
            assertNull(segments.first().distanceMeters)
            assertEquals(.5f, segments.last().horizontalCenter, .001f)
            assertTrue(segments.last().text.contains("vier Stühle"))
        }
    }

    @Test
    fun verifiedBehindPlacementSurvivesWhenObstaclesMustBeIntroducedSeparately() {
        val plan = plan(
            items = listOf(item(0, "Person", priority = true), item(1, "Stuhl", priority = true)),
            edges = listOf(Edge(0, 1, "OCCLUDES", 0)),
            placements = mapOf(1 to Placement(null, RangeOrder.FARTHER, behind = true)),
        )
        for (detail in details) {
            val text = verify(plan, detail).last().text
            assertTrue(text.startsWith("Dahinter "))
            assertTrue(text.endsWith("teilweise verdeckt."))
        }
    }

    @Test
    fun distanceBelongsToTheObjectIntroducedThroughTheRelation() {
        val plan = plan(
            items = listOf(item(0, "Person"), item(1, "Buch")),
            edges = listOf(Edge(0, 1, "HOLDS", 0)),
            ranges = mapOf(1 to 25),
        )
        for (detail in details) {
            val segments = verify(plan, detail)
            assertNull(segments.first().distanceMeters)
            assertEquals(2.5f, segments.last().distanceMeters)
            assertTrue(segments.last().text.startsWith("Sie hält ein Buch, etwa 2,5 Meter"))
        }
    }

    private val details = listOf("SHORT", "STANDARD")

    private fun verify(plan: Plan, detail: String): List<NarrationSegment> {
        val input = input(plan.items)
        val segments = renderer.renderSegments(input, plan, detail)
        assertEquals(segments.joinToString(" ") { it.text }, renderer.render(input, plan, detail))
        assertEquals(plan.items.flatMap { it.ids }.sorted(), segments.flatMap { it.objectIds }.sorted())
        return segments
    }

    private fun cls(noun: String) = language.getValue("nouns").jsonArray.indexOfFirst {
        it.jsonObject.getValue("singular").jsonPrimitive.content == noun
    }.also { check(it >= 0) { noun } }

    private fun item(id: Int, name: String, direction: Int = 1, priority: Boolean = false) =
        Item(listOf(id), cls(name), 1, direction, complete = true, priority = priority)

    private fun plan(
        items: List<Item>,
        edges: List<Edge> = emptyList(),
        placements: Map<Int, Placement> = emptyMap(),
        ranges: Map<Int, Int> = emptyMap(),
    ) = Plan(0, emptyList(), items, placements, edges, ranges, ranges.keys.firstOrNull())

    private fun input(items: List<Item>): TinyWorldInput {
        val objects = items.flatMap { item ->
            val center = listOf(200, 500, 800, 500)[item.direction]
            item.ids.map { id ->
                TinyObject(id, item.cls, listOf(center - 50, 100, center + 50, 200), 9,
                    TrackStatus.CONFIRMED, Availability.AVAILABLE, 0,
                    TinyDepth(Availability.UNKNOWN), Availability.UNKNOWN)
            }
        }.sortedBy { it.id }
        // The renderer consumes the selected items, their boxes, and the planner's verified relationships.
        return TinyWorldInput(0, 0, "", "", WorldTask.DESCRIBE, "de", false,
            TinyPolicy(emptyList(), TinyLimits()), emptyList(), emptyList(), emptyList(), objects,
            emptyList(), TinySemantics(Availability.UNKNOWN), emptyList(), emptyList(), "",
            TinyLoss(0, 0, 0, 0, 0, 0))
    }
}
