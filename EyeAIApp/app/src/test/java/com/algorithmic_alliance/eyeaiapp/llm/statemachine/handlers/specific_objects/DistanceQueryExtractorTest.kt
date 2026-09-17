package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class DistanceQueryExtractorTest {
	@Test
	fun extractsObjectFromNaturalLanguageDistanceQuestion() {
		assertEquals(
			"Auto",
			DistanceQueryExtractor.extract("Wie weit ist das Auto entfernt"),
		)
	}

	@Test
	fun prefersLongestKnownLabel() {
		assertEquals(
			"Hot Dog",
			DistanceQueryExtractor.extract("Wie weit ist der Hot Dog entfernt"),
		)
	}

	@Test
	fun returnsNullWhenNoDetectorLabelIsMentioned() {
		assertNull(DistanceQueryExtractor.extract("Wie weit ist es entfernt"))
	}
}
