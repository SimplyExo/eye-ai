package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class DeterministicSettingsParsersTest {
	private val magnitude = GermanMagnitudeParser()
	private val units = SettingUnitParser()

	@Test
	fun `magnitude values and null semantics are deterministic`() {
		val cases = listOf(
			Triple("etwas höher", SettingOperation.INCREASE, ChangeMagnitude.SMALL),
			Triple("ein bisschen schneller", SettingOperation.INCREASE, ChangeMagnitude.SMALL),
			Triple("deutlich höher", SettingOperation.INCREASE, ChangeMagnitude.LARGE),
			Triple("stark reduzieren", SettingOperation.DECREASE, ChangeMagnitude.LARGE),
			Triple("erhöhe die frequenz", SettingOperation.INCREASE, ChangeMagnitude.DEFAULT)
		)
		cases.forEach { (text, operation, expected) ->
			val result = magnitude.parse(text, operation, null)
			assertEquals(expected, result.value)
			assertEquals(MagnitudeParseStatus.CLEAR, result.status)
		}
		assertNull(magnitude.parse("erhöhe um hundert", SettingOperation.INCREASE, 100.0).value)
		assertNull(magnitude.parse("setze auf hundert", SettingOperation.SET_ABSOLUTE, 100.0).value)
	}

	@Test
	fun `negation and conflicting modifiers are ambiguous`() {
		listOf("nicht stark erhöhen", "kein bisschen schneller", "etwas deutlich erhöhen").forEach { text ->
			val result = magnitude.parse(text, SettingOperation.INCREASE, null)
			assertNull(result.value)
			assertEquals(MagnitudeParseStatus.AMBIGUOUS, result.status)
		}
	}

	@Test
	fun `target supplies defaults but unsupported explicit unit does not`() {
		assertEquals(SettingUnit.HZ, units.parse(SettingTarget.FREQUENCY, "setze sie auf 700").unit)
		assertEquals(SettingUnit.BPS, units.parse(SettingTarget.BPS, "setze sie auf 4").unit)
		assertNull(units.parse(SettingTarget.SPEAKER, "andere stimme").unit)
		listOf("kiloherz", "kilohertz", "prozent").forEach { word ->
			val result = units.parse(SettingTarget.FREQUENCY, "setze sie auf 700 $word")
			assertFalse(result.isValid)
			assertNull(result.unit)
			assertEquals("UNSUPPORTED_EXPLICIT_UNIT", result.diagnostic)
		}
	}

	@Test
	fun `explicit wrong supported unit stays invalid`() {
		val result = units.parse(SettingTarget.FREQUENCY, "setze sie auf 700 bps")
		assertFalse(result.isValid)
		assertEquals(SettingUnit.BPS, result.unit)
		assertEquals("INVALID_UNIT_FOR_FREQUENCY", result.diagnostic)
	}
}
