package com.algorithmic_alliance.eyeaiapp.ui

import android.Manifest
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class VisionPermissionsTest {
	@Test fun `Android 12 only needs location for WiFi discovery`() {
		assertArrayEquals(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), visionPermissionsForApi(32))
	}

	@Test fun `Android 13 through 16 also need nearby WiFi`() {
		for (api in 33..36) {
			assertArrayEquals(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION,
				Manifest.permission.NEARBY_WIFI_DEVICES), visionPermissionsForApi(api))
		}
	}

	@Test fun `Android 17 and newer require LAN access in addition to joining WiFi`() {
		for (api in 37..38) {
			assertArrayEquals(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION,
				Manifest.permission.NEARBY_WIFI_DEVICES, Manifest.permission.ACCESS_LOCAL_NETWORK),
				visionPermissionsForApi(api))
		}
	}
}
