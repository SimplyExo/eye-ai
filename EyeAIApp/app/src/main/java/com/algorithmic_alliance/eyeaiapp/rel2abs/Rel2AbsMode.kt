package com.algorithmic_alliance.eyeaiapp.rel2abs

/** The explicit user-selected REL2ABS calibration path. */
enum class Rel2AbsMode(
	val preferenceValue: String,
	val neuralGateId: String? = null,
	val cameraHeightPriorM: Float? = null,
	val cameraHeightTenths: Int? = null,
) {
	Z1("Z1"),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_160(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H1.60",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_160",
		cameraHeightPriorM = 1.60f,
		cameraHeightTenths = 16,
	),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_170(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H1.70",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_170",
		cameraHeightPriorM = 1.70f,
		cameraHeightTenths = 17,
	),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_180(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H1.80",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_180",
		cameraHeightPriorM = 1.80f,
		cameraHeightTenths = 18,
	),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_190(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H1.90",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_190",
		cameraHeightPriorM = 1.90f,
		cameraHeightTenths = 19,
	),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_200(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H2.00",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_200",
		cameraHeightPriorM = 2.00f,
		cameraHeightTenths = 20,
	),

	;

	val isNeuralGate: Boolean
		get() = neuralGateId != null

	companion object {
		val DEFAULT: Rel2AbsMode = OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_170
		const val MIN_CAMERA_HEIGHT_TENTHS = 16
		const val MAX_CAMERA_HEIGHT_TENTHS = 20
		const val DEFAULT_CAMERA_HEIGHT_TENTHS = 17
		val OBJECT_GATE_MODES: List<Rel2AbsMode> = listOf(
			OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_160,
			OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_170,
			OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_180,
			OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_190,
			OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_200,
		)

		fun forCameraHeightTenths(value: Int): Rel2AbsMode =
			OBJECT_GATE_MODES.firstOrNull { it.cameraHeightTenths == value } ?: DEFAULT
	}
}
