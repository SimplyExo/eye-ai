package com.algorithmic_alliance.eyeaiapp.rel2abs

/** The explicit user-selected REL2ABS calibration path. */
enum class Rel2AbsMode(
	val preferenceValue: String,
	val neuralGateId: String? = null,
	val cameraHeightPriorM: Float? = null,
) {
	Z1("Z1"),
	S2("S2"),
	NEURAL_GATE_BASE_COCO_WAYMO(
		"V6 Gate COCO+Waymo",
		neuralGateId = "E_NeuralGate_Base_COCO_WAYMO",
	),
	NEURAL_GATE_CONTEXT_V3_COCO(
		"V6 Gate V3+COCO Context",
		neuralGateId = "E_NeuralGate_Context_V3_COCO",
	),
	NEURAL_GATE_CONTEXT_V3_WAYMO(
		"V6 Gate V3+Waymo Context",
		neuralGateId = "E_NeuralGate_Context_V3_WAYMO",
	),
	OBJECT_GATE_V3_WAYMO_FULL_CONTEXT_HEIGHT_170(
		"V6 Object Gate V3+Waymo Full Object Context + Camera H1.70",
		neuralGateId = "E_ObjectGate_DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION_V3_WAYMO_CAMERA_HEIGHT_170",
		cameraHeightPriorM = 1.70f,
	),

	;

	val isNeuralGate: Boolean
		get() = neuralGateId != null

	companion object {
		fun fromPreference(value: String?): Rel2AbsMode =
			entries.firstOrNull { it.preferenceValue == value } ?: Z1
	}
}
