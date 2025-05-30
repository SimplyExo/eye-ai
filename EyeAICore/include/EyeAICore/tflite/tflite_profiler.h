#pragma once

// NOLINTBEGIN

/*
 * This part is copied from:
 * https://github.com/google-ai-edge/LiteRT/blob/v1.2.0/tflite/profiling/telemetry/c/telemetry_setting.h
 */

#include <stddef.h>
#include <stdint.h>

// TFLite model, interpreter or delegate settings that will be reported by
// telemetry.
// Note: This struct does not comply with ABI stability.
typedef struct TfLiteTelemetrySettings {
	// Source of the settings. Determines how `data` is interpreted.
	// See tflite::telemetry::TelemetrySource for definition.
	uint32_t source;

	// Settings data. Interpretation based on `source`.
	// If `source` is TFLITE_INTERPRETER, the type of `data` will
	// be `TelemetryInterpreterSettings`.
	// Otherwise, the data is provided by the individual delegate.
	// Owned by the caller that exports TelemetrySettings (e.g. Interpreter).
	const void* data;
} TfLiteTelemetrySettings;

/*
 * This part is copied from:
 * https://github.com/google-ai-edge/LiteRT/blob/v1.2.0/tflite/profiling/telemetry/c/profiler.h
 */

// C API for TFLite telemetry profiler.
// See C++ interface in tflite::telemetry::TelemetryProfiler.
// Note: This struct does not comply with ABI stability.
typedef struct TfLiteTelemetryProfilerStruct {
	// Data that profiler needs to identify itself. This data is owned by the
	// profiler. The profiler is owned in the user code, so the profiler is
	// responsible for deallocating this when it is destroyed.
	void* data;

	// Reports a telemetry event with status.
	// `event_name` indicates the name of the event (e.g. "Invoke") and should
	// not be nullptr. `status`: uint64_t representation of TelemetryStatusCode.
	void (*ReportTelemetryEvent)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		const char* event_name,
		uint64_t status
	);

	// Reports an op telemetry event with status.
	// Same as `ReportTelemetryEvent`, with additional args `op_idx` and
	// `subgraph_idx`.
	// `status`: uint64_t representation of TelemetryStatusCode.
	void (*ReportTelemetryOpEvent)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		const char* event_name,
		int64_t op_idx,
		int64_t subgraph_idx,
		uint64_t status
	);

	// Reports the model and interpreter settings.
	// `setting_name` indicates the name of the setting and should not be
	// nullptr. `settings`'s lifespan is not guaranteed outside the scope of
	// `ReportSettings` call.
	void (*ReportSettings)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		const char* setting_name,
		const TfLiteTelemetrySettings* settings
	);

	// Signals the beginning of an operator invocation.
	// `op_name` is the name of the operator and should not be nullptr.
	// Op invoke event are triggered with OPERATOR_INVOKE_EVENT type for TfLite
	// ops and delegate kernels, and DELEGATE_OPERATOR_INVOKE_EVENT for delegate
	// ops within a delegate kernels, if the instrumentation is in place.
	// Returns event handle which can be passed to `EndOpInvokeEvent` later.
	uint32_t (*ReportBeginOpInvokeEvent)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		const char* op_name,
		int64_t op_idx,
		int64_t subgraph_idx
	);

	// Signals the end to the event specified by `event_handle`.
	void (*ReportEndOpInvokeEvent)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		uint32_t event_handle
	);

	// For op / delegate op with built-in performance measurements, they
	// are able to report the elapsed time directly.
	// `elapsed_time` is in microsecond.
	void (*ReportOpInvokeEvent)( // NOLINT
		struct TfLiteTelemetryProfilerStruct* profiler,
		const char* op_name,
		uint64_t elapsed_time,
		int64_t op_idx,
		int64_t subgraph_idx
	);
} TfLiteTelemetryProfilerStruct;

// NOLINTEND