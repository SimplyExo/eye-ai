use std::ffi::CString;
use std::sync::LazyLock;
use tracing::{
	Event, Level, Subscriber,
	field::{Field, Visit},
};
use tracing_subscriber::{Layer, layer::Context};

#[link(name = "log")]
unsafe extern "C" {
	fn __android_log_print(prio: i32, tag: *const i8, msg: *const i8) -> i32;
}

static LOG_TAG: LazyLock<CString> = LazyLock::new(|| CString::new("eye-ai-core-rs").unwrap());

const ANDROID_LOG_VERBOSE: i32 = 2;
const ANDROID_LOG_DEBUG: i32 = 3;
const ANDROID_LOG_INFO: i32 = 4;
const ANDROID_LOG_WARN: i32 = 5;
const ANDROID_LOG_ERROR: i32 = 6;

struct MessageVisitor {
	message: Option<String>,
	fields: Vec<(String, String)>,
}
impl Visit for MessageVisitor {
	fn record_str(&mut self, field: &Field, value: &str) {
		if field.name() == "message" {
			self.message = Some(value.to_string());
		} else {
			self.fields
				.push((field.name().to_string(), value.to_string()));
		}
	}

	fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
		if field.name() == "message" {
			self.message = Some(format!("{:?}", value));
		} else {
			self.fields
				.push((field.name().to_string(), format!("{:?}", value)));
		}
	}
}

pub struct AndroidLogLayer;

impl<S> Layer<S> for AndroidLogLayer
where
	S: Subscriber,
{
	fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
		let metadata = event.metadata();
		let mut visitor = MessageVisitor {
			message: None,
			fields: Vec::new(),
		};
		event.record(&mut visitor);

		let module_path = metadata.module_path().unwrap_or_default();

		let msg = visitor.message.unwrap_or_default();

		let fields_str = visitor
			.fields
			.iter()
			.map(|(k, v)| format!("{}={}", k, v))
			.collect::<Vec<_>>()
			.join(" ");

		let formatted_msg = if fields_str.is_empty() {
			format!("{module_path}: {msg}")
		} else {
			format!("{module_path}: {msg} {{{fields_str}}}")
		};

		let prio = match *metadata.level() {
			Level::TRACE => ANDROID_LOG_VERBOSE,
			Level::DEBUG => ANDROID_LOG_DEBUG,
			Level::INFO => ANDROID_LOG_INFO,
			Level::WARN => ANDROID_LOG_WARN,
			Level::ERROR => ANDROID_LOG_ERROR,
		};

		let formatted_msg_cstr = CString::new(formatted_msg).unwrap();

		unsafe {
			__android_log_print(
				prio,
				LOG_TAG.as_ptr() as *const i8,
				formatted_msg_cstr.as_ptr() as *const i8,
			);
		}
	}
}
