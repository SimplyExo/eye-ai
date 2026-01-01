use crossbeam::queue::SegQueue;
use std::{
	sync::{
		RwLock,
		atomic::{AtomicUsize, Ordering},
	},
	time::Instant,
};
#[cfg(feature = "enable_tracy_profiling")]
use tracing_tracy::client::FrameName;

#[derive(Debug, Clone)]
pub struct ProfileScopeRecord {
	pub name: String,
	pub scope_depth: usize,
	pub start: Instant,
	pub duration: dur::Duration,
}
impl std::fmt::Display for ProfileScopeRecord {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let padding = (0..(self.scope_depth * 4)).map(|_| ' ').collect::<String>();
		write!(f, "{}{}: {:.2}", padding, self.name, self.duration)
	}
}

pub struct ProfileScope<'a> {
	pub name: String,
	pub scope_depth: usize,
	pub start: Instant,
	profiling_frame: &'a ProfilingFrame,
}

impl<'a> ProfileScope<'a> {
	pub fn new(name: String, scope_depth: usize, profiling_frame: &'a ProfilingFrame) -> Self {
		Self {
			name,
			scope_depth,
			start: Instant::now(),
			profiling_frame,
		}
	}
}
impl<'a> Drop for ProfileScope<'a> {
	fn drop(&mut self) {
		self.profiling_frame
			._internal_submit_scope(ProfileScopeRecord {
				name: std::mem::take(&mut self.name),
				scope_depth: self.scope_depth,
				start: self.start,
				duration: dur::Duration::from_std(Instant::now() - self.start),
			});
	}
}

pub struct ProfilingFrame {
	name: String,
	#[cfg(feature = "enable_tracy_profiling")]
	frame_name: FrameName,
	start: RwLock<Instant>,
	profile_scopes: SegQueue<ProfileScopeRecord>,
	current_scope_depth: AtomicUsize,
}
impl std::fmt::Debug for ProfilingFrame {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ProfilingFrame")
			.field("name", &self.name)
			.finish_non_exhaustive()
	}
}
impl ProfilingFrame {
	pub fn new(name: impl Into<String>) -> Self {
		let name = name.into();
		Self {
			name: name.clone(),
			#[cfg(feature = "enable_tracy_profiling")]
			frame_name: FrameName::new_leak(name),
			start: RwLock::new(Instant::now()),
			profile_scopes: SegQueue::new(),
			current_scope_depth: AtomicUsize::new(0),
		}
	}

	pub fn scope(&self, name: impl Into<String>) -> ProfileScope<'_> {
		let scope_depth = self.current_scope_depth.fetch_add(1, Ordering::Relaxed);
		ProfileScope::new(name.into(), scope_depth, self)
	}

	/// This is only public so that `ProfileScope` can submit its records when being dropped.
	/// Don't call this directly!
	pub(crate) fn _internal_submit_scope(&self, record: ProfileScopeRecord) {
		self.current_scope_depth.fetch_sub(1, Ordering::Relaxed);
		self.profile_scopes.push(record);
	}

	/// Returns None if the frame is not yet finished, i.e. current_scope_depth != 0
	pub fn finish(&self) -> Option<String> {
		if self.current_scope_depth.load(Ordering::Relaxed) != 0 {
			return None;
		}

		#[cfg(feature = "enable_tracy_profiling")]
		tracing_tracy::client::Client::running()
			.expect("tracy client not running")
			.secondary_frame_mark(self.frame_name);

		let end = Instant::now();

		let mut profile_scopes = Vec::with_capacity(self.profile_scopes.len());
		while let Some(scope_record) = self.profile_scopes.pop() {
			profile_scopes.push(scope_record);
		}

		profile_scopes.sort_by(|a, b| {
			a.start
				.cmp(&b.start)
				.then(a.scope_depth.cmp(&b.scope_depth))
		});
		let profile_scopes_formatted = profile_scopes
			.into_iter()
			.map(|scope| format!("    {}\n", scope))
			.collect::<String>();

		let frame_duration = end - (*self.start.read().unwrap());
		let frame_duration_secs = frame_duration.as_secs_f64();
		let frame_fps = 1.0 / frame_duration_secs;

		self.current_scope_depth.store(0, Ordering::Relaxed);
		*self.start.write().unwrap() = Instant::now();

		Some(format!(
			"{} Frame: {:.2} fps ({:.2})\n{}",
			self.name,
			frame_fps,
			dur::Duration::from_std(frame_duration),
			profile_scopes_formatted
		))
	}
}
