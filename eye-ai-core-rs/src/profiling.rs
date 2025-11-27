use crossbeam::queue::SegQueue;
use std::{
	sync::{
		RwLock,
		atomic::{AtomicUsize, Ordering},
	},
	time::Instant,
};

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
	start: RwLock<Instant>,
	profile_scopes: SegQueue<ProfileScopeRecord>,
	current_scope_depth: AtomicUsize,
	pub formatted_info: RwLock<String>,
}

impl ProfilingFrame {
	pub fn new(name: impl Into<String>) -> Self {
		Self {
			name: name.into(),
			start: RwLock::new(Instant::now()),
			profile_scopes: SegQueue::new(),
			current_scope_depth: AtomicUsize::new(0),
			formatted_info: RwLock::new(String::new()),
		}
	}

	pub fn scope(&self, name: impl Into<String>) -> ProfileScope<'_> {
		let scope_depth = self.current_scope_depth.fetch_add(1, Ordering::Relaxed);
		ProfileScope::new(name.into(), scope_depth, self)
	}

	/// This is only public so that `ProfileScope` can submit its records when being dropped.
	/// Don't call this directly!
	pub(crate) fn _internal_submit_scope(&self, record: ProfileScopeRecord) {
		let previous_scope_depth = self.current_scope_depth.fetch_sub(1, Ordering::Relaxed);
		self.profile_scopes.push(record);

		// in other words: if the top most scope finished -> finish this frame
		if previous_scope_depth == 1 {
			*self.formatted_info.write().unwrap() = self.finish();
		}
	}

	// gets called, when the top most scope ends
	fn finish(&self) -> String {
		// TODO: tracy framemark!
		let end = Instant::now();

		/*if self.current_scope_depth.load(Ordering::Relaxed) != 0 {
			panic!(
				"frame {}: finish called, but current_scope_depth is {}",
				self.name,
				self.current_scope_depth.load(Ordering::Relaxed)
			);
		}*/
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

		format!(
			"{} Frame: {:.2} fps ({:.2})\n{}",
			self.name,
			frame_fps,
			dur::Duration::from_std(frame_duration),
			profile_scopes_formatted
		)
	}
}
