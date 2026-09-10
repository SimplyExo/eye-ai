use super::*;
use std::sync::Arc;

fn retained_scope_count(frame: &ProfilingFrame) -> usize {
	frame.profile_scopes.len()
}

#[test]
fn unretained_frame_does_not_accumulate_records_across_starts() {
	let frame = ProfilingFrame::new_unretained("Audio");

	for _ in 0..3 {
		for _ in 0..10_000 {
			drop(frame.scope("audio_scope"));
		}

		assert_eq!(retained_scope_count(&frame), 0);
		assert!(frame.finish().is_some());
	}
}

#[test]
fn unretained_frame_parallel_scope_completion_stays_balanced() {
	let frame = Arc::new(ProfilingFrame::new_unretained("Audio"));

	std::thread::scope(|scope| {
		for _ in 0..8 {
			let frame = Arc::clone(&frame);
			scope.spawn(move || {
				for _ in 0..1_250 {
					drop(frame.scope("audio_scope"));
				}
			});
		}
	});

	assert_eq!(frame.current_scope_depth.load(Ordering::Acquire), 0);
	assert_eq!(retained_scope_count(&frame), 0);
	assert!(frame.finish().is_some());
}

#[test]
fn unretained_finish_returns_while_a_scope_is_active() {
	let frame = ProfilingFrame::new_unretained("Audio");
	let scope = frame.scope("audio_scope");

	assert!(frame.finish().is_none());
	drop(scope);
	assert!(frame.finish().is_some());
}

#[test]
fn retained_frame_records_are_consumed_by_finish() {
	let frame = ProfilingFrame::new("Object");
	drop(frame.scope("object_scope"));

	assert_eq!(retained_scope_count(&frame), 1);
	let info = frame.finish().expect("completed frame should be available");
	assert!(info.contains("object_scope"));
	assert_eq!(retained_scope_count(&frame), 0);
}
