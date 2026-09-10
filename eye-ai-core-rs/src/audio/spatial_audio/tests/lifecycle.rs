use super::super::*;
use std::time::Instant;

#[test]
#[ignore = "requires OpenAL Soft null backend: ALSOFT_DRIVERS=null cargo test --offline real_playback_threads -- --ignored"]
fn real_playback_threads_exit_on_session_invalidation_and_engine_retirement() {
	let active = Arc::new(AtomicBool::new(true));
	let settings = Arc::new(RwLock::new(SpatialAudioSettings {
		depth_audio_paused: true,
		object_audio_paused: true,
		..Default::default()
	}));
	let content = Arc::new(SpatialAudioContent::new(
		AudioFileData {
			samples: vec![0; 48_000],
			sample_rate: 48_000,
		},
		HashMap::new(),
	));
	let profiling = Arc::new(ProfilingFrame::new_unretained("audio_lifecycle_test"));
	let create = || {
		SpatialAudio::new_in_session(
			settings.clone(),
			content.clone(),
			profiling.clone(),
			active.clone(),
			Arc::new(AtomicU64::new(0)),
		)
		.unwrap()
	};
	let first = create();
	drop(first); // Engine retirement must not invalidate the enclosing session.
	assert!(active.load(Ordering::Acquire));
	let second = create();
	active.store(false, Ordering::Release);
	let deadline = Instant::now() + Duration::from_secs(3);
	while !second.depth_audio_thread.as_ref().unwrap().is_finished()
		|| !second.object_audio_thread.as_ref().unwrap().is_finished()
	{
		assert!(
			Instant::now() < deadline,
			"playback thread survived invalidation"
		);
		std::thread::sleep(Duration::from_millis(2));
	}
	drop(second);
	let late = create();
	drop(late);
}
