use super::super::*;
use std::{sync::mpsc, thread, time::Duration};

#[test]
fn delayed_settings_and_stale_exports_are_bound_to_their_original_session() {
	let a = beginSpatialAudioSession();
	let old = SPATIAL_AUDIO.get(a).unwrap();
	let settings_guard = old.settings.write().unwrap();
	let (entered_tx, entered_rx) = mpsc::channel();
	let settings_call = thread::spawn(move || {
		entered_tx.send(()).unwrap();
		setAudioSettings(a, 999.0, 3);
		setDepthAudioPaused(a, false);
		setObjectAudioPaused(a, false);
	});
	entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
	invalidateSpatialAudioSession(a);
	let b = beginSpatialAudioSession();
	let b_playback_epoch = SPATIAL_AUDIO
		.get(b)
		.unwrap()
		.object_audio_playback_epoch
		.clone();
	assert_eq!(b_playback_epoch.load(Ordering::Acquire), 0);
	setAudioSettings(b, 321.0, 5);
	setDepthAudioPaused(b, true);
	setObjectAudioPaused(b, true);
	assert_eq!(b_playback_epoch.load(Ordering::Acquire), 1);
	invalidateObjectAudioPlayback(b);
	assert_eq!(b_playback_epoch.load(Ordering::Acquire), 2);
	invalidateObjectAudioPlayback(a);
	assert_eq!(b_playback_epoch.load(Ordering::Acquire), 2);
	drop(settings_guard);
	settings_call.join().unwrap();
	createSpatialAudio(a); // Must not try opening a device, even without content.
	let mut depth = [1_000.0; 256 * 256];
	sendAIDataForSpatialAudio(a, (&mut depth).into(), vec![]);
	destroySpatialAudio(a);
	destroySpatialAudio(a);
	let fresh = SPATIAL_AUDIO.get(b).unwrap();
	assert!(fresh.is_active());
	let settings = fresh.settings.read().unwrap();
	assert_eq!(settings.frequency, 321.0);
	assert!(settings.depth_audio_paused && settings.object_audio_paused);
	drop(settings);
	destroySpatialAudio(b);
	assert!(SPATIAL_AUDIO.get(a).is_none());
	assert!(SPATIAL_AUDIO.get(b).is_none());
}
