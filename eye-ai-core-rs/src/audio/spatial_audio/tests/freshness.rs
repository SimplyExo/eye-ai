use super::super::*;
use std::sync::atomic::AtomicU64;

fn queue() -> (Arc<AtomicU64>, ObjectAudioQueue) {
	let epoch = Arc::new(AtomicU64::new(0));
	(epoch.clone(), ObjectAudioQueue::new(epoch))
}

fn item(id: usize, x: f32) -> ObjectAudioSourceData {
	ObjectAudioSourceData {
		object_id: id,
		name: "person".into(),
		sound_begin: 0,
		sound_end: 1,
		position: Vec3 { x, y: 0.0, z: 1.0 },
	}
}

#[test]
fn empty_snapshot_clears_pending_without_interrupting_popped_playback() {
	let (playback_epoch, mut queue) = queue();
	queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
	let (playing_epoch, _) = queue.pop().unwrap();
	queue.update(VecDeque::new());
	assert!(queue.pop().is_none());
	assert_eq!(playback_epoch.load(Ordering::Acquire), playing_epoch);
}

#[test]
fn explicit_invalidation_stops_popped_playback_and_discards_pending_items() {
	let (playback_epoch, mut queue) = queue();
	queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
	let (playing_epoch, _) = queue.pop().unwrap();
	playback_epoch.fetch_add(1, Ordering::AcqRel);
	assert!(queue.pop().is_none());
	assert_ne!(playback_epoch.load(Ordering::Acquire), playing_epoch);

	queue.update(VecDeque::from([item(1, 3.0)]));
	let (new_epoch, source) = queue.pop().unwrap();
	assert_ne!(new_epoch, playing_epoch);
	assert_eq!(source.position.x, 3.0);
}

#[test]
fn fresh_updates_preserve_order_refresh_positions_and_remove_missing_objects() {
	let (_, mut queue) = queue();
	queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
	queue.update(VecDeque::from([item(2, 4.0), item(3, 3.0)]));
	let (epoch, first) = queue.pop().unwrap();
	assert_eq!(epoch, 0);
	assert_eq!(first.object_id, 2);
	assert_eq!(first.position.x, 4.0);
	assert_eq!(queue.pop().unwrap().1.object_id, 3);
	assert!(queue.pop().is_none());
}

#[test]
fn queue_keeps_existing_six_source_bound() {
	let (_, mut queue) = queue();
	queue.update((0..10).map(|id| item(id, 0.0)).collect());
	assert_eq!(queue.pending.len(), 6);
	assert_eq!(queue.pop().unwrap().1.object_id, 4);
}
