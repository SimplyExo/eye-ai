use super::*;
use std::{
	panic::{AssertUnwindSafe, catch_unwind},
	sync::{atomic::AtomicUsize, mpsc},
	thread,
	time::Duration,
};

const TIMEOUT: Duration = Duration::from_secs(5);

struct Engine {
	updates: usize,
	running: Arc<AtomicBool>,
	thread: Option<thread::JoinHandle<()>>,
}

impl Engine {
	fn new(session: &AudioSession<Self>, live: &Arc<AtomicUsize>) -> Self {
		let active = session.active.clone();
		let running = Arc::new(AtomicBool::new(true));
		let thread_running = running.clone();
		let live = live.clone();
		live.fetch_add(1, Ordering::SeqCst);
		let thread = thread::spawn(move || {
			while active.load(Ordering::Acquire) && thread_running.load(Ordering::Acquire) {
				thread::park_timeout(Duration::from_millis(2));
			}
			live.fetch_sub(1, Ordering::SeqCst);
		});
		Self {
			updates: 0,
			running,
			thread: Some(thread),
		}
	}
}

impl Drop for Engine {
	fn drop(&mut self) {
		self.running.store(false, Ordering::Release);
		let thread = self.thread.take().unwrap();
		thread.thread().unpark();
		thread.join().unwrap();
	}
}

fn create(session: &AudioSession<Engine>, live: &Arc<AtomicUsize>) {
	session
		.change(|| Ok::<_, ()>(Engine::new(session, live)), |_| false)
		.unwrap();
}

mod recovery;
mod session_isolation;
