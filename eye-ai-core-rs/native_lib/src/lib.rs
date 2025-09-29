#![allow(non_snake_case)]

use eye_ai_core_rs::greet;

#[uniffi::export]
fn stringFromJni() -> String {
	greet()
}

uniffi::setup_scaffolding!("NativeLib");
