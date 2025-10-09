pub struct IVec2 {
	pub x: i32,
	pub y: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vec3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
}
impl From<Vec3> for [f32; 3] {
	fn from(value: Vec3) -> Self {
		[value.x, value.y, value.z]
	}
}
#[derive(Debug, Clone, Copy, Default)]
pub struct Vec2 {
	pub x: f32,
	pub y: f32,
}
impl Vec2 {
	pub fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}
	pub fn square_distance(&self, other: &Self) -> f32 {
		(self.x - other.x).powi(2) + (self.y - other.y).powi(2)
	}
}
