use crate::audio::{IVec2, SpatialAudioSettings, Vec3};

/// Converts a pixel angle (in degrees) into a unit direction in the horizontal
/// plane in front of the listener.
///
/// The listener faces the -Z axis (OpenAL default orientation), so:
///  0°   -> straight ahead (0, 0, -1)
///  -90° -> 100% left  (-1, 0, 0)
///  +90° -> 100% right (+1, 0, 0)
fn get_vector_to_origin(pixel_angle: f32) -> Vec3 {
	let pixel_angle_radians = pixel_angle * (std::f32::consts::PI / 180.0);
	Vec3 {
		x: pixel_angle_radians.sin(),
		y: 0.0,
		z: -pixel_angle_radians.cos(),
	}
}

pub struct CalculateSoundOrigin {
	max_angle: f32,
	distance_to_object: f32,
	pixel_coord_x: i32,
	picture_resolution_x: i32,
}
impl Default for CalculateSoundOrigin {
	fn default() -> Self {
		Self::new()
	}
}
impl CalculateSoundOrigin {
	pub fn new() -> Self {
		Self {
			max_angle: 90.0,
			distance_to_object: 0.0,
			pixel_coord_x: 0,
			picture_resolution_x: 0,
		}
	}

	pub fn calculate_sound_origin(&mut self, pixel_coords: IVec2, distance_to_object: f32) -> Vec3 {
		self.picture_resolution_x = SpatialAudioSettings::PICTURE_RESOLUTION.x;
		self.pixel_coord_x = pixel_coords.x;
		self.distance_to_object = distance_to_object;

		let pixel_angle = self.get_pixel_angle();
		self.get_origin(get_vector_to_origin(pixel_angle))
	}

	/// Maps the pixel column across the image to an angle between -max_angle
	/// (leftmost pixel) and +max_angle (rightmost pixel).
	fn get_pixel_angle(&self) -> f32 {
		let max_x = (self.picture_resolution_x - 1) as f32;
		let relative_position = if max_x > 0.0 {
			(self.pixel_coord_x as f32 / max_x).clamp(0.0, 1.0)
		} else {
			0.0
		};
		let angle_span = 2.0 * self.max_angle;
		-self.max_angle + relative_position * angle_span
	}

	fn get_origin(&self, directional_vector: Vec3) -> Vec3 {
		Vec3 {
			x: directional_vector.x * (self.distance_to_object + 1.0),
			y: 0.0,
			z: directional_vector.z * (self.distance_to_object + 1.0),
		}
	}
}
