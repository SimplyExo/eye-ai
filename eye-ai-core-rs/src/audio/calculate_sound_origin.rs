use crate::audio::{IVec2, SpatialAudioSettings, Vec3};

fn get_vector_to_origin(pixel_angle: f32) -> Vec3 {
	let pixel_angle_radians = pixel_angle * (std::f32::consts::PI / 180.0);
	Vec3 {
		x: pixel_angle_radians.sin(),
		y: pixel_angle_radians.cos(),
		z: 0.0,
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
			max_angle: 80.0,
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

	fn get_pixel_angle(&self) -> f32 {
		let half_x_resolution = (self.picture_resolution_x as f32 / 2.0).ceil();
		let adjusted_pixel_coord_x = if self.pixel_coord_x as f32 > half_x_resolution {
			self.pixel_coord_x as f32 - half_x_resolution
		} else {
			self.pixel_coord_x as f32 - half_x_resolution - 1.0
		};
		let relative_angle = adjusted_pixel_coord_x / half_x_resolution;
		relative_angle * self.max_angle
	}

	fn get_origin(&self, directional_vector: Vec3) -> Vec3 {
		Vec3 {
			x: directional_vector.x * (self.distance_to_object + 1.0),
			y: directional_vector.y * (self.distance_to_object + 1.0),
			z: 0.0,
		}
	}
}
