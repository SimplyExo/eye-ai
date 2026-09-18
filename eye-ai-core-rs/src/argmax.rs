use rayon::prelude::*;
use std::simd::cmp::SimdPartialOrd;
use std::simd::{Select, f32x8, i32x8};

/// Finds, for every element position, the highest value and the index of the
/// plane it came from.
///
/// `values` holds `num_classes` planes of `num_elements` elements each, laid out
/// class-major. The element range is split across rayon workers, and every
/// worker scans class plane by class plane, which keeps the comparisons and the
/// index selection branchless in SIMD lanes. Ties keep the lower class index,
/// matching a strict `>` scan.
pub(crate) fn argmax_over_planes(
	values: &[f32],
	num_elements: usize,
	num_classes: usize,
	initial: f32,
) -> (Vec<f32>, Vec<i32>) {
	let mut max_values = vec![initial; num_elements];
	let mut max_indices = vec![0i32; num_elements];

	if num_elements == 0 || num_classes == 0 {
		return (max_values, max_indices);
	}

	let planes = values
		.chunks_exact(num_elements)
		.take(num_classes)
		.collect::<Vec<_>>();

	// one chunk per worker, rounded up to a multiple of 8 so the SIMD inner
	// loop stays aligned; the (unique) tail chunk falls back to scalar
	let num_threads = rayon::current_num_threads();
	let chunk_size = num_elements.div_ceil(num_threads);
	let chunk_size = chunk_size.div_ceil(8) * 8;

	max_values
		.par_chunks_mut(chunk_size)
		.zip(max_indices.par_chunks_mut(chunk_size))
		.enumerate()
		.for_each(|(chunk_index, (value_chunk, index_chunk))| {
			let start = chunk_index * chunk_size;
			for (class_index, class_plane) in planes.iter().enumerate() {
				let class = i32x8::splat(class_index as i32);
				let class_segment = &class_plane[start..start + value_chunk.len()];
				let (value_chunks, value_remainder) = value_chunk.as_chunks_mut::<8>();
				let (index_chunks, index_remainder) = index_chunk.as_chunks_mut::<8>();
				let (class_chunks, class_remainder) = class_segment.as_chunks::<8>();

				for ((value_chunk_i, index_chunk_i), class_chunk) in value_chunks
					.iter_mut()
					.zip(index_chunks.iter_mut())
					.zip(class_chunks)
				{
					let candidate = f32x8::from_slice(class_chunk);
					let current = f32x8::from_slice(value_chunk_i);
					let mask = candidate.simd_gt(current);
					value_chunk_i.copy_from_slice(mask.select(candidate, current).as_array());

					let index = i32x8::from_slice(index_chunk_i);
					index_chunk_i.copy_from_slice(mask.select(class, index).as_array());
				}

				for ((value, index), &candidate) in value_remainder
					.iter_mut()
					.zip(index_remainder.iter_mut())
					.zip(class_remainder)
				{
					if candidate > *value {
						*value = candidate;
						*index = class_index as i32;
					}
				}
			}
		});

	(max_values, max_indices)
}
