#pragma once

#include "ByteTrack/Object.h"
#include "EyeAICore/YoloModel.hpp"
#include <ByteTrack/BYTETracker.h>

using byte_track::BYTETracker, byte_track::Rect, byte_track::Object;

class ObjectTracker {
  private:
	using BoundingBox = YoloModel::BoundingBox;

  public:
	struct TrackedBoundingBox {
		BoundingBox bounding_box;
		int tracking_id;

		TrackedBoundingBox() = delete;
		explicit TrackedBoundingBox(BoundingBox bounding_box, int tracking_id)
			: bounding_box(std::move(bounding_box)), tracking_id(tracking_id) {}
	};

	explicit ObjectTracker(std::vector<std::string> labels)
		: labels(std::move(labels)) {}

	std::vector<TrackedBoundingBox>
	update(std::span<const BoundingBox> detected_objects);

  private:
	const std::vector<std::string> labels;
	/// TODO: update byte-track fork to allow updating fps / tracked buffer size
	/// at runtime
	constexpr static int EXPECTED_FPS = 10;
	constexpr static int TRACKED_BUFFER_COUNT = 60;
	/// How many times a 100% confident prediction needs to be tracked in order to be
	/// considered valid (for ex. 150% -> 1x100% + 1x50% or 3x50% etc.
	constexpr static float MIN_VALID_PREDICTION_SCORE = 1.5f;
	BYTETracker tracker{EXPECTED_FPS, TRACKED_BUFFER_COUNT};
	/// sums up all the confidence of a tracked object
	std::unordered_map<int, float> tracked_object_valid_score;
};