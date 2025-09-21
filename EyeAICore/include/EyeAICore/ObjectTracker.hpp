#pragma once

#include "ByteTrack/Object.h"
#include "EyeAICore/YoloModel.hpp"
#include <ByteTrack/BYTETracker.h>
#include <chrono>

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
	constexpr static float MAX_TRACKING_TIME_SECONDS = 10.f;
	constexpr static float EXPECTED_FPS = 10.f;
	/// For how many seconds a 100% confident prediction needs to be tracked in
	/// order to be considered valid
	constexpr static float MIN_WAITING_PREDICTION_TIME_BEFORE_VALID = 0.5f;
	BYTETracker tracker{MAX_TRACKING_TIME_SECONDS, EXPECTED_FPS};
	std::chrono::high_resolution_clock::time_point last_update =
		std::chrono::high_resolution_clock::now();
	/// sums up all the confidence of a tracked object
	std::unordered_map<int, float> tracked_object_valid_score;
};