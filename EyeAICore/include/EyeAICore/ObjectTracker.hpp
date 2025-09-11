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
	constexpr static int EXPECTED_FPS = 10;
	BYTETracker tracker{EXPECTED_FPS};
};