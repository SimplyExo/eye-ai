#include "EyeAICore/ObjectTracker.hpp"
#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/utils/Profiling.hpp"

std::vector<ObjectTracker::TrackedBoundingBox>
ObjectTracker::update(std::span<const BoundingBox> detected_objects) {
	PROFILE_OBJECT_FUNCTION()

	std::vector<Object> byte_track_objects;
	byte_track_objects.reserve(detected_objects.size());

	for (const auto& box : detected_objects) {
		const Rect<float> rect{box.x1, box.y1, box.w, box.h};

		byte_track_objects.emplace_back(rect, box.cls, box.cnf);
	}

	const auto tracked_objects = tracker.update(byte_track_objects);
	std::vector<TrackedBoundingBox> tracked_object_boxes;
	tracked_object_boxes.reserve(tracked_objects.size());

	for (const auto& tracked_object : tracked_objects) {
		const int cls = tracked_object->getLabel();
		if (cls < 0 || cls >= (int)labels.size())
			continue;
		const std::string& cls_name = labels[cls];
		const auto& rect = tracked_object->getRect();
		const float x1 = rect.tl_x();
		const float y1 = rect.tl_y();
		const float x2 = rect.br_x();
		const float y2 = rect.br_y();
		const float w = x2 - x1;
		const float h = y2 - y1;
		const float cx = (x1 + x2) / 2.f;
		const float cy = (y1 + y2) / 2.f;
		const float cnf = tracked_object->getScore();
		const BoundingBox bounding_box{cls_name, cx, cy, w,	  h,  x1,
									   y1,		 x2, y2, cls, cnf};
		tracked_object_boxes.emplace_back(
			bounding_box, tracked_object->getTrackId()
		);
	}

	return tracked_object_boxes;
}