#include <ByteTrack/BYTETracker.h>
#include <ByteTrack/c_api.h>

using namespace byte_track;

Rect<float> from_c_style_rect_float(const byte_track_Rect_float &rect) {
  return Rect<float>(rect.x, rect.y, rect.width, rect.height);
}

byte_track_Rect_float to_c_style_rect_float(const Rect<float> &rect) {
  return byte_track_Rect_float{rect.x(), rect.y(), rect.width(), rect.height()};
}

byte_track_STrack from_c_style_STrack(BYTETracker::STrackPtr s) {
  return byte_track_STrack{
      .rect = to_c_style_rect_float(s->getRect()),
      .state = static_cast<byte_track_STrackState>(s->getSTrackState()),
      .is_activated = s->isActivated(),
      .score = s->getScore(),
      .track_id = static_cast<int>(s->getTrackId()),
      .frame_id = static_cast<int>(s->getFrameId()),
      .start_frame_id = static_cast<int>(s->getStartFrameId()),
      .tracklet_length = static_cast<int>(s->getTrackletLength()),
      .label = s->getLabel()};
}

Object from_c_style_Object(const byte_track_Object &object) {
  return Object(from_c_style_rect_float(object.rect), object.label,
                object.prob);
}

byte_track_Object to_c_style_Object(const Object &object) {
  return byte_track_Object{.rect = to_c_style_rect_float(object.rect),
                           .label = object.label,
                           .prob = object.prob};
}

extern "C" {
void *byte_track_BYTETracker_create(double max_time_lost_seconds,
                                    float track_thresh,
                                    float high_thresh, float match_thresh) {
  BYTETracker *tracker =
      new BYTETracker(max_time_lost_seconds, track_thresh, high_thresh,
                      match_thresh);
  return tracker;
}

void byte_track_BYTETracker_destroy(void *tracker) {
  delete static_cast<BYTETracker *>(tracker);
}

void byte_track_BYTETracker_update(void *tracker,
                                   const byte_track_Object *objects,
                                   int num_objects,
                                   uint64_t elapsed_nanoseconds,
                                   byte_track_STrack **out_stracks,
                                   int *out_num_stracks) {
  std::vector<Object> objects_vec;
  objects_vec.reserve(num_objects);
  for (int i = 0; i < num_objects; ++i) {
    objects_vec.push_back(from_c_style_Object(objects[i]));
  }

  BYTETracker *byte_tracker = static_cast<BYTETracker *>(tracker);
  std::vector<BYTETracker::STrackPtr> stracks =
      byte_tracker->update(objects_vec, elapsed_nanoseconds);

  byte_track_STrack *stracks_array = new byte_track_STrack[stracks.size()];
  for (size_t i = 0; i < stracks.size(); ++i) {
    const auto &s = stracks[i];
    stracks_array[i] = from_c_style_STrack(s);
  }
  *out_stracks = stracks_array;
  *out_num_stracks = stracks.size();
}

void byte_track_STrack_array_destroy(byte_track_STrack *stracks_array) {
  delete[] stracks_array;
}

void byte_track_BYTETracker_set_max_time_lost(void *tracker,
                                              double max_time_lost_seconds) {
  BYTETracker *byte_tracker = static_cast<BYTETracker *>(tracker);
  byte_tracker->setMaxTimeLost(max_time_lost_seconds);
}
}
