#ifndef _BYTETRACK_C_API_H
#define _BYTETRACK_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct {
  float x;
  float y;
  float width;
  float height;
} byte_track_Rect_float;

// Test-only inspection hook for the native Rect implementation. It is not
// exposed by the Rust production API.
extern float byte_track_Rect_float_calc_iou_for_testing(
    byte_track_Rect_float first, byte_track_Rect_float second);

typedef struct {
  byte_track_Rect_float rect;
  int label;
  float prob;
} byte_track_Object;

typedef enum {
  New = 0,
  Tracked = 1,
  Lost = 2,
  Removed = 3,
} byte_track_STrackState;

typedef struct {
  byte_track_Rect_float rect;
  byte_track_STrackState state;
  bool is_activated;
  float score;
  int track_id;
  int frame_id;
  int start_frame_id;
  int tracklet_length;
  int label;
} byte_track_STrack;

extern void *byte_track_BYTETracker_create(double max_time_lost_seconds,
                                           float track_thresh,
                                           float high_thresh,
                                           float match_thresh);

extern void byte_track_BYTETracker_destroy(void *tracker);

// DO NOT forget to destroy the allocated out_stracks array by calling
// byte_track_STrack_array_destroy(out_stracks)!
extern void byte_track_BYTETracker_update(void *tracker,
                                          const byte_track_Object *objects,
                                          int num_objects,
                                          uint64_t elapsed_nanoseconds,
                                          byte_track_STrack **out_stracks,
                                          int *out_num_stracks);

// Benchmark-only strategy control. It deliberately shares the exact same
// BYTETracker instance and association path as the production update above.
extern void byte_track_BYTETracker_update_for_benchmark(
    void *tracker,
    const byte_track_Object *objects,
    int num_objects,
    uint64_t elapsed_nanoseconds,
    bool enable_motion_prediction,
    float process_noise_scale,
    byte_track_STrack **out_stracks,
    int *out_num_stracks);

extern void byte_track_STrack_array_destroy(byte_track_STrack *stracks_array);

extern void byte_track_BYTETracker_set_max_time_lost(
    void *tracker, double max_time_lost_seconds);

#ifdef __cplusplus
}
#endif

#endif // _BYTETRACK_C_API_H
