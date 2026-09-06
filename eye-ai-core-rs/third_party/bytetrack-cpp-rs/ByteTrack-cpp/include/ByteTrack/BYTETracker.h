#pragma once

#include "ByteTrack/STrack.h"
#include "ByteTrack/lapjv.h"
#include "ByteTrack/Object.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace byte_track
{
class BYTETracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    BYTETracker(double max_time_lost_seconds = 5.0,
                const float& track_thresh = 0.5,
                const float& high_thresh = 0.6,
                const float& match_thresh = 0.8);
    ~BYTETracker();

    // elapsed_nanoseconds is the monotonic duration since the previous actual
    // detector/tracker update. A source frame skipped before detection must not
    // call this method.
    std::vector<STrackPtr> update(const std::vector<Object>& objects,
                                  std::uint64_t elapsed_nanoseconds);

    // Isolated benchmark entry point. Production uses update(), whose behavior
    // remains prediction enabled with an unscaled process-noise covariance.
    std::vector<STrackPtr> updateForBenchmark(
        const std::vector<Object>& objects,
        std::uint64_t elapsed_nanoseconds,
        bool enable_motion_prediction,
        float process_noise_scale);

    void setMaxTimeLost(double max_time_lost_seconds);

private:
    std::vector<STrackPtr> updateImpl(const std::vector<Object>& objects,
                                      std::uint64_t elapsed_nanoseconds,
                                      bool enable_motion_prediction,
                                      float process_noise_scale);

    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

private:
    const float track_thresh_;
    const float high_thresh_;
    const float match_thresh_;
    std::uint64_t max_time_lost_nanoseconds_;
    std::uint64_t current_time_nanoseconds_;

    // Sequence number retained for ByteTrack's ordering/output semantics; it
    // no longer represents elapsed time or lost-track lifetime.
    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
};
}
