#include "ByteTrack/KalmanFilter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

byte_track::KalmanFilter::KalmanFilter(const float& std_weight_position,
                                       const float& std_weight_velocity) :
    std_weight_position_(std_weight_position),
    std_weight_velocity_(std_weight_velocity)
{
    motion_mat_ = Eigen::MatrixXf::Identity(8, 8);
    update_mat_ = Eigen::MatrixXf::Identity(4, 8);
}

void byte_track::KalmanFilter::initiate(StateMean &mean, StateCov &covariance, const DetectBox &measurement)
{
    mean.block<1, 4>(0, 0) = measurement.block<1, 4>(0, 0);
    mean.block<1, 4>(0, 4) = Eigen::Vector4f::Zero();

    StateMean std;
    std(0) = 2 * std_weight_position_ * measurement[3];
    std(1) = 2 * std_weight_position_ * measurement[3];
    std(2) = 1e-2;
    std(3) = 2 * std_weight_position_ * measurement[3];
    std(4) = 10 * std_weight_velocity_ * measurement[3];
    std(5) = 10 * std_weight_velocity_ * measurement[3];
    std(6) = 1e-5;
    std(7) = 10 * std_weight_velocity_ * measurement[3];

    StateMean tmp = std.array().square();
    covariance = tmp.asDiagonal();
}

void byte_track::KalmanFilter::predict(StateMean &mean, StateCov &covariance,
                                       double elapsed_seconds)
{
    constexpr size_t ndim = 4;
    const double normalized_dt_double =
        std::isfinite(elapsed_seconds) && elapsed_seconds > 0.0
            ? elapsed_seconds / REFERENCE_INTERVAL_SECONDS
            : 0.0;
    const float dt = static_cast<float>(normalized_dt_double);

    for (size_t i = 0; i < ndim; i++)
    {
        motion_mat_(i, ndim + i) = dt;
    }

    StateMean std;
    std(0) = std_weight_position_ * mean(3);
    std(1) = std_weight_position_ * mean(3);
    std(2) = 1e-2;
    std(3) = std_weight_position_ * mean(3);
    std(4) = std_weight_velocity_ * mean(3);
    std(5) = std_weight_velocity_ * mean(3);
    std(6) = 1e-5;
    std(7) = std_weight_velocity_ * mean(3);

    // Preserve the original marginal process-noise variances at dt=1 while
    // extending them to variable time with a continuous constant-velocity
    // model. Each position component has independent diffusion and each
    // velocity component has white noise whose exact discretization adds the
    // dt^3/3 position variance and dt^2/2 position/velocity covariance terms.
    // Subtracting q_velocity/3 from q_position keeps the dt=1 position
    // variance calibrated to the original ByteTrack value.
    StateCov motion_cov = StateCov::Zero();
    const float dt_squared = dt * dt;
    const float dt_cubed = dt_squared * dt;
    for (size_t i = 0; i < ndim; i++)
    {
        const float reference_position_variance = std(i) * std(i);
        const float velocity_diffusion = std(ndim + i) * std(ndim + i);
        const float position_diffusion =
            std::max(0.0f, reference_position_variance - velocity_diffusion / 3.0f);

        motion_cov(i, i) =
            position_diffusion * dt + velocity_diffusion * dt_cubed / 3.0f;
        motion_cov(i, ndim + i) = velocity_diffusion * dt_squared / 2.0f;
        motion_cov(ndim + i, i) = motion_cov(i, ndim + i);
        motion_cov(ndim + i, ndim + i) = velocity_diffusion * dt;
    }

    mean = motion_mat_ * mean.transpose();
    covariance = motion_mat_ * covariance * (motion_mat_.transpose()) + motion_cov;
}

void byte_track::KalmanFilter::update(StateMean &mean, StateCov &covariance, const DetectBox &measurement)
{
    StateHMean projected_mean;
    StateHCov projected_cov;
    project(projected_mean, projected_cov, mean, covariance);

    Eigen::Matrix<float, 4, 8> B = (covariance * (update_mat_.transpose())).transpose();
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose();
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean;

    const auto tmp = innovation * (kalman_gain.transpose());
    mean = (mean.array() + tmp.array()).matrix();
    covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
}

void byte_track::KalmanFilter::project(StateHMean &projected_mean, StateHCov &projected_covariance,
                                       const StateMean& mean, const StateCov& covariance)
{
    DetectBox std;
    std << std_weight_position_ * mean(3),
           std_weight_position_ * mean(3),
           1e-1,
           std_weight_position_ * mean(3);

    projected_mean = update_mat_ * mean.transpose();
    projected_covariance = update_mat_ * covariance * (update_mat_.transpose());

    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    projected_covariance += diag.array().square().matrix();
}
