#include "ByteTrack/Rect.h"

#include <algorithm>
#include <cmath>

template <typename T>
byte_track::Rect<T>::Rect(const T &x, const T &y, const T &width, const T &height) :
    tlwh({x, y, width, height})
{
}

template <typename T>
byte_track::Rect<T>::~Rect()
{
}

template <typename T>
const T& byte_track::Rect<T>::x() const
{
    return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::y() const
{
    return tlwh[1];
}

template <typename T>
const T& byte_track::Rect<T>::width() const
{
    return tlwh[2];
}

template <typename T>
const T& byte_track::Rect<T>::height() const
{
    return tlwh[3];
}

template <typename T>
T& byte_track::Rect<T>::x()
{
    return tlwh[0];
}

template <typename T>
T& byte_track::Rect<T>::y()
{
    return tlwh[1];
}

template <typename T>
T& byte_track::Rect<T>::width()
{
    return tlwh[2];
}

template <typename T>
T& byte_track::Rect<T>::height()
{
    return tlwh[3];
}

template <typename T>
const T& byte_track::Rect<T>::tl_x() const
{
    return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::tl_y() const
{
    return tlwh[1];
}

template <typename T>
T byte_track::Rect<T>::br_x() const
{
    return tlwh[0] + tlwh[2];
}

template <typename T>
T byte_track::Rect<T>::br_y() const
{
    return tlwh[1] + tlwh[3];
}

template <typename T>
byte_track::Tlbr<T> byte_track::Rect<T>::getTlbr() const
{
    return {
        tlwh[0],
        tlwh[1],
        tlwh[0] + tlwh[2],
        tlwh[1] + tlwh[3],
    };
}

template <typename T>
byte_track::Xyah<T> byte_track::Rect<T>::getXyah() const
{
    return {
        tlwh[0] + tlwh[2] / 2,
        tlwh[1] + tlwh[3] / 2,
        tlwh[2] / tlwh[3],
        tlwh[3],
    };
}

template<typename T>
float byte_track::Rect<T>::calcIoU(const Rect<T>& other) const
{
    // EyeAI sends continuous, normalized tlwh coordinates to ByteTrack. The
    // upstream pixel-index convention of adding one to widths/heights would
    // make distinct [0, 1] boxes overlap and is not valid here.
    const double first_x = static_cast<double>(tlwh[0]);
    const double first_y = static_cast<double>(tlwh[1]);
    const double first_width = static_cast<double>(tlwh[2]);
    const double first_height = static_cast<double>(tlwh[3]);
    const double second_x = static_cast<double>(other.tlwh[0]);
    const double second_y = static_cast<double>(other.tlwh[1]);
    const double second_width = static_cast<double>(other.tlwh[2]);
    const double second_height = static_cast<double>(other.tlwh[3]);

    const auto is_valid_box = [](double x, double y, double width, double height)
    {
        return std::isfinite(x) && std::isfinite(y) &&
            std::isfinite(width) && std::isfinite(height) &&
            width > 0.0 && height > 0.0;
    };
    if (!is_valid_box(first_x, first_y, first_width, first_height) ||
        !is_valid_box(second_x, second_y, second_width, second_height))
    {
        return 0.0f;
    }

    const double first_right = first_x + first_width;
    const double first_bottom = first_y + first_height;
    const double second_right = second_x + second_width;
    const double second_bottom = second_y + second_height;
    if (!std::isfinite(first_right) || !std::isfinite(first_bottom) ||
        !std::isfinite(second_right) || !std::isfinite(second_bottom))
    {
        return 0.0f;
    }

    const double intersection_width = std::max(
        0.0, std::min(first_right, second_right) - std::max(first_x, second_x));
    const double intersection_height = std::max(
        0.0, std::min(first_bottom, second_bottom) - std::max(first_y, second_y));
    const double intersection_area = intersection_width * intersection_height;
    const double union_area = first_width * first_height + second_width * second_height -
        intersection_area;
    if (!std::isfinite(intersection_area) || !std::isfinite(union_area) ||
        union_area <= 0.0)
    {
        return 0.0f;
    }

    const double iou = intersection_area / union_area;
    if (!std::isfinite(iou))
    {
        return 0.0f;
    }
    return static_cast<float>(std::clamp(iou, 0.0, 1.0));
}

template<typename T>
byte_track::Rect<T> byte_track::generate_rect_by_tlbr(const byte_track::Tlbr<T>& tlbr)
{
    return byte_track::Rect<T>(tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]);
}

template<typename T>
byte_track::Rect<T> byte_track::generate_rect_by_xyah(const byte_track::Xyah<T>& xyah)
{
    const auto width = xyah[2] * xyah[3];
    return byte_track::Rect<T>(xyah[0] - width / 2, xyah[1] - xyah[3] / 2, width, xyah[3]);
}

// explicit instantiation
template class byte_track::Rect<int>;
template class byte_track::Rect<float>;

template byte_track::Rect<int> byte_track::generate_rect_by_tlbr<int>(const byte_track::Tlbr<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_tlbr<float>(const byte_track::Tlbr<float>&);

template byte_track::Rect<int> byte_track::generate_rect_by_xyah<int>(const byte_track::Xyah<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_xyah<float>(const byte_track::Xyah<float>&);
