#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(CalculateSoundOriginTest, CorrectOutput){
    constexpr float tolerance = 1e-4f;

    std::array<int,2> input_coordinates = {60, 1};
    float input_distance = 2.0f;

    std::array<float, 3> expected_output = {1.90211f, 0.618034f, 0.0f};

    std::array<float, 3> output = CalculateSoundOrigin().calculateSoundOrigin(input_coordinates, input_distance);

    EXPECT_THAT(output, Pointwise(FloatNear(tolerance), expected_output));
}