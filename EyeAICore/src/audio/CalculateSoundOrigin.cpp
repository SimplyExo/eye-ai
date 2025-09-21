#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include <cmath>
#include "EyeAICore/utils/Profiling.hpp"
#include <numbers>

std::array<float, 3> CalculateSoundOrigin::calculateSoundOrigin(
	std::array<int, 2> pixelCoordinates,
	float distanceToObject,
	int pictureXResolution
) {
	PROFILE_AUDIO_FUNCTION()
	this->pictureXResolution = pictureXResolution;
	this->pixelXCoordinate = pixelCoordinates[0];
	// this->pixelYCoordinate = pixelCoordinates[1];
	this->distanceToObject = distanceToObject;

	float pixelAngle = getPixelAngle();

	std::array<float, 3> vectorToOrigin = getVectorToOrigin(pixelAngle);

	return getOrigin(vectorToOrigin);
}

/*
Calculates the angle that the pixel has, relative to the POV of the camera
The angle can be negative (right side of the camera), or positive (left side
of the camera).
*/
float CalculateSoundOrigin::getPixelAngle() const {
	PROFILE_AUDIO_FUNCTION()
	/*
	adjust the x-coordinate of the pixel, so that there are positive and
	negative values, depending whether the pixel is to the left or right
	of the middle
	*/
	int halfXResolution = std::ceil((float)pictureXResolution / 2);
	int adjustedPixelXCoordinate = pixelXCoordinate > halfXResolution
									   ? pixelXCoordinate - halfXResolution
									   : pixelXCoordinate - halfXResolution - 1;

	float relativeAngle =
		(float)adjustedPixelXCoordinate / (float)halfXResolution;
	return relativeAngle * maxAngle;
}

/*
Calculates the direction from the camera to the origin.
It does this by ignoring the distance, and assuming the origin
is on a 1m circle around the camera, it's position only depending
on the angle from the camera
*/
std::array<float, 3> CalculateSoundOrigin::getVectorToOrigin(float pixelAngle){
	PROFILE_AUDIO_FUNCTION()
	float x1_vector = NAN; // x1 meaning to the side of the camera
	float x2_vector = NAN; // x2 meaning in front of the camera

	float pixelAngleRadian = pixelAngle * (float) (std::numbers::pi / 180);

	x1_vector = std::sin(pixelAngleRadian);
	x2_vector = std::cos(pixelAngleRadian);
	return std::array<float, 3>{x1_vector, x2_vector, 0};
}

/*
Calculates the origin of the Sound, by multiplying the
directional vector to the sound with the distance of the
pixel from the camera
*/
std::array<float, 3>
CalculateSoundOrigin::getOrigin(std::array<float, 3> directionalVector) const{
	PROFILE_AUDIO_FUNCTION()
	float x1_position = NAN; // x1 meaning in front of the camera
	float x2_position = NAN; // x2 meaning to the side of the camera

	x1_position = directionalVector[0] * (distanceToObject + 1);
	x2_position = directionalVector[1] * (distanceToObject + 1);

	return std::array<float, 3>{x1_position, x2_position, 0};
}