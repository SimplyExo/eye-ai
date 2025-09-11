#pragma once

#include <array>

/*
This class converts a pixel of the depth estimation
data into a position in 3d space, depending on the FOV
of the camera and the distance of the object in the pixel
*/

class CalculateSoundOrigin {
  public:
	std::array<float, 3> calculateSoundOrigin(
		std::array<int, 2> pixelCoordinates,
		float distanceToObject, int pictureXResolution
	);
	float pixelAngle;

  private:
  	//FOV of the camera to ONE side
	float maxAngle = 90;
	float distanceToObject;
	//coordinates of the pixel: from 1 to Resolution
	int pixelXCoordinate;
	//int pixelYCoordinate;
	int pictureXResolution;
	//int pictureYResolution;
	float getPixelAngle();
	
	std::array<float, 3> getVectorToOrigin(float pixelAngle);
	std::array<float, 3> getOrigin(std::array<float, 3> directionalVector);
};