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

  private:
  	//FOV of the camera to ONE side
	float maxAngle = 80;
	float distanceToObject  =0;
	//coordinates of the pixel: from 1 to Resolution
	int pixelXCoordinate = 0;
	//int pixelYCoordinate;
	int pictureXResolution  =0;
	//int pictureYResolution;
	[[nodiscard]] float getPixelAngle() const;
	
	static std::array<float, 3> getVectorToOrigin(float pixelAngle);
	[[nodiscard]] std::array<float, 3> getOrigin(std::array<float, 3> directionalVector) const;
};