#pragma once

/*
Struct of all the data a source needs to play a specific
sound at a specific location
*/

struct AudioSourceData{
    float frequency;
    float x1_position; //to the side of the camera
    float x2_position; //in front of the camera
    float x3_position; // over/under the camera (currently not used)
};