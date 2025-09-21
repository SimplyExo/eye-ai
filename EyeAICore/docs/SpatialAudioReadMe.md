# SpatialAudio
The spatial Audio is an essential component of the EyeAI-Project, because it is the connection of the AI-Data (such as Depth-Estimation and Object-Detection) to the user.

## Documentation
### <ins>SpatialAudio</ins>:
Heart of the spatial audio. It retrieves the AI-Data, processes into **DepthAudioSourceData** and **ObjectAudioSourceData** it and sends it to **AudioMain** for the playback.

It also reads the JSON-File containing the information, at which point in the wav file of all objects a specific object is located.

### <ins>AudioMain</ins>:
Handles the actual playback of the audio, by creating a playback device and context and by managing all the sources, that play sound.

For the Depth-Audio there is a vector of sources, buffers and **DepthAudioSourceData** for the depth audio playback. So each of the sources in the sources vector is assigned a specified amount of buffers from the buffer vector and a **DepthAudioSourceData** struct from the **DepthAudioSourcesData**, which contains all the necessary information for the source.

For the Object-Audio there is only one source and buffer, because only one object will be play at a time, and a queue of **ObjectAudioSourcesData** structs, representing the objects and all information necessary for playing it. Additionally there is a vector containing the wav data of all possibles objects, for the playback.

**AudioMain** is divided into to loops which continuously play the Depth-Audio and Object-Audio:
#### DepthAudioLoop:
Fills buffers for the depth audio playback with the samples specified in the **DepthAudioSourceData** for this source. It also sets the right position for each sources, also according to the **DepthAudioSourceData** of this source.
#### ObjectAudioLoop:
First the sound of the object to play is loaded into a smaller buffer, by extracting the right time-span, specified int the **ObjectAudioSourceData** out of the vector containing the wav data.
This buffer is then attached to a source and played.

### <ins>DepthAudioSourceData</ins>
This struct contains all the data necessary for a source playing depth data:
* Frequency of the sound (in Hz)
* Duration of the sound (in s)
* The sample rate
* The total number of samples
* The position of the source
* The sound to be played (inside samples)

The samples are created by the **createAudioData** function. It creates a short click sound.

### <ins>ObjectAudioSourceData</ins>
This struct contains all the data necessary for a source playing object data:
* The object id (for identifying a object)
* The name of the object
* The beginning and ending of the sound in the big sound buffer (in ms)
* The position of the source 

### <ins>SpatialAudioSettings</ins>
This struct contains the general settings for the spatial audio:
* The buffers used for each Depth-Source
* The sample rate of the sounds
* The resolution of the image the ai processed (important for calculating the position of a pixel)
* The total number of sources responsible for the Depth-Audio
* The raw data of the .wav and .json files containing the data for object playback (provided by EyeAIApp on start)
* Whether the playback is paused
* The Frequency specified by the user
* The Buffer Duration (correlating with how often a click is played)

## <ins>CalculateSoundOrigin</ins>
Calculates the position (in 3D-Space) of a pixel from the AI-Data images, using basic trigonometric functions.

## <ins>ByteArrayParser</ins>
Converts the byte-array of the .wav file containing the object sound into a state readable by **libsndfile**, the library used for reading the wav file.