# Data Gathering
Application for gathering training data using opencv and flask for yolo model training.

## Usage
This application will be preinstalled in the EyeAIVisionPro OS image for data gathering. The following guide is made for this specific image.
1. After starting up the EyeAIVisionPro, connect a WiFi compatible device to the hotspot with following credentials:
```
SSID: EyeAIVisionPro
Password: 123456789
```
2. To access the control panel for your session, open the following web page in the browser:
```
http://192.168.4.1:5000/
```
3. If needed, change the capture delay, which sets the timespan between each image capture.

4. Click on the "Start recording" button to start the recording session. After closing the web page, your recording will continue until you come back and click the "Stop recording" button.

5. To access the images you recorded, connect the SD-Card to your computer. All images are now available on the FAT32 partition with the label "DATA". Inside of it, open the directory "output", in which all images were saved.
