\page AIModelSources AI Model sources

## Speech Recognition: Vosk

Documentation: <https://alphacephei.com/vosk/>

Model source: <https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip>

## Depth Estimation Models: MiDaS 2.1

Repo: <https://github.com/isl-org/MiDaS>

Download source: <https://aihub.qualcomm.com/models/midas>

**midas_v2_1_256x256.tflite**:

    Input shape: float32[1, 256, 256, 3]

**midas_v2_1_256x256_quantized.tflite**:

    Input shape: uint8[1, 256, 256, 3]

## Object Detection: yolo26n

Download source (.pt): <https://storage.googleapis.com/alpha-ultralytics-eu/users/user_38BClQ6JUxGqFjffFWUROzyWDIf/models/696741ee97b2c1b662ad39d9/yolo26n.pt?X-Goog-Algorithm=GOOG4-HMAC-SHA256&X-Goog-Credential=GOOG1EVYATYKKOGSZSQSTG4P6ISYXQTE4HWDCBAWNAEWGN34SPK6JC6CK22HP%2F20260908%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260908T201130Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22yolo26n.pt%22&X-Goog-Signature=11b2a8e215be09078efb87c83683382b05af6c6deec4b4a2d4546519cd4b1cd2>

Exported with: ```yolo export model=./yolo26n.pt format=tflite```
(`ultralytics/ultralytics:latest` docker container is recommended)

## Semantic Segmentation: yolo26n-sem

Download source: <https://storage.googleapis.com/alpha-ultralytics-eu/users/user_38BClQ6JUxGqFjffFWUROzyWDIf/models/6a0f16bd97f68c65eef2e20c/yolo26n-sem.pt?X-Goog-Algorithm=GOOG4-HMAC-SHA256&X-Goog-Credential=GOOG1EVYATYKKOGSZSQSTG4P6ISYXQTE4HWDCBAWNAEWGN34SPK6JC6CK22HP%2F20260909%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260909T122541Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22yolo26n-sem.pt%22&X-Goog-Signature=406900f74fa7372d7a3b9b28d3934d6ccb0cff89d01aa9dad4a0b3d52da7239c>

Export command: `yolo export model=yolo26n-sem format=tflite imgsz=256`
