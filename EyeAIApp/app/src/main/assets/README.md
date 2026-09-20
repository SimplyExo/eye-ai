# AI Model sources

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

> [!note]
> The original model download links were temporary, time-limited signed URLs (per uploader/Ultralytics).
> They expire quickly and are therefore not suitable as stable references. Use the official
> <https://github.com/ultralytics/assets/releases> (or the Ultralytics Docker CLI) to obtain the `.pt` weights.

Download source: <https://github.com/ultralytics/assets/releases>

Exported with: ```yolo export model=./yolo26n.pt format=tflite```
(`ultralytics/ultralytics:latest` docker container is recommended)

## Semantic Segmentation: yolo26n-sem

Download source: <https://github.com/ultralytics/assets/releases>

Export command: `yolo export model=yolo26n-sem format=tflite imgsz=256`
