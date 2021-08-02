### Algorithm - Object detection on Video with YOLO v3 
1. Reading input video
2. Load Yolo V3 Network
3. Read frames in the Loop
4. Get blob from the frame
5. Implement forward pass
6. Get bounding boxes
7. Do Non-maximum Suppression 
8. Draw bounding boxes with labels
9. Writting processed frames

### Output
New video file with detected objects, bounding boxes and labels

### To convert video into images

```python
$ffmpeg -i <video file.mp4> -vf fps=4 image-%d.jpeg

```

### Yolo V3 image test result

![yolo_result](https://github.com/joyjeni/ComputerVision/blob/main/s11_YoloV3/images/jenisha_car.jpeg)



