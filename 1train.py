from ultralytics import YOLO


if __name__ == '__main__':


    model = YOLO("D:/Program/LI-YOLOv8/ultralytics/cfg/models/v8/"
                 "LI-YOLOv8.yaml")
    results = model.train(data="D:/Program/yolov8-main/ultralytics/cfg/datasets/RSOD.yaml",
                         imgsz=640, epochs=200, batch=16, device=0, workers=8,conf=0.5, mixup=0.5,
                          mosaic=0.5, patience=0, lr0=0.01, lrf=0.1, close_mosaic=10)








