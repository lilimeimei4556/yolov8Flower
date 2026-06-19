from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/train-2/weights/best.pt")
# 预测（⽀持图⽚、⽂件夹、视频、URL）
results = model.predict(
    source="D:/picture/bus.jpg",  # 可改为本地图⽚路径
    imgsz=640,
    conf=0.25,  # 置信度
    device="cpu",
    project="runs/detect",
    name="predict",
    exist_ok=True,
    save=True,  # 保存画框后的图⽚
)
for result in results:
    print("检测到的类别:", result.names)
    print("检测框坐标:", result.boxes.xyxy)
    print("置信度:", result.boxes.conf)
