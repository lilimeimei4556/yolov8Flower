from ultralytics import YOLO
# 加载训练好的最佳权重
model = YOLO("runs/detect/train-2/weights/best.pt")
# 验证
metrics = model.val(
 data="coco8.yaml",
 imgsz=640,
 batch=2,
 device="cpu",
 project="runs/detect",
 name="val",
 exist_ok=True
)
# 打印核⼼指标
print("验证集精确率 mAP50:", metrics.box.map50)
print("验证集平均精确率 mAP50-95:", metrics.box.map)