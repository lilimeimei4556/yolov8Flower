from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/detect/coco8_train/weights/best.pt")

# 测试：在 coco8 测试集上做最终评估
test_metrics = model.val(
    data="coco8.yaml",
    imgsz=640,
    batch=2,
    device="cpu",
    split="test"    # 关键：使用测试集！
)

# 输出最终测试结果
print("===== 模型最终测试结果 =====")
print(f"mAP50: {test_metrics.box.map50:.3f}")
print(f"mAP50-95: {test_metrics.box.map:.3f}")
print(f"精确率: {test_metrics.box.p:.3f}")
print(f"召回率: {test_metrics.box.r:.3f}")