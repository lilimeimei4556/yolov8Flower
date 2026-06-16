import matplotlib
import torch

from ultralytics import YOLO

matplotlib.use("Agg")  # ⽆界⾯后端
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 默认安全字体
# ===================== 核⼼配置（你只需要改这⾥）=====================
DATA_YAML = "flower.yaml"  # 数据集配置⽂件路径
MODEL_TYPE = "yolov8n.pt"  # 模型⼤⼩：n/s/m/l/x（m平衡速度精度）
EPOCHS = 100  # 训练轮数（2000张图推荐100轮）
BATCH_SIZE = 16  # 批次⼤⼩（有显卡直接16/32，爆显存就改8）
IMGSZ = 640  # 训练图⽚尺⼨（标准640）
DEVICE = 0  # 显卡编号（单显卡固定0）
# ==================================================================
# 检查CUDA是否可⽤（⾃动判断显卡）
print("=" * 50)
print(f"CUDA可⽤: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
print("=" * 50)
# 1. 加载预训练模型
model = YOLO(MODEL_TYPE)
# 2. 开始训练（核⼼命令，⾃动保存最优模型）
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMGSZ,
    device=DEVICE,
    patience=15,
    save=True,  # 保存最优模型
    save_period=10,  # 每10轮保存⼀次
    cache=True,  # 缓存数据（加速训练）
    verbose=True,  # 打印详细⽇志
    seed=42,  # 固定随机种⼦
    deterministic=True,
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 末学习率
    warmup_epochs=3,  # 热身轮数
    cos_lr=True,  # 余弦学习率
    flipud=0.0,  # 关闭上下翻转（花朵不需要）
    fliplr=0.5,  # 左右翻转（数据增强）
    mosaic=1.0,  # 开启⻢赛克增强
)
# 3. 训练完成后⾃动验证最优模型
# print("\n" + "="*50)
# print("开始在验证集上评估最优模型")
# model.val()
# 4. 导出模型（可选，导出成ONNX通⽤格式）
# model.export(format="onnx")
