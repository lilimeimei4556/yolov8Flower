import os

img_dir = r"datasets/flower/images/train"
lab_dir = r"datasets/flower/labels/train"

# 图片名（不带后缀）
imgs = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(("jpg", "jpeg", "png"))]

# 标注名（不带后缀）
labs = [os.path.splitext(f)[0] for f in os.listdir(lab_dir) if f.endswith("txt")]

imgs = set(imgs)
labs = set(labs)

print("总图片数：", len(imgs))
print("已标注数：", len(labs))
print("未标注（有图无标）：", sorted(imgs - labs))
print("多余标注（无图有标）：", sorted(labs - imgs))
