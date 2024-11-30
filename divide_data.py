"""
@Project ：demo2.py
@File    ：divide_data.py
@IDE     ：PyCharm
@Author  ：MFK
@Date    ：2024/8/13 下午3:34
"""

import os
import random
import shutil

# 需要分割的图片目录
SOURCE_IMG_DIR = './data'

# 识别分割图片类型"jpg", "png"
file_type = False

# # 定义数据集目录和分割比例
source_root = SOURCE_IMG_DIR
target_root = 'data_01'
train_ratio = 0.8
valid_ratio = 0.2

# 创建目标文件夹及其子文件夹
train_dir = os.path.join(target_root, "train")
valid_dir = os.path.join(target_root, "valid")

for phase in ['train', 'valid']:
    os.makedirs(os.path.join(target_root, phase, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_root, phase, 'labels'), exist_ok=True)

# 获取所有文件列表
files = os.listdir(source_root)

if files[0].split('.')[1] == "jpg" or files[1].split('.')[1] == "jpg":
    png_files = [f for f in files if f.endswith(".jpg")]
    file_type = True
    print(len(png_files))
else:
    png_files = [f for f in files if f.endswith(".png")]
    print(len(png_files))

# 随机打乱文件列表
random.shuffle(png_files)

# 计算分割点
num_files = len(png_files)
num_train = int(train_ratio * num_files)

# 移动文件到目标位置
# 将文件复制到相应目录
for i, file in enumerate(png_files):
    file = file.split('/')[-1].split('.')[-2]
    if file_type:
        image_path = os.path.join(source_root, file + '.jpg')
    else:
        image_path = os.path.join(source_root, file + '.png')
    label_path = os.path.join(source_root, file + '.txt')
    print(image_path)
    print(label_path)
    if i < num_train:
        dst_dir = train_dir
    else:
        dst_dir = valid_dir
    # 复制到目标文件夹
    shutil.copy(image_path, os.path.join(dst_dir, 'images'))
    shutil.copy(label_path, os.path.join(dst_dir, 'labels'))
