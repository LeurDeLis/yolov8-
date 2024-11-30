"""
@Project ：yolo_demo
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2024/8/11 下午4:22 
"""

from ultralytics import YOLO

# 加载模型
model = YOLO("./yolov8n.pt")

if __name__ == '__main__':
    # 训练模型
    model.train(data="./data_01/config.yaml", epochs=10)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
