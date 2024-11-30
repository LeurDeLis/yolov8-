# -*- coding: UTF-8 -*-
"""
@Project ：yolov8 
@File    ：detect_img_demo.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2024/11/28 下午7:46 
"""

from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
frame = cv2.imread("C:\\Users\\Lenovo\\Desktop\\yolov8\\data_01\\train\\images\\1.jpg")

results = model(frame)
# print(results)
# print("***********************")
# print(results[0].boxes)
# print("***********************")
# print(results[0].boxes.data)
# print("***********************")
# 获取检测结果
res = results[0].boxes.data.to('cpu').numpy()

# 遍历检测结果
for x1, y1, x2, y2, conf, cls in res:
    # 检查置信度是否满足阈值
    # if conf > 0.7:
    # 提取边界框坐标Q
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    # 获取类别名称
    label = model.names[int(cls)]
    # 绘制边界框和标签
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绿色矩形框，线宽为2
    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                2)  # 在框上方显示标签和置信度
    print("------")
    cv2.imwrite("output.jpg", frame)
