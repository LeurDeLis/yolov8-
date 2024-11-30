"""
@Project ：yolo_demo
@File    ：detect_video_demo.py
@IDE     ：PyCharm
@Author  ：MFK
@Date    ：2024/8/6 上午9:57
"""

import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")  # 请确保路径正确


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 进行人体姿态检测
        results = model(frame)
        frame = results[0].plot()

        cv2.imshow("Person-Pose", frame)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            break

        # 完成所有操作后，释放捕获器
    cap.release()
    cv2.destroyAllWindows()
