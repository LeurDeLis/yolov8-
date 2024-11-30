"""
@Project : yolo_demo
@File    : detect_pt.py
@IDE     : PyCharm
@Author  : MaFukang
@Date    : 2024/10/18 上午10:13
"""

from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("./yolov8n.pt")

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
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
        # 进行检测
        results = model(frame)
        # 获取检测结果
        res = results[0].boxes.data.to('cpu').numpy()
        # 遍历检测结果
        for x1, y1, x2, y2, conf, cls in res:
            # 检查置信度是否满足阈值
            if conf > 0.7:
                # 提取边界框坐标Q
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                # 获取类别名称
                label = model.names[int(cls)]
                # 绘制边界框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绿色矩形框，线宽为2
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)  # 在框上方显示标签和置信度

        cv2.imshow("Collect", frame)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            break

    # 完成所有操作后，释放捕获器
    cap.release()
    cv2.destroyAllWindows()
