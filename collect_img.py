# -*- coding: UTF-8 -*-
"""
@Project ：yolov8 
@File    ：collect_img.py
@IDE     ：PyCharm 
@Author  ：MFK
@Date    ：2024/11/28 下午5:10
"""

import os
import sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from collect_img_ui import Ui_MainWindow

# 设置保存图片的目录
SAVE_PATH = "./mydata/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# 自定义 QT 线程类
class MyThread(QThread):
    def __init__(self, function, parent=None):
        super(MyThread, self).__init__(parent)
        self.function = function
        self.start()

    def run(self):
        self.function()


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.frame = None
        self.save_jpg_img_flag = False
        self.save_png_img_flag = False
        self.count = 0

        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.showimg_thread = MyThread(self.show_video)

    def save_jpg_img(self):
        self.count += 1
        self.save_jpg_img_flag = True

    def save_png_img(self):
        self.count += 1
        self.save_png_img_flag = True

    def show_video(self):
        while True:
            # 逐帧捕获
            ret, self.frame = self.cap.read()
            # 如果正确读取帧，ret为True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # 将捕获到的帧 BGR --> RGB --> QImage格式
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = QImage(self.frame.data.tobytes(), self.frame.shape[1], self.frame.shape[0],
                                QImage.Format_RGB888)
            if self.save_jpg_img_flag:
                self.save_jpg_img_flag = False
                self.frame.save(SAVE_PATH + f"{self.count:03}.jpg")
            elif self.save_png_img_flag:
                self.save_png_img_flag = False
                self.frame.save(SAVE_PATH + f"{self.count:03}.png")
            # 显示在label标签上
            self.label_img.setPixmap(QPixmap.fromImage(self.frame))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())
