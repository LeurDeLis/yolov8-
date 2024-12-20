# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'collect_img_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(730, 480)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.label_img.setStyleSheet("background-color: rgba(255, 255, 255);")
        self.label_img.setText("")
        self.label_img.setObjectName("label_img")
        self.pushButton_jpg = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_jpg.setGeometry(QtCore.QRect(655, 40, 60, 180))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.pushButton_jpg.setFont(font)
        self.pushButton_jpg.setStyleSheet("QPushButton\n"
                                          "{\n"
                                          "    background-color: rgba(232, 241, 248);\n"
                                          "    border-radius: 10px;\n"
                                          "}\n"
                                          "QPushButton:hover\n"
                                          "{\n"
                                          "    background-color: rgba(64, 70, 104, 100);\n"
                                          "    border-radius: 10px;\n"
                                          "}")
        self.pushButton_jpg.setObjectName("pushButton_jpg")
        self.pushButton_png = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_png.setGeometry(QtCore.QRect(655, 260, 60, 180))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.pushButton_png.setFont(font)
        self.pushButton_png.setStyleSheet("QPushButton\n"
                                          "{\n"
                                          "    background-color: rgba(232, 241, 248);\n"
                                          "    border-radius: 10px;\n"
                                          "}\n"
                                          "QPushButton:hover\n"
                                          "{\n"
                                          "    background-color: rgba(64, 70, 104, 100);\n"
                                          "    border-radius: 10px;\n"
                                          "}")
        self.pushButton_png.setObjectName("pushButton_png")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.pushButton_jpg.clicked.connect(MainWindow.save_jpg_img)  # type: ignore
        self.pushButton_png.clicked.connect(MainWindow.save_png_img)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_jpg.setText(_translate("MainWindow", "保\n"
                                                             "存\n"
                                                             "img\n"
                                                             "图\n"
                                                             "片"))
        self.pushButton_png.setText(_translate("MainWindow", "保\n"
                                                             "存\n"
                                                             "png\n"
                                                             "图\n"
                                                             "片"))
