# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\PycharmProjects\SpeechEnhancementApp\ui\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(1166, 563)
        MainWindow.setMinimumSize(QtCore.QSize(993, 429))
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1080))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(490, 10, 661, 401))
        self.groupBox_3.setObjectName("groupBox_3")
        self.textBrowser_log = QtWidgets.QTextBrowser(self.groupBox_3)
        self.textBrowser_log.setGeometry(QtCore.QRect(19, 53, 631, 331))
        self.textBrowser_log.setObjectName("textBrowser_log")
        self.pushButton_clearlog = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_clearlog.setGeometry(QtCore.QRect(590, 20, 51, 31))
        self.pushButton_clearlog.setObjectName("pushButton_clearlog")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 280, 200, 130))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.radioButton_cpu = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_cpu.setObjectName("radioButton_cpu")
        self.radioButton_cpu.setChecked(True)
        self.gridLayout.addWidget(self.radioButton_cpu, 0, 0, 1, 1)
        self.radioButton_gpu = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_gpu.setObjectName("radioButton_gpu")
        self.gridLayout.addWidget(self.radioButton_gpu, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(269, 280, 211, 130))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.radioButton_tasnet = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_tasnet.setObjectName("radioButton_tasnet")
        self.gridLayout_3.addWidget(self.radioButton_tasnet, 1, 0, 1, 1)
        self.radioButton_segan = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_segan.setObjectName("radioButton_segan")
        self.radioButton_segan.setChecked(True)
        self.gridLayout_3.addWidget(self.radioButton_segan, 0, 0, 1, 1)
        self.pushButton_forward = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_forward.setGeometry(QtCore.QRect(640, 430, 151, 51))
        self.pushButton_forward.setObjectName("pushButton_forward")
        self.pushButton_clearall = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_clearall.setGeometry(QtCore.QRect(420, 430, 151, 51))
        self.pushButton_clearall.setObjectName("pushButton_clearall")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 10, 461, 261))
        self.groupBox_5.setObjectName("groupBox_5")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(10, 30, 71, 24))
        self.label.setObjectName("label")
        self.label_cpu = QtWidgets.QLabel(self.groupBox_5)
        self.label_cpu.setGeometry(QtCore.QRect(130, 30, 321, 24))
        self.label_cpu.setObjectName("label_cpu")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setGeometry(QtCore.QRect(10, 70, 71, 24))
        self.label_2.setObjectName("label_2")
        self.label_gpu = QtWidgets.QLabel(self.groupBox_5)
        self.label_gpu.setGeometry(QtCore.QRect(130, 70, 321, 24))
        self.label_gpu.setObjectName("label_gpu")
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setGeometry(QtCore.QRect(10, 110, 81, 24))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 108, 24))
        self.label_4.setObjectName("label_4")
        self.label_cuda = QtWidgets.QLabel(self.groupBox_5)
        self.label_cuda.setGeometry(QtCore.QRect(130, 110, 321, 24))
        self.label_cuda.setObjectName("label_cuda")
        self.label_pytorch = QtWidgets.QLabel(self.groupBox_5)
        self.label_pytorch.setGeometry(QtCore.QRect(130, 150, 321, 24))
        self.label_pytorch.setObjectName("label_pytorch")
        self.pushButton_hardware = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_hardware.setGeometry(QtCore.QRect(40, 190, 381, 31))
        self.pushButton_hardware.setObjectName("pushButton_hardware")
        self.toolButton_readfiles = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_readfiles.setGeometry(QtCore.QRect(190, 430, 151, 51))
        self.toolButton_readfiles.setObjectName("toolButton_readfiles")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1166, 37))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menu.addAction(self.actionAbout)
        self.menu.addAction(self.actionExit)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SpeechEnhancementDemo"))
        self.groupBox_3.setTitle(_translate("MainWindow", "日志"))
        self.textBrowser_log.setStatusTip(_translate("MainWindow", "日志显示区"))
        self.pushButton_clearlog.setText(_translate("MainWindow", "清空"))
        self.groupBox.setTitle(_translate("MainWindow", "选择设备"))
        self.radioButton_cpu.setStatusTip(_translate("MainWindow", "使用CPU进行推理(速度慢)"))
        self.radioButton_cpu.setText(_translate("MainWindow", "CPU"))
        self.radioButton_gpu.setStatusTip(_translate("MainWindow", "使用GPU进行推理(速度快)"))
        self.radioButton_gpu.setText(_translate("MainWindow", "GPU"))
        self.groupBox_2.setTitle(_translate("MainWindow", "选择模型"))
        self.radioButton_tasnet.setStatusTip(_translate("MainWindow", "TASNET模型(处理速度慢)"))
        self.radioButton_tasnet.setText(_translate("MainWindow", "TASNET"))
        self.radioButton_segan.setStatusTip(_translate("MainWindow", "SEGAN模型(处理速度快)"))
        self.radioButton_segan.setText(_translate("MainWindow", "SEGAN"))
        self.pushButton_forward.setStatusTip(_translate("MainWindow", "开始推理"))
        self.pushButton_forward.setText(_translate("MainWindow", "开始处理"))
        self.pushButton_clearall.setStatusTip(_translate("MainWindow", "清空当前处理队列"))
        self.pushButton_clearall.setText(_translate("MainWindow", "清空队列"))
        self.groupBox_5.setTitle(_translate("MainWindow", "系统信息"))
        self.label.setText(_translate("MainWindow", "CPU型号:"))
        self.label_cpu.setText(_translate("MainWindow", "..."))
        self.label_2.setText(_translate("MainWindow", "GPU型号:"))
        self.label_gpu.setText(_translate("MainWindow", "..."))
        self.label_3.setText(_translate("MainWindow", "CUDA版本:"))
        self.label_4.setText(_translate("MainWindow", "Pytorch版本:"))
        self.label_cuda.setText(_translate("MainWindow", "..."))
        self.label_pytorch.setText(_translate("MainWindow", "..."))
        self.pushButton_hardware.setText(_translate("MainWindow", "检测环境"))
        self.toolButton_readfiles.setStatusTip(_translate("MainWindow", "将文件加入处理队列"))
        self.toolButton_readfiles.setText(_translate("MainWindow", "读取"))
        self.menu.setTitle(_translate("MainWindow", "菜单"))
        self.actionAbout.setText(_translate("MainWindow", "关于"))
        self.actionAbout.setStatusTip(_translate("MainWindow", "By Nitin 2021.06"))
        self.actionExit.setText(_translate("MainWindow", "退出"))
        self.actionExit.setStatusTip(_translate("MainWindow", "退出程序"))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())