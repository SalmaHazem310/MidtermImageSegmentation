# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(503, 464)
        MainWindow.setMinimumSize(QtCore.QSize(503, 464))
        MainWindow.setMaximumSize(QtCore.QSize(503, 464))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(11, 135, 471, 221))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.horizontalLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.org_img_disp_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.org_img_disp_label.setText("")
        self.org_img_disp_label.setObjectName("org_img_disp_label")
        self.gridLayout_2.addWidget(self.org_img_disp_label, 0, 0, 1, 1)
        self.seg_img_disp_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.seg_img_disp_label.setText("")
        self.seg_img_disp_label.setObjectName("seg_img_disp_label")
        self.gridLayout_2.addWidget(self.seg_img_disp_label, 0, 1, 1, 1)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 360, 471, 51))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox.setFont(font)
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_2.addWidget(self.spinBox)
        self.segment_pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.segment_pushButton.setFont(font)
        self.segment_pushButton.setObjectName("segment_pushButton")
        self.horizontalLayout_2.addWidget(self.segment_pushButton)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 80, 471, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.org_img_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.org_img_label.setFont(font)
        self.org_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.org_img_label.setObjectName("org_img_label")
        self.horizontalLayout_3.addWidget(self.org_img_label)
        self.seg_img_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.seg_img_label.setFont(font)
        self.seg_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.seg_img_label.setObjectName("seg_img_label")
        self.horizontalLayout_3.addWidget(self.seg_img_label)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(11, 11, 471, 71))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.verticalLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.GUI_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.GUI_label.setFont(font)
        self.GUI_label.setAlignment(QtCore.Qt.AlignCenter)
        self.GUI_label.setObjectName("GUI_label")
        self.gridLayout.addWidget(self.GUI_label, 0, 1, 1, 1)
        self.browse_pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.browse_pushButton.setFont(font)
        self.browse_pushButton.setStyleSheet("")
        self.browse_pushButton.setObjectName("browse_pushButton")
        self.gridLayout.addWidget(self.browse_pushButton, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 503, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "C-means Clustring"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Shallow Neural Network"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Dataset Accuracy"))
        self.comboBox.setItemText(3, _translate("MainWindow", "SVM"))
        self.segment_pushButton.setText(_translate("MainWindow", "Start Segmentation"))
        self.org_img_label.setText(_translate("MainWindow", "Original Image"))
        self.seg_img_label.setText(_translate("MainWindow", "Segmented Image"))
        self.GUI_label.setText(_translate("MainWindow", "GUI"))
        self.browse_pushButton.setText(_translate("MainWindow", "Browse"))

