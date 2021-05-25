from PyQt5 import QtWidgets, QtGui, QtWidgets , uic
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QMessageBox
from PyQt5.QtGui import QPixmap
from gui import Ui_MainWindow
from support_vector_machine import SupportVectorMachine
from PIL import Image
from Cmeans_clustering import Cmeans_clustering
from dataset_accuracy import shallow_neural_network
from shallow_neural_network_image import NeuralNetwork, get_results
import PIL.ImageQt
from scipy import misc
import numpy as np
import os
import cv2
import sys


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):

        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.svm = SupportVectorMachine()
        self.nn_model = NeuralNetwork()


        self.test_image = []
        self.final_image = []
        self.imgPath = 0

        self.ui.browse_pushButton.clicked.connect(lambda: self.browse(self.ui.org_img_disp_label))
        self.ui.segment_pushButton.clicked.connect(self.segmentation_method)


    def browse(self, Ui_MainWindow):

        filename = QtWidgets.QFileDialog.getOpenFileNames( directory = os.path.dirname(__file__) ,filter= '*.bmp')
        if (filename == ([], '')) | (filename ==  0 ):
            return
        self.imgPath = str(filename[0][0])

        self.test_image = cv2.imread(self.imgPath)
        self.final_image = np.zeros((self.test_image.shape[0], self.test_image.shape[1], self.test_image.shape[2]))

        self.showImage(self.test_image, Ui_MainWindow)


    def showImage(self, arr, Ui_MainWindow):
        """
        Convert an array to image and display it given image handler
        """

        imgBack = Image.fromarray(arr)
        qimage = PIL.ImageQt.ImageQt(imgBack)
        pixmap = QtGui.QPixmap.fromImage(qimage) 
        Ui_MainWindow.setPixmap(pixmap)
        Ui_MainWindow.setMask(pixmap.mask())
        Ui_MainWindow.show()


    def SVM_segmentation(self, curr_svm, Ui_MainWindow):
        """
        Image segmentation with SVM method
        """
        self.final_image = np.zeros((self.test_image.shape[0], self.test_image.shape[1], self.test_image.shape[2]))

        curr_svm.read_resize_objects()

        mean_object_1 = curr_svm.get_mean_of_image(curr_svm.object_1_image)
        mean_object_2 = curr_svm.get_mean_of_image(curr_svm.object_2_image)
        mean_object_3 = curr_svm.get_mean_of_image(curr_svm.object_3_image)

        self.final_image = curr_svm.my_svm_segmentation(mean_object_1, mean_object_2,\
             mean_object_3, self.test_image, self.final_image)

        self.showImage((self.final_image * 255).astype(np.uint8), Ui_MainWindow)


    def neural_network(self, curr_svm, curr_nn, Ui_MainWindow):
        """
        Image segmentation with neural network method
        """
        curr_svm.read_resize_objects()

        mean_object_1 = curr_svm.get_mean_of_image(curr_svm.object_1_image)
        mean_object_2 = curr_svm.get_mean_of_image(curr_svm.object_2_image)
        mean_object_3 = curr_svm.get_mean_of_image(curr_svm.object_3_image)

        training_inputs = np.array([[mean_object_1], [mean_object_2], [mean_object_3]])
        training_outputs = np.array([1, 2, 3]).T

        curr_nn.train(training_inputs, training_outputs, 10000)

        self.final_image = get_results(self.test_image, curr_nn)
    
        self.showImage((self.final_image * 255).astype(np.uint8), Ui_MainWindow)


    def segmentation_method(self):

        if self.ui.comboBox.currentText() == "C-means Clustring":

            if(self.ui.spinBox.value() != 0):

                output = Cmeans_clustering(self.imgPath, self.ui.spinBox.value())
                self.showImage((output* 255).astype(np.uint8), self.ui.seg_img_disp_label)
            else:
                return

        elif self.ui.comboBox.currentText() == "Shallow Neural Network":
            
            self.neural_network(self.svm, self.nn_model, self.ui.seg_img_disp_label)

        elif self.ui.comboBox.currentText() == "Dataset Accuracy":

            accuracy_train, accuracy_test = shallow_neural_network()
            QMessageBox.about(self, "Dataset Accuracy", 'Accuracy Train: ' + str(accuracy_train) + \
                ', Accuracy Test: ' + str(accuracy_test))

        elif self.ui.comboBox.currentText() == "SVM":
            self.SVM_segmentation(self.svm, self.ui.seg_img_disp_label)
            

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()
