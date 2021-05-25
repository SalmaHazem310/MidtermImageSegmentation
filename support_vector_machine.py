import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SupportVectorMachine:
    def __init__(self):
        super(SupportVectorMachine, self).__init__()

        self.object_1_image = []
        self.object_2_image = []
        self.object_3_image = []


    def read_resize_objects(self):
        """
        Read the different objects (training data) after cropping them 
        in individual images, and resize all of them to be of the same size.
        """
        
        # reading the three objects
        self.object_1_image = cv2.imread('object_1.bmp')
        self.object_2_image = cv2.imread('object_2.bmp')
        self.object_3_image = cv2.imread('object_3.bmp')

        # getting sizes of object images
        H1 = self.object_1_image.shape[0]
        H2 = self.object_2_image.shape[0]
        H3 = self.object_3_image.shape[0]

        W1 = self.object_1_image.shape[1]
        W2 = self.object_2_image.shape[1]
        W3 = self.object_3_image.shape[1]

        # get new image sizes
        height = max(H1, H2, H3)
        width = max(W1, W2, W3)

        dsize = (width, height)

        # resize all images with the new size
        self.object_1_image = cv2.resize(self.object_1_image, dsize)
        self.object_2_image = cv2.resize(self.object_2_image, dsize)
        self.object_3_image = cv2.resize(self.object_3_image, dsize)


    def object_channels_means(self, object):
        """
        Get each channel mean of an object.
        """

        r_channel_mean = np.mean(object[:, :, 2])
        g_channel_mean = np.mean(object[:, :, 1])
        b_channel_mean = np.mean(object[:, :, 0])

        return r_channel_mean, g_channel_mean, b_channel_mean

    def get_mean_of_image(self, img):
        """
        Get mean of each pixel, then get mean of the whole image.
        """

        one_channel_image = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                # get mean of each pixel
                current_pixel_mean = np.mean([img[i][j][2], img[i][j][1], img[i][j][0]])
                # append it to an array
                one_channel_image.append(current_pixel_mean)
        
        # get mean of whole array
        one_channel_image = np.array(one_channel_image)

        return np.mean(one_channel_image)


    def my_svm_segmentation(self, mean_object_1, mean_object_2, mean_object_3, image_test, image_return):
        """
        Train objects mean and classify given data
        """

        # features vector
        X = np.array([[mean_object_1], [mean_object_2], [mean_object_3]])

        # labels vactor
        Y = np.array([1, 2, 3])

        # make the classification
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)

        # read test image
        test_image = image_test

        # identify final image
        final_image = image_return

        # get channel means of each object
        r_channel_mean_1, g_channel_mean_1, b_channel_mean_1 = self.object_channels_means(self.object_1_image)
        r_channel_mean_2, g_channel_mean_2, b_channel_mean_2 = self.object_channels_means(self.object_2_image)
        r_channel_mean_3, g_channel_mean_3, b_channel_mean_3 = self.object_channels_means(self.object_3_image)

        for i in range(test_image.shape[0]):
            for j in range(test_image.shape[1]):

                # get mean of each pixel
                current_pixel_mean = np.mean([test_image[i][j][2], test_image[i][j][1], test_image[i][j][0]])

                # predict its label
                predicted_label = clf.predict([[current_pixel_mean]])

                # visualise it
                if predicted_label[0] == 1: 
                    final_image[i][j][0] = b_channel_mean_1/255
                    final_image[i][j][1] = g_channel_mean_1/255
                    final_image[i][j][2] = r_channel_mean_1/255
                
                elif predicted_label[0] == 2: 
                    final_image[i][j][0] = b_channel_mean_2/255
                    final_image[i][j][1] = g_channel_mean_2/255
                    final_image[i][j][2] = r_channel_mean_2/255

                elif predicted_label[0] == 3:
                    final_image[i][j][0] = b_channel_mean_3/255
                    final_image[i][j][1] = g_channel_mean_3/255
                    final_image[i][j][2] = r_channel_mean_3/255

        return final_image


