import numpy as np
import cv2
import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self):

        self.weights = np.random.uniform(-1, 1, size = 1)

    def segmoid(self, x):
        return 1 / (1+ np.exp(-x))

    def segmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for i in range(training_iterations):

            output = self.predict(training_inputs)
            
            error = training_outputs.T[0] - output

            updates = np.dot(training_inputs.T, error* self.segmoid_derivative(output))
            
            self.weights += updates

    def predict(self, inputs):

        output = self.segmoid(np.dot(inputs, self.weights))

        return output


def get_results(test_image, nn):

    # Reading the three objects
    object_1_image = cv2.imread('object_1.bmp')
    object_2_image = cv2.imread('object_2.bmp')
    object_3_image = cv2.imread('object_3.bmp')

    # Getting sizes of object images
    H1 = object_1_image.shape[0]
    H2 = object_2_image.shape[0]
    H3 = object_3_image.shape[0]

    W1 = object_1_image.shape[1]
    W2 = object_2_image.shape[1]
    W3 = object_3_image.shape[1]

    # Get new image sizes
    height = max(H1, H2, H3)
    width = max(W1, W2, W3)

    dsize = (width, height)

    # Resize all images with the new size
    object_1_image = cv2.resize(object_1_image, dsize)
    object_2_image = cv2.resize(object_2_image, dsize)
    object_3_image = cv2.resize(object_3_image, dsize)

    # Getting each channel mean in all objects
    r_channel_mean_1 = np.mean(object_1_image[:, :, 2])
    g_channel_mean_1 = np.mean(object_1_image[:, :, 1])
    b_channel_mean_1 = np.mean(object_1_image[:, :, 0])

    r_channel_mean_2 = np.mean(object_2_image[:, :, 2])
    g_channel_mean_2 = np.mean(object_2_image[:, :, 1])
    b_channel_mean_2 = np.mean(object_2_image[:, :, 0])

    r_channel_mean_3 = np.mean(object_3_image[:, :, 2])
    g_channel_mean_3 = np.mean(object_3_image[:, :, 1])
    b_channel_mean_3 = np.mean(object_3_image[:, :, 0])

    final_image = np.zeros((test_image.shape[0], test_image.shape[1], test_image.shape[2]))

    for i in range(test_image.shape[0]):
        for j in range(test_image.shape[1]):

            current_pixel_mean = np.mean([test_image[i][j][2], test_image[i][j][1], test_image[i][j][0]])
            predicted_label = nn.predict([current_pixel_mean])

            if int(predicted_label) == 0: 
                final_image[i][j][0] = b_channel_mean_1/255
                final_image[i][j][1] = g_channel_mean_1/255
                final_image[i][j][2] = r_channel_mean_1/255
            
            elif predicted_label == 0.5: 
                final_image[i][j][0] = b_channel_mean_2/255
                final_image[i][j][1] = g_channel_mean_2/255
                final_image[i][j][2] = r_channel_mean_2/255

            elif int(predicted_label) == 1:
                final_image[i][j][0] = b_channel_mean_3/255
                final_image[i][j][1] = g_channel_mean_3/255
                final_image[i][j][2] = r_channel_mean_3/255

    return final_image

