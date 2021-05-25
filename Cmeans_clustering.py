import cv2
import numpy as np
import math
import sys
import matplotlib.pyplot as plt


def Cmeans_clustering(imPath, clusters):

    # Reading the image
    image = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    shape = image.shape # image shape

    X = image.flatten().astype('float') # flatted image shape: (number of pixels,1) 
    pixels = image.size
    # clusters = 3   

    # Small number
    epsilon = sys.float_info.epsilon 

    # fuzziness_degree is the fuzziness parameter (generally taken as 2)
    fuzziness_degree = 2
        
    delta = 100  
    max_iterations = 100   

    #======================================================================================================================================
    # Initialize W with random values; W is Weights
    W = np.random.rand(pixels,clusters)
    i = 0
    while delta > epsilon:
        # Calculating the CENTROIDS 
        numerator = np.dot(X,pow(W,fuzziness_degree))
        denominator = np.sum(pow(W,fuzziness_degree),axis=0)
        C = numerator/denominator   # New Centroids 

        # Saves the old weight to calculate the delta 
        old_w = np.copy(W) 

        # Calculating the new Weights
        x,y = np.meshgrid(C,X)
        power = 2./(fuzziness_degree-1)
        p1 = pow(abs(y-x),power)
        p2 = np.sum(pow((1./abs(y-x)),power),axis=1)
        W = 1./(p1*p2[:,None])  # New Weights 
        
        # Calculating Delta 
        delta = math.sqrt(np.sum(pow((W - old_w), 2))) 
        # if the iterations exceed the maximum no. of iterations break 
        if delta < epsilon or i > max_iterations:
            break
        i+=1

    #======================================================================================================================================
    # Getting the indices of the maximum values
    result = np.argmax(W, axis = 1)
    result = result.reshape(shape).astype('int')
    plt.imshow(result)
    plt.show()
    return result
