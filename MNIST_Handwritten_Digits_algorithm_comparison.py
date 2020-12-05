#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from random import randint


# In[ ]:


def plot_digit(image):
    """
    This function receives an image and plots the digit. 
    """
    nameI = str(randint(1,299))+ ".png"
    plt.imshow(image, cmap='gray')
    plt.show()
    #plt.savefig(nameI)


# In[ ]:


def feed_forward(train_x, train_y, test1_x, test1_y, test2_x, test2_y):

    #Train and evaluate a feedforward network with a single hidden layer.
    
    model = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(512, activation='relu'),
      Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=10)

    print("Evaluating feedforward ")
    model.evaluate(test1_x, test1_y)

    print("Evaluating feedforward on shifted test data")
    model.evaluate(test2_x, test2_y)


# In[ ]:


def cnn(train_x, train_y, test1_x, test1_y, test2_x, test2_y):
    
    #Train and evaluate a feedforward network with hidden layers.
    
    # Add a single "channels" dimension at the end
    # allows for satisfying tensorflow neural networks structure that require the channels dimension
    # to be added even though we are using black and white images so it is unimportant
    
    trn_x = train_x.reshape([-1, 28, 28, 1])
    tst1_x = test1_x.reshape([-1, 28, 28, 1])
    tst2_x = test2_x.reshape([-1, 28, 28, 1])

    # First layer will need argument `input_shape=(28,28,1)`
    model = Sequential([
        Conv2D(32,(5,5)),
        MaxPooling2D((2,2),(2,2)),
        Conv2D(64,(5,5)),
        MaxPooling2D((2,2),(1,1)),
        Flatten(input_shape=(28,28,1)),
        Dense(512,activation='relu'),
        Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(trn_x, train_y, epochs=10)

    print("Evaluating CNN ")
    model.evaluate(tst1_x, test1_y)
    print("Evaluating CNN on shifted test data")
    model.evaluate(tst2_x, test2_y)


# In[ ]:


def decenter(X,pad=2):
    out = np.roll(X, 2, axis=1)
    out = np.roll(X, 2, axis=2)
    return out


# In[ ]:


def h_theta(X, W, B):
    """
    For a given matrix W and vector B, this function predicts the value
    of each digit for each image of X. Here we assume that each column of X
    is a flattened image. 
    """
    return W.dot(X) + B 


# In[ ]:


def l5GradDesc(images, labels, images_test, test_labels, images_test2, test_labels2):
    images = images.reshape(images.shape[0],28*28)
    images = images.T
    
    images_test = images_test.reshape(images_test.shape[0], 28*28)
    images_test = images_test.T
    
    images_test2 = images_test2.reshape(images_test2.shape[0], 28*28)
    images_test2 = images_test2.T
    # Learning rate alpha, for controlling the step of gradient descent
    alpha = 0.01

    # Number of instances in the training set
    m = images.shape[1]

    # Matrix W initialized with zeros
    W = np.zeros((10, 28*28))

    # Matrix B also initialized with zeros
    B = np.zeros((10,1))

    # Creating Y matrix where each column is an one-hot vector
    Y = np.zeros((10, m))
    for index, value in enumerate(labels):
        Y[value][index] = 1

    # Performs 1000 iterations of gradient descent
    # print("start W shape is ", W.shape)
    # print("start B shape is ", B.shape)
    for i in range(1000):
        # Write here your implementation of gradient descent

        temp = ((W.dot(images) + B - Y)).dot(np.transpose(images))
        w_pderivativ = (1/m) * temp
        b_pderivativ = (1/m) * (W.dot(images) + B - Y)
        b_pderivative = []
        for row in b_pderivativ:
            n_row= sum(row)
            b_pderivative.append(n_row)
        b_pderivative = np.array(b_pderivative).reshape(10,1)

        W = W - (alpha * w_pderivativ)
        B = B - (alpha * b_pderivative)
        
    #make predictions for first test set
    Y_hat = W.dot(images_test) + B

    #one hot encode the results
    results = []
    for row in np.transpose(Y_hat):
        results.append(np.argmax(row))

    #get accuracy of the data based on the training results
    accurate = 0
    for i in range(len(results)):
        if results[i] == test_labels[i]:
            accurate += 1
    print("percentage accuracy is: ", accurate/len(results))
    
    
    Y_hat2 = W.dot(images_test2) + B

    #one hot encode the results
    results = []
    for row in np.transpose(Y_hat2):
        results.append(np.argmax(row))

    #get accuracy of the data based on the training results
    accurate = 0
    for i in range(len(results)):
        if results[i] == test_labels2[i]:
            accurate += 1
    print("percentage accuracy of shifted test data is: ", accurate/len(results))


# In[ ]:


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #change x_train and x_test data to float from uint8
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    #use one hot encoding to represent the answers ie labels for the data
    #both for training and testing
    with tf.compat.v1.Session():
        y_tran = tf.one_hot(y_train, 10).eval()
        y_tst = tf.one_hot(y_test, 10).eval()
    
    
    #shift the test data by 2 pixels to the bottom right
    x_test2 = decenter(x_test, 2)
    
    num  = randint(256,456)
    plot_digit(x_test[num])
    plot_digit(x_test2[num])
    print('Label: ', y_test[num])
    
    
    num  = randint(256,456)
    plot_digit(x_test[num])
    plot_digit(x_test2[num])
    print('Label: ', y_test[num])
    #call classification algorithms    
    
#     Evaluate the 1 layer feed forward neural network model
    feed_forward(x_train, y_tran, x_test, y_tst, x_test2, y_tst)
#     Evaluate the convolutional neural network model
    cnn(x_train, y_tran, x_test, y_tst, x_test2, y_tst)
#     Evaluate the lab 5 code
    l5GradDesc(x_train, y_train, x_test, y_test, x_test2, y_test)
    


# In[ ]:


main()


# In[ ]:




