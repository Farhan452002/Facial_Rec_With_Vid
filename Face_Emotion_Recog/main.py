# Import Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

Train_dataPath = 'C:/Users/farha/Downloads/archive/train'

# Display a test image
train_image = cv2.imread(Train_dataPath+'/angry/Training_3908.jpg')
plt.imshow(train_image)
plt.show()

# Find the Image Size
print(f'Image Size: {train_image.shape}')

# Create a list (array) of classes (facial emotions)
Classes=['angry','disgust','fear','happy','sad','surprise','neutral']

# Display the first image in each category of the Train dataset
for category in Classes:
    path = os.path.join(Train_dataPath,category)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path,img))
        plt.imshow(image)
        plt.show()
        break
    break