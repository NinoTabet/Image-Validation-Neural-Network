from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image
import numpy as np
import cv2
import os

testing_yes_path = ("C:\\Users\\Nino\\Desktop\\neural network photos\\Testing\\download.jpg")
testing_path = ("C:\\Users\\Nino\\Desktop\\neural network photos\\Testing")
train_path = ("C:\\Users\\Nino\\Desktop\\neural network photos\\Training")
validation_path = ("C:\\Users\\Nino\\Desktop\\neural network photos\\Validation")

img = image.load_img(testing_yes_path)
plt.imshow(img)

cv2.imread(testing_yes_path)


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory(train_path,
                                          target_size= (200,200),
                                          batch_size= 3,
                                          class_mode= 'binary')

validation_dataset = train.flow_from_directory(validation_path,
                                          target_size= (200,200),
                                          batch_size= 3,
                                          class_mode= 'binary')

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     ##
                                     tf.keras.layers.Flatten(),
                                     ##
                                     tf.keras.layers.Dense(512,activation = 'relu'),
                                     ##
                                     tf.keras.layers.Dense(1,activation = 'sigmoid')
                                     ])

model.compile(loss= 'binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch= 3,
                      epochs = 20,
                      validation_data = validation_dataset)

validation_dataset.class_indices

dir_path = testing_path

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '\\' + i , target_size = (200,200))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    val = model.predict(images)
    print (val)
    if val == 0:
        print("Crying")
    else:
        print("Smiling")
