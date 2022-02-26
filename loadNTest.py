# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel
import sys
 
from PyQt5.QtGui import QPixmap
 
# Create the same model as the one we trained
classifier = Sequential()
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3),
                      input_shape = (512, 512, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (6,6)))

# layer 2
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (6,6)))

classifier.add(Flatten())

classifier.add(Dense(units = 252, activation = "relu"))
classifier.add(Dense(units=4, activation='softmax'))

classifier.compile(optimizer = "adam", loss = "CategoricalCrossentropy", metrics = ["accuracy"])

test_datagen = ImageDataGenerator(rescale=1./255,
                                    featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range=0,
                                    zoom_range = 0,
                                    width_shift_range=0,
                                    height_shift_range=0,
                                    horizontal_flip=True,
                                    vertical_flip=False)

##################################################################
#LOADING TEST IMAGES
##################################################################

testing_dataset = test_datagen.flow_from_directory('data/Testing',
                                                   target_size=(512, 512),
                                                   batch_size=64,
                                                   class_mode='categorical')

print(testing_dataset.data_format)

##################################################################
#LOAD THE MODEL
##################################################################

# load weights into new model
classifier.load_weights("model-1-64.h5")

print("Loaded model from disk")

import numpy as np
from tensorflow.keras.preprocessing import image

# evaluate loaded model on test data
# classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss, acc = classifier.evaluate(testing_dataset,verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

##################################################################
#GUI for the model
##################################################################

class Window(QWidget):
    def __init__(self):
        super().__init__()
 
        self.title = "Intelligent Neurosurgical Patient Profiling Model"
        self.top = 200
        self.left = 500
        self.width = 600
        self.height = 800
 
        self.InitWindow()
 
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        vbox = QVBoxLayout()
 
        self.btn1 = QPushButton("pick image")
        self.btn1.clicked.connect(self.getImage)
 
        vbox.addWidget(self.btn1)
 
        self.label1 = QLabel("Welcome to the model")
        self.label2 = QLabel()
        
        vbox.addWidget(self.label1)
        vbox.addWidget(self.label2)
 
        self.setLayout(vbox)
 
        self.show()

    def getImage(self):
        global test_image, classifier
        fname = QFileDialog.getOpenFileName(self, 'pick image', 'data/Testing', "Image files (*.jpg)")
        imagePath = fname[0]
        test_image = image.load_img(imagePath, target_size = (512, 512))
        pixmap = QPixmap(imagePath)
        self.label1.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

        size = (512,512)
        image.smart_resize(test_image, size)    
        test_image = image.img_to_array(test_image) 

        test_image = np.expand_dims(test_image, axis = 0) 
        result = classifier.predict(test_image)
        
        self.plot_chart(result[0])
        plt.show()

        self.label2.setText(str(testing_dataset.class_indices) + "\n" + str(result) + "\n" + str(np.argmax(result[0])))

    def plot_chart(self, predictions):
        plt.grid(False)
        plt.xticks(range(4))
        plt.yticks(range(100))
        thisplot = plt.bar(range(4), predictions, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions)

        thisplot[predicted_label].set_color('red')   
 
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())