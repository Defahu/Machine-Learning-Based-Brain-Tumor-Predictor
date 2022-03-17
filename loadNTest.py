# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from tensorflow.keras.preprocessing import image
import numpy as np  # linear algebra
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QStackedWidget, \
    QDialog, QMessageBox
from PyQt5.uic import loadUi
import sys

from PyQt5.QtGui import QPixmap

# Create the same model as the one we trained

classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      input_shape=(512, 512, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(6, 6)))

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(6, 6)))

classifier.add(Flatten())

classifier.add(Dense(units=252, activation="relu"))
classifier.add(Dense(units=4, activation='softmax'))

classifier.compile(
    optimizer="adam", loss="CategoricalCrossentropy", metrics=["accuracy"])

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    zoom_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True,
    vertical_flip=False)

##################################################################
# LOADING TEST IMAGES
##################################################################

testing_dataset = test_datagen.flow_from_directory('data/Testing',
                                                   target_size=(512, 512),
                                                   batch_size=64,
                                                   class_mode='categorical')

print(testing_dataset.data_format)

##################################################################
# LOAD THE MODEL
##################################################################

# load weights into new model
classifier.load_weights("model-50times.h5")

print("Loaded model from disk")

# evaluate loaded model on test data
classifier.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
loss, acc = classifier.evaluate(testing_dataset, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


##################################################################
# GUI for the model
##################################################################


class HomePage(QMainWindow):
    def __init__(self):
        super(HomePage, self).__init__()
        loadUi("ui/homePage.ui", self)
        self.loginButton.clicked.connect(self.gotologin)
        self.signupButton.clicked.connect(self.gotosignup)

    def gotologin(self):
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotosignup(self):
        signup = Signup()
        widget.addWidget(signup)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotohelp(self):
        help = Help()
        widget.addWidget(help)
        widget.setCurrentIndex(widget.currentIndex() + 1)  
        
class MainFunction(QDialog):
    def __init__(self):
        super(MainFunction, self).__init__()
        loadUi("ui/mainFunction.ui", self)
        self.loadButton.clicked.connect(self.getimage)
        self.analyzeButton.clicked.connect(self.analyze)

    def getimage(self):
        global test_image, classifier
        self.textLabel.setText('')
        fname = QFileDialog.getOpenFileName(
            self, 'pick image', 'data/Testing', "Image files (*.jpg)")
        imagePath = fname[0]
        test_image = image.load_img(imagePath, target_size=(512, 512))
        pixmap = QPixmap(imagePath)
        self.imageLabel.setPixmap(QPixmap(pixmap))

        # self.resize(pixmap.width(), pixmap.height())

        size = (512, 512)
        image.smart_resize(test_image, size)
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

    def analyze(self):
        global test_image, classifier
        print("before")
        # if test_image:
        print("123")
        # if test_image:
        result = classifier.predict(test_image)
        self.plot_chart(result[0])
        plt.show()
        self.textLabel.setText(str(testing_dataset.class_indices) +
                               "\n" + str(result) + "\n" + str(np.argmax(result[0])))

    def plot_chart(self, predictions):
        plt.grid(False)
        label = ['glioma', 'meningioma', 'notumor', 'pituitary']
        plt.xticks(range(4))
        plt.yticks(range(100))
        thisplot = plt.bar(label, predictions, color="#777777")
        plt.ylabel('Probabilities')
        plt.ylim([0, 1.0])
        predicted_label = np.argmax(predictions)

        thisplot[predicted_label].set_color('red')


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi("ui/login.ui", self)
        self.homeButton.clicked.connect(self.backtohome)
        self.pushButton.clicked.connect(self.confirmlogin)

    def backtohome(self):
        widget.addWidget(homePage)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def confirmlogin(self):
        username = self.usernameLine.text()
        password = self.passwordLine.text()
        account = username + " " + password + "\n"
        l = list()
        if len(username) < 1 or len(password) < 1:
            return 0
        with open("Account/accountInfo.txt") as file:
            for line in file.readlines():
                print(line)
                l.append(line)

        for info in l:
            if account == info:
                main = MainFunction()
                widget.addWidget(main)
                file.close()
                widget.setCurrentIndex(widget.currentIndex() + 1)


class Signup(QDialog):
    def __init__(self):
        super(Signup, self).__init__()
        loadUi("ui/signup.ui", self)
        self.pushButton.clicked.connect(self.confirmsignup)
        self.homeButton.clicked.connect(self.backtohome)

    def confirmsignup(self):
        username = self.usernameLine.text()
        password = self.passwordLine.text()
        confirmpw = self.confirmPasswordLine.text()
        msg = QMessageBox()
        msg.setWindowTitle("Warning!")
        if not username or not password or not confirmpw:
            return 0

        if password != confirmpw:
            msg.setText("password is different than confirmPassword")
            msg.exec()
            return 0

        else:
            with open("Account/accountInfo.txt") as file:
                for line in file:
                    info = line.split(" ")
                    if username == info[0]:
                        msg.setText("Username has been taken, make another one.")
                        msg.exec()
                        return 0
            file.close()

            with open("Account/accountInfo.txt", 'a') as file:
                info = username + " " + password + "\n"
                file.write(info)
                file.close()
                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex() + 1)

    def backtohome(self):
        widget.addWidget(homePage)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class Help(QDialog):
    def __init__(self):
        super(Help, self).__init__()
        loadUi("D:/google downloads/Intelligent-Neurosurgical-Patient-Profiling-Model_Project-master/Intelligent-Neurosurgical-Patient-Profiling-Model_Project-master/ui/help.ui", self)
        self.homeButton.clicked.connect(self.backtohome)

        

    def backtohome(self):
        widget.addWidget(homePage)
        widget.setCurrentIndex(widget.currentIndex() + 1)
App = QApplication(sys.argv)
homePage = HomePage()
widget = QStackedWidget()
widget.addWidget(homePage)
widget.setFixedWidth(800)
widget.setFixedHeight(950)
widget.show()
sys.exit(App.exec())
