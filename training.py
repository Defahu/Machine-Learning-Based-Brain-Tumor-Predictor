# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import csv

import matplotlib.pyplot as plt

import os

classifier = Sequential()

# layer 1
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      input_shape=(512, 512, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(6, 6)))

# layer 2
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(6, 6)))

classifier.add(Flatten())

classifier.add(Dense(units=252, activation="relu"))
classifier.add(Dense(units=4, activation='softmax'))

classifier.compile(
    optimizer="adam", loss="CategoricalCrossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
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
# LOADING IMAGES
##################################################################

training_dataset = train_datagen.flow_from_directory('data/Training',
                                                     target_size=(512, 512),
                                                     batch_size=64,
                                                     class_mode='categorical')

testing_dataset = test_datagen.flow_from_directory('data/Testing',
                                                   target_size=(512, 512),
                                                   batch_size=64,
                                                   class_mode='categorical')

##################################################################
# Fitting the model with the supplied images
##################################################################

# Creating callbacks for the model.
# If the model dosen't continue to improve (loss), the trainning will stop.
# Stop training if loss doesn't keep decreasing.
model1_es = EarlyStopping(monitor='loss', min_delta=0.001,
                          patience=6, verbose=1, baseline=None)
model1_rlr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, verbose=1, baseline=None)

# Automatically saves the best weights of the model, based on best val_accuracy
model1_mcp = ModelCheckpoint(filepath='model1_weights.h5', monitor='val_categorical_accuracy',
                             save_best_only=True, verbose=1)

history = classifier.fit(training_dataset,
               steps_per_epoch=89,
               epochs=50,
               validation_data=testing_dataset,
               validation_steps=20,
               callbacks=[model1_es, model1_rlr, model1_mcp])

##################################################################
# Plots
##################################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc_plt')

plt.figure(2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_plt')

##################################################################
# Save accuracy and results to file
##################################################################

resultf = open('rf.txt', 'w')
resultf.close()
resultf = open('rf.csv', 'a')
n = 1 #epoch count in data
csv.writer(resultf).writerow('accuracy')
for d in acc:
    row = [n,str(d)]
    csv.writer(resultf).writerow(row)
    n = n+1
n = 1
csv.writer(resultf).writerow('val_accuracy')
for d in val_acc:
    row = [n,str(d)]
    csv.writer(resultf).writerow(row)
    n = n+1
n = 1
csv.writer(resultf).writerow('loss')
for d in loss:
    row = [n,str(d)]
    csv.writer(resultf).writerow(row)
    n = n+1
n = 1
csv.writer(resultf).writerow('val_loss')
for d in val_loss:
    row = [n,str(d)]
    csv.writer(resultf).writerow(row)
    n = n+1
resultf.close()


##################################################################
# Save Progress
##################################################################

classifier.evaluate(testing_dataset)

classifier.summary()

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
classifier.save_weights("model-50times-rmffd.h5")
print("Saved model to disk")
