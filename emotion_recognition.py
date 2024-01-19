
################################################################################
#                      EMOTION RECOGNITION FULL CODE                           #
################################################################################

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop



folder_path = "/content/drive/MyDrive/emotions_dataset/" # the data resides in google drive


#function which creates the training and test dataset from google drive
def create_train_and_test_dataset():
  picture_size = 48
  batch_size  = 256

  datagen_train  = ImageDataGenerator()
  datagen_val = ImageDataGenerator()


  # flow_from_directory method from keras takes all images in the subdirectory
  # and labels them automatically according to the folder names in that directory
  # target size is the size of the input image
  # class mode specifies whether it is binary or categorical which is our case

  train_set = datagen_train.flow_from_directory(folder_path+"train_set",
                                                target_size = (picture_size,picture_size),
                                                color_mode = "grayscale",
                                                batch_size=batch_size,
                                                class_mode='categorical')


  # do the same with test_set also
  test_set = datagen_val.flow_from_directory(folder_path+"test_set",
                                                target_size = (picture_size,picture_size),
                                                color_mode = "grayscale",
                                                batch_size=batch_size,
                                                class_mode='categorical')

  return train_set, test_set




# create the model for CNN
def model_CNN():

  number_of_classes = 4

  model = Sequential()


  # add 5 CNN layers to model
  # Convolutional filters detect specific features in the image data,
  # it is increased in each layer since it can capture patterns better with this way
  # padding is used in images to have the same sized inputs
  # Batch Norm normalizes the data between the layers for compatibility
  # pool layer is added to reduce the dimension of feature map
  # we take maxpooling since we want to take maximum pixel value in each kernel
  # lastly I add dropout in several layers to eliminate the effect of some neurons in hidden layer to prevent overfitting


  model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))


  model.add(Conv2D(128,(5,5),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout (0.25))


  model.add(Conv2D(512,(3,3),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))

  model.add(Conv2D(128,(5,5),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout (0.25))

  model.add(Conv2D(128,(5,5),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout (0.25))



  # flatten 2D data from CNN to linear data by flattening
  model.add(Flatten())

  #Fully connected 5 layers
  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  #lastly add the output layer with the same number of output numbers as emotions
  # using softmax since it is not binary, is categorical classification
  model.add(Dense(number_of_classes, activation='softmax'))


  #use adam optimizer with learning rate as 0.0001
  # categorical_crossentropy is used as loss function since it is good when we have more than two labels in data
  optimizer_type = Adam(lr = 0.001)
  model.compile(optimizer=optimizer_type,loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  return model


train_set, test_set = create_train_and_test_dataset()
model = model_CNN()

print("---------------",train_set)

#train and validate the model with 2 datasets
history = model.fit_generator(generator=train_set,
                                epochs=10,
                                validation_data = test_set)


import numpy as np
import keras
#save the model for later use
saved_model_path  = '/content/drive/MyDrive/keras_models/emotion_recognition_2.keras'
#model.save(saved_model_path) # YOU CAN DELETE THE COMMENT SIGN TO SAVE THE NEWLY TRAINED MODEL FOR LATER USE
