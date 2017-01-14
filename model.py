import keras
import os
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import json
from keras.optimizers import SGD, Adam, RMSprop
#Added for second submission
from keras.layers.normalization import BatchNormalization

#Initializing
image_path = '/home/siddarthd2919/Downloads/behaviorial_cloning/data'

#Data Augumentation to increase the number of samples
data = pd.read_csv(os.path.join(image_path,'driving_log.csv'))
data['left_angle'] = data['steering'].apply(lambda x: x + .25 if np.abs(x) > .25 else x + .1)
data['right_angle'] = data['steering'].apply(lambda x: x - .25 if np.abs(x) > .25 else x - .1)
X_train_center = np.zeros((len(data),25,80,3))
X_train_left = np.zeros((len(data),25,80,3))
X_train_right = np.zeros((len(data),25,80,3))

#Iterate over DataFrame rows
for index,row in data.iterrows():
    filepath = os.path.join(image_path,row['center'])
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_center[index,:,:,:] = img

    filepath = os.path.join(image_path,row['left'].replace(' ', ''))
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_left[index,:,:,:] = img

    filepath = os.path.join(image_path,row['right'].replace(' ', ''))
    img=mpimg.imread(filepath)
    img = img/255.0
    img = img[40:140,:]
    img = cv2.resize(img,(80,25))
    X_train_right[index,:,:,:] = img

y_train_center = np.array(data.steering)
y_train_left = np.array(data.left_angle)
y_train_right = np.array(data.right_angle)

X_train = np.append(X_train_center, X_train_left, axis = 0)
X_train = np.append(X_train, X_train_right, axis = 0)

y_train = np.append(y_train_center, y_train_left, axis = 0)
y_train = np.append(y_train, y_train_right, axis = 0)

#splitting the training and test data. The data is randomized and 15% of it is used as the test data

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = .15,random_state = 10)

#
def model():
    # Initiating the model
    model = Sequential()
    # The first convolutional layer will turn 1 channel into 60 channels. 
    model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=(25, 80, 3), activation='relu'))
    #Added for second submission -Normalization Layer
    model.add(BatchNormalization())
    #Applying Relu
    model.add(Activation('relu'))
    # Apply Max Pooling for each 2 x 2 pixels
    model.add(MaxPooling2D(pool_size=(2, 2)))   
    # The second convolutional layer will convert 60 channels into 30 channels
    model.add(Convolution2D(30, 3, 3, activation='relu'))
    #Added for second submission - Normalization Layer
    model.add(BatchNormalization())
    #Applying Relu
    model.add(Activation('relu'))
    # Apply Max Pooling for each 2 x 2 pixels
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Apply dropout of 20%
    model.add(Dropout(0.2))
    # Flatten the matrix.
    model.add(Flatten())
    # Output 128
    model.add(Dense(128, activation='relu'))
    # Input 128 Output 50
    model.add(Dense(50, activation='relu'))
    # Input 50 Output 1
    model.add(Dense(1, init = 'normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Print out summary of the model
    model.summary()
    return model

#Training the model
model = model()
#Added for second submission - Point 3 
early_stop_callback = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=1, mode='auto')]
model.fit(X_train, y_train, nb_epoch=15, batch_size=128, verbose=2, validation_data = (X_val,y_val), callbacks = early_stop_callback)

model_name = 'model'
model.save_weights(model_name+'.h5')

# save as JSON
json_string = model.to_json()
import json
with open(model_name+'.json', 'w') as outfile:
    json.dump(json_string, outfile)
