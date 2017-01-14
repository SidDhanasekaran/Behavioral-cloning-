Version 2: 

Added the following information based on the feedback from the first submission.

1) Added BatchNormalization in the neural network. Updates in model.py and the Model Architecture of this document.
2) Added keras.callbacks.EarlyStopping to avaoid the model from overfitting or underfitting. Update in model.py
3) Added how the model was designed in this document.


Overview

The following python codes were used to create a behaviorial cloning model that would drive the car in the provided track. The simulatour created in unity engine and provided by Udacity. 

Hardware:

I used AWS EC2 Linux instance for running my code. Due to the amount of image processing involved GPU instance is recommended. I used the below AWS nvidia GPU instance

Model		GPUs	vCPU	Mem (GiB)	SSD Storage (GB)
g2.8xlarge	4	32	60		120

Data: 

Attempt 1 - I recorded my own data using the keyboard. The steering angles were unreliable
Attempt 2 - I used a PS3 remote to record the data. I drove the track 5 times on both directions to remove any bias. I didnt have enough data.
Attempt 3 - I used the data provided by Udacity and I added recovery data created by Annie Flippo. Still not enough data.
Attempt 4 - I used the data provided by udacity and augumented the data as suggested by Vivek Yadav on carND slack. I was able to get 20,000 plus samples. The model seems to perform well at 20,000+ samples. 

Model Architecture: 
  
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 21, 76, 60)    4560        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
batchnormalization_1 (BatchNormal(None, 21, 76, 60)    120         convolution2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 21, 76, 60)    0           batchnormalization_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 10, 38, 60)    0           activation_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 8, 36, 30)     16230       maxpooling2d_1[0][0]
____________________________________________________________________________________________________
batchnormalization_2 (BatchNormal(None, 8, 36, 30)     60          convolution2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 8, 36, 30)     0           batchnormalization_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 4, 18, 30)     0           activation_2[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4, 18, 30)     0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2160)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           276608      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            6450        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             51          dense_2[0][0]
====================================================================================================
Total params: 304079
____________________________________________________________________________________________________
Train on 20491 samples, validate on 3617 samples

Model creation: 

I tried with the NVidia's architecture. I was not able to get past the first right turn. I then tried to use VGG16's first two convolution blocks as feature extractors, but this method failed. I started looking at Commai model as suggested in Slack. The model used is based on Commai model, I used just the 2 convolutional layers instead of 3 to start out with and to avoid stress on my local machine. I was able to get better performance with 2 convolutional layers than the 3 provided in commai. The parameters used are similar to the commai model. 


Code 1: model.py

In this script I first augument the input images to create enough samples and the input images are normalized and processed. I then train the model using the architecture above. When the training is done, the model and weights are saved as model.json and model.h5

Code 2: drive.py

This is the python script that receives the data from the Udacity simulator, generates the required steering angle using the deep learning model, and sends the throttle and the steering angles back to the simulator.

For treating the incoming pictures the same way as the model was trained. I processed the image just like in model.py


Training:

For Training I used 80% of the the images to train the model. And 20% for validation purposes. Test was always performed in the Simulator

My Batch-Size is 128 images. The network ran 3 epochs. And finally it gave me a result of one steering wheel value.


Output :

Here is the link to the screen capture of the output - https://youtu.be/1DfDQ_7NPIM






