#!/usr/bin/env python
# coding: utf-8

# # Required functions for loading and processing data set

# ## Loading libraries

# In[1]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import glob
#import cv2
import os, sys, pickle, gzip, time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D,Activation,Dropout
from tensorflow.keras.layers import Dense,Flatten,Input,concatenate, GlobalAveragePooling2D
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications import EfficientNetB7, VGG16, VGG19, imagenet_utils

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

print("INFO............................")

print("python version:{}\n".format(sys.version))

print("TF version:{}\n".format(tf.__version__))

t1 = time.time()
# ## Loading images min-max

# In[2]:


def load_image_minmax(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["Cr","Co","temperature","min", "max","time", "image_name"]
    df = pd.read_csv(inputPath,header=None, names=cols)
    # return the data frame
    return df


# ## Processing of image min_max

# In[3]:


def process_image_minmax(df, train, test):
    # initialize the column names of the continuous data
    continuous = ["min", "max"]
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]

#     cs = MinMaxScaler()
#     trainContinuous = cs.fit_transform(train[continuous])
#     testContinuous = cs.transform(test[continuous])
    trainContinuous = train[continuous]
    testContinuous = test[continuous]
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    # trainX = np.hstack([trainContinuous])
    # testX = np.hstack([testContinuous])
    # return the concatenated training and testing data
    trainX = np.array(trainContinuous)
    testX = np.array(testContinuous)
    return (trainX, testX)


# ## Extract image features by EfficientNetB4 convolutional layers

# In[28]:


def create_features(df, inputPath, pre_model,size_img):

    images = []

    # loop over the images
    for i in df['image_name']:
        imagePath = os.path.join(inputPath,i)
    # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=size_img)
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        images.append(image)

    x = np.vstack(images)
    features = pre_model.predict(x, batch_size=32)
    print(features.shape)
    features_flatten = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
    return features_flatten
    #return features


# # Model development

# In[5]:


# Swish defination
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))


# In[6]:


get_custom_objects().update({'swish': Activation(swish)})


# ## FC for extracted image features

# In[7]:


def create_feature_model(dim, regress=False):
    model = Sequential()
    model.add(Dense(512, input_dim=dim))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.7))
    # model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(swish))

    # model.add(Dense(16, input_dim=dim, activation="relu"))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model


# # building 2 fully connected layer
# x = model.output

# x = BatchNormalization()(x)
# x = Dropout(0.7)(x)

# x = Dense(512)(x)
# x = BatchNormalization()(x)
# x = Activation(swish_act)(x)
# x = Dropout(0.5)(x)

# x = Dense(128)(x)
# x = BatchNormalization()(x)
# x = Activation(swish_act)(x)


# ## FC for combining output of CNN with numeric data (image min_max)

# In[8]:


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
    # return our model
    return model
def plot_loss(history,plot_name):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.title(plot_name)
    plt.savefig(plot_name, dpi=120, facecolor='w', edgecolor='w')

def parity_plot(ground_truth, prediction, lims_min, lims_max, plot_name):
    plt.figure()
    plt.scatter(ground_truth, prediction, label='Data')
    #plt.plot(ground_truth, ground_truth, color='k', label='Predictions')
    lims = [lims_min, lims_max]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.xlabel('Groud truth', fontsize=18)
    plt.ylabel('Prediction', fontsize=18)
    #plt.legend()
    plt.title(plot_name)
    plt.savefig(plot_name,dpi=120, facecolor='w', edgecolor='w')


# # Data

# ## Data path

# In[13]:


path='/bsuhome/afarizhandi/scratch/process_chem_time_predict/Clean_data_remove_co0.05'
prerained_model = EfficientNetB7()
size_img = (224, 224)
num_used_layers = [25, 108, 212, 286, 346, 406, 464, 509, 613, 673, 806, 810]
epch = 750

# In[29]:


print("[INFO] loading images min-max...")
inputPath = os.path.sep.join([path, "clean_data.csv"])
df = load_image_minmax(inputPath)

print("Data shape")
print(df.shape)
accuracy_results=pd.DataFrame()
for i in num_used_layers:
    print('--------------------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------------------')
    print('[INFO] AFTER LAYER '+str(i))
    print('--------------------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------------------')
    print("[INFO] loading spinodal images...")
    pre_model = Model(inputs=prerained_model.inputs, outputs=prerained_model.layers[i].output)
    images = create_features(df, path, pre_model, size_img)

    print('Image shape layer' +str(i))
    print(images.shape)


    # In[15]:


    print("[INFO] processing data...")
    split = train_test_split(df, images, test_size=0.25, random_state=42)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = split

    print('Train sahpe layer' +str(i))
    print(trainAttrX.shape)
    print('Test sahpe layer' +str(i))
    print(testAttrX.shape)

    colY = ["time", "temperature", "Cr", "Co"]

    maxTemp = trainAttrX["temperature"].max()
    trainAttrX["temperature"] = trainAttrX["temperature"] / maxTemp
    testAttrX["temperature"] = testAttrX["temperature"] / maxTemp

    maxTime = trainAttrX["time"].max()
    trainAttrX["time"] = trainAttrX["time"] / maxTime
    testAttrX["time"] = testAttrX["time"] / maxTime

    trainY = np.array(trainAttrX[colY])
    print("Train labels shape")
    print(trainY.shape)
    #print(testAttrX)
    testY = np.array(testAttrX[colY])
    print("Test labels shape")
    print(testY.shape)
    # process the image minmax data by performing min-max scaling
    (trainAttrX, testAttrX) = process_image_minmax(df,trainAttrX, testAttrX)
    df_trainY=pd.DataFrame(trainY)
    df_testY=pd.DataFrame(testY)
    df_trainY.to_csv('trainY_layer{}.csv'.format(i))
    df_testY.to_csv('testY_layer{}.csv'.format(i))


    # # Compile the Model

    # In[16]:


    # create the MLP and CNN models
    mlp = create_mlp(trainAttrX.shape[1], regress=False)
    cnn = create_feature_model(trainImagesX.shape[1], regress=False)
    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn.output])
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(32, activation="relu")(combinedInput)
    x = Dense(16, activation="relu")(x)
    x = Dense(4, activation="linear")(x)
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted temperature of the spinodal)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our temperature *predictions* and the *actual temperatures*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


    # In[17]:


    model.summary()


    # # train the model

    # In[18]:


    best_weights='outputs/weights_best_'+str(i)+'layers_.hdf5'
    checkpoint = ModelCheckpoint(best_weights, monitor='val_loss', verbose=1, save_best_only=True,
                                                    mode='min')
    print("[INFO] training model...")
    history = model.fit(x=[trainAttrX, trainImagesX], y=trainY,validation_data=([testAttrX, testImagesX], testY),
                        epochs=epch, batch_size=8, callbacks=[checkpoint],verbose=0)
    print('--------------------TRAINING LAST EPOCH---------------------------------')
    print("[INFO] predicting training with the best weights...")
    preds = model.predict([trainAttrX, trainImagesX])
    #print("Actual Temperature & Cr & Co ...")
    print("Time accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,0], preds[:,0]))
    print("MSE:")
    print(mean_squared_error(trainY[:,0], preds[:,0]))

    print("Temperature accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,1], preds[:,1]))
    print("MSE:")
    print(mean_squared_error(trainY[:,1], preds[:,1]))

    print("Cr accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,2], preds[:,2]))
    print("MSE:")
    print(mean_squared_error(trainY[:,2], preds[:,2]))

    print("Co accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,3], preds[:,3]))
    print("MSE:")
    print(mean_squared_error(trainY[:,3], preds[:,3]),'\n')


    print('--------------------TEST LAST EPOCH---------------------------------')
    # make predictions on the testing data
    print("[INFO] predicting with the last weights...")
    preds = model.predict([testAttrX, testImagesX])
    #print("Actual Temperature & Cr & Co ...")
    print("Time accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,0], preds[:,0]))
    print("MSE:")
    print(mean_squared_error(testY[:,0], preds[:,0]))

    print("Temperature accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,1], preds[:,1]))
    print("MSE:")
    print(mean_squared_error(testY[:,1], preds[:,1]))

    print("Cr accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,2], preds[:,2]))
    print("MSE:")
    print(mean_squared_error(testY[:,2], preds[:,2]))

    print("Co accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,3], preds[:,3]))
    print("MSE:")
    print(mean_squared_error(testY[:,3], preds[:,3]),'\n')

    print('--------------------TRAINING BEST EPOCH---------------------------------')
    print("[INFO] predicting with the best weights...")
    print('Test accuracy of the epoch that resulted  in the best validation accuracy_EfficientNetB7_'+str(i)+'layers\n')
    model.load_weights(best_weights)
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    preds = model.predict([trainAttrX, trainImagesX])
    df_pred_train=pd.DataFrame(preds)
    df_pred_train.to_csv('pred_train_layer{}.csv'.format(i))

    print("Time accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,0], preds[:,0]))
    time_R_square_train=r2_score(trainY[:,0], preds[:,0])
    print("MSE:")
    print(mean_squared_error(trainY[:,0], preds[:,0]))
    time_RMSE_train=mean_squared_error(trainY[:,0], preds[:,0])

    print("Temperature accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,1], preds[:,1]))
    Temp_R_square_train=r2_score(trainY[:,1], preds[:,1])
    print("MSE:")
    print(mean_squared_error(trainY[:,1], preds[:,1]))
    Temp_RMSE_train=mean_squared_error(trainY[:,1], preds[:,1])

    print("Cr accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,2], preds[:,2]))
    Cr_R_square_train=r2_score(trainY[:,2], preds[:,2])
    print("MSE:")
    print(mean_squared_error(trainY[:,2], preds[:,2]))
    Cr_RMSE_train= mean_squared_error(trainY[:,2], preds[:,2])

    print("Co accuracy ...")
    print("R-squared:")
    print(r2_score(trainY[:,3], preds[:,3]))
    Co_R_square_train=r2_score(trainY[:,3], preds[:,3])
    print("MSE:")
    print(mean_squared_error(trainY[:,3], preds[:,3]))
    Co_RMSE_train=mean_squared_error(trainY[:,3], preds[:,3])

    # make predictions on the testing data


    # In[19]:

    print('--------------------TEST BEST EPOCH---------------------------------')
    print("[INFO] predicting with the best weights...")
    print('Test accuracy of the epoch that resulted  in the best validation accuracy_ResNet50V2_'+str(i)+'layers\n')
    preds = model.predict([testAttrX, testImagesX])
    df_pred_test=pd.DataFrame(preds)
    df_pred_test.to_csv('pred_test_layer{}.csv'.format(i))
    print("Time accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,0], preds[:,0]))
    time_R_square_test=r2_score(testY[:,0], preds[:,0])
    print("MSE:")
    print(mean_squared_error(testY[:,0], preds[:,0]))
    time_RMSE_test=mean_squared_error(testY[:,0], preds[:,0])


    print("Temperature accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,1], preds[:,1]))
    Temp_R_square_test=r2_score(testY[:,1], preds[:,1])
    print("MSE:")
    print(mean_squared_error(testY[:,1], preds[:,1]))
    Temp_RMSE_test=mean_squared_error(testY[:,1], preds[:,1])

    print("Cr accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,2], preds[:,2]))
    Cr_R_square_test=r2_score(testY[:,2], preds[:,2])
    print("MSE:")
    print(mean_squared_error(testY[:,2], preds[:,2]))
    Cr_RMSE_test=mean_squared_error(testY[:,2], preds[:,2])

    print("Co accuracy ...")
    print("R-squared:")
    print(r2_score(testY[:,3], preds[:,3]))
    Co_R_square_test=r2_score(testY[:,3], preds[:,3])
    print("MSE:")
    print(mean_squared_error(testY[:,3], preds[:,3]))
    Co_RMSE_test=mean_squared_error(testY[:,3], preds[:,3])

    df1=pd.DataFrame([[time_R_square_train,time_RMSE_train,Temp_R_square_train,Temp_RMSE_train,
    Cr_R_square_train,Cr_RMSE_train,Co_R_square_train,Co_RMSE_train,
    time_R_square_test,time_RMSE_test,Temp_R_square_test,Temp_RMSE_test,
    Cr_R_square_test,Cr_RMSE_test,Co_R_square_test,Co_RMSE_test]],
    columns=['time_R_square_train','time_RMSE_train',
    'Temp_R_square_train','Temp_RMSE_train',
    'Cr_R_square_train','Cr_RMSE_train',
    'Co_R_square_train','Co_RMSE_train',
    'time_R_square_test','time_RMSE_test',
    'Temp_R_square_test','Temp_RMSE_test',
    'Cr_R_square_test','Cr_RMSE_test',
    'Co_R_square_test','Co_RMSE_test'],
    index=[str(i)])
    #print(df1)
    accuracy_results=accuracy_results.append(df1,ignore_index=True)


    # In[27]:
    plot_loss(history,'EfficientNetB7_'+str(i)+'layers')
    name_graph_temp = 'Time_parity_plot_EfficientNetB7_'+str(i)+'layers'
    parity_plot(testY[:,0]*1080000, preds[:,0]*1080000, (testY[:,0]*1080000).min()-10 ,(testY[:,0]*963).max()+10, name_graph_temp)

    name_graph_temp = 'Temp_parity_plot_EfficientNetB7_'+str(i)+'layers'
    parity_plot(testY[:,1]*963, preds[:,1]*970, (testY[:,1]*970).min()-10 ,(testY[:,1]*970).max()+10, name_graph_temp)

    name_graph_cr = 'Cr_parity_plot_EfficientNetB7_'+str(i)+'layers'
    parity_plot(testY[:,2], preds[:,2], 0, 1, name_graph_cr)

    name_graph_co = 'Co_parity_plot_EfficientNetB7_'+str(i)+'layers'
    parity_plot(testY[:,3], preds[:,3], 0, 1, name_graph_co)

    ################################ SAVE THE MODEL HISTORY ####################################

    picklefile = 'outputs/model_history_'+str(i)+'layers_.pkl'

    output = open(picklefile,'wb')

    pickle.dump(history.history, output)

    output.close()

    print('pickle file written to:{}'.format(picklefile))



    print('Total time taken:{}'.format(time.time()-t1))

accuracy_results.to_csv('accuracy_results.csv')
