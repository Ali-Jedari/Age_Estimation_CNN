#!/usr/bin/python
#coding: utf-8

'''
Bachelor's Final Project:
Implementing a Convolutional Neural Network for the Task of Age Estimation 
Based on the following paper:
C. Miron, V. Manta, R. Timofte, A. Pasarica, R. Ciucu, “Efficient convolutional neural network for
apparent age prediction”, 2019 IEEE 15th International Conference on Intelligent Computer
Communication and Processing (ICCP)

Datasets: 
    UTKFace available at https://susanqq.github.io/UTKFace/
    APPA-REAL available at http://158.109.8.102/AppaRealAge/appa-real-release.zip
Original notebook at https://colab.research.google.com/drive/1GTDfkU_g1YEKDk64PTF7t9Rnzl61gct5

Run this code as:
python3 Age_Estimation_CNN.py dataset /path/to/dataset
dataset for APPA-REAL dataset can either be appa or appa-real
and for UTKFace it can be utk or utkface
[case insensitive]

for example:
    python Age_Estimation_CNN.py appa ~/Documents/Datasets/Images/APPA-Real/

Ali Jedari Heidarzadeh
University of Tabriz, Tabriz, Iran
Winter 2021-2022
'''

#import the necessary libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sys import argv
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Sequential

def utk_data(path):
    '''
    Especially designed for UTKFace dataset
    Reads all the image files in the given path
    Resizes the images to 50x50
    Splits them into three subsets: Train, Validation, and Test
    returns X_train, X_valid, X_test, y_train, y_valid, y_test
    '''

    print("Preparing the data...")
    files = os.listdir(path)
    images, ages = [], []
    
    for f in files:
        ages.append(int(f.split('_')[0]))
        image = cv2.imread(path+f)
        image = cv2.resize(image, dsize=(50, 50))
        images.append(image / 255)
    
    images = np.array(images)
    ages = np.array(ages)
    train_samples, X_test, train_labels, y_test = train_test_split(images, ages, test_size=0.15, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(train_samples, train_labels, test_size=0.05, shuffle=True)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def appa_real_data(path):
    '''
    Especially designed for APPA-REAL dataset
    Reads all the image files in the given path
    Resizes the images to 50x50
    Splits them into three subsets: Train, Validation, and Test
    returns X_train, X_valid, X_test, y_train, y_valid, y_test
    '''
    
    print("Preparing the data...")
    
    train_path = path+'train/'
    gt = pd.read_csv(path+'gt_avg_train.csv')
    tr = gt['real_age']
    X = []
    for i in range(gt.shape[0]):
        img = cv2.imread(train_path+gt.iloc[i]['file_name']+'_face.jpg')
        X.append(cv2.resize(img, dsize=(50, 50)))    

    test_path = path+'test/'
    gt = pd.read_csv(path+'gt_avg_test.csv')
    ts = gt['real_age']
    for i in range(gt.shape[0]):
        img = cv2.imread(test_path+gt.iloc[i]['file_name']+'_face.jpg')
        X.append(cv2.resize(img, dsize=(50, 50)))

    valid_path = path+'valid/'
    gt = pd.read_csv(path+'gt_avg_valid.csv')
    vl = gt['real_age']
    for i in range(gt.shape[0]):
        img = cv2.imread(valid_path+gt.iloc[i]['file_name']+'_face.jpg')
        X.append(cv2.resize(img, dsize=(50, 50)))

    X = np.array(X) / 255
    y = pd.concat([tr, ts, vl])
    train, X_test, train_labels, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(train, train_labels, test_size=0.1, shuffle=True)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def main():
    try:
        #dataset has to be chosen by the user
        dataset = argv[1].lower()
    except IndexError:
        #otherwise, the following message is printed and the code exits
        print('Dataset is not provided!')
        print('Available datasets are:\nUTK (or UTKFace) and APPA (or APPA-REAL)')
        print('Run this code as "python Age_Estimation_CNN.py [dataset] [path/to/dataset](optional)"')
        exit()
    
    try:
        #path to the dataset can be given via argv
        path = argv[2] if argv[2][-1] == '/' else argv[2]+'/'
    except IndexError:
        #if it is not provided, path variable is set to a default value according to the chosen dataset
        if argv[1].lower() in ['utk', 'utkface']:
            path = './UTKFace/'
        elif argv[1].lower() in ['appa', 'appa-real']:
            path = './appa-real-release/'
        else:
            print('Provided dataset name is invalid!')
            print('Valid dataset names are: \nUTK (or UTKFace) and APPA (or APPA-REAL)')
            print('[case insensitive]')
            exit()
    
    #preparing the data for train, validation, and test
    if dataset in ['utk', 'utkface']:
        X_train, X_valid, X_test, y_train, y_valid, y_test = utk_data(path) 
    elif dataset in ['appa', 'appa-real']:
        X_train, X_valid, X_test, y_train, y_valid, y_test = appa_real_data(path) 
    print('Successful!')
    
    mx = np.max(np.array([y_train.max(), y_test.max(), y_valid.max()]))
    mn = np.min(np.array([y_train.min(), y_test.min(), y_valid.min()]))
    output_shape = mx - mn + 1
    
    print("Defining the model...")
    model = Sequential()  #define the model

    #Adding layers to the model
    model.add(Input(shape=(50, 50, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='relu'))

    #uncomment the following line if you want the summary of the model to be printed
    #model.summary()

    print("Training the model...")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

    print("Testing the model...")
    results = model.evaluate(X_test, y_test)

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])
    plt.show()

if __name__ == '__main__':
    main()