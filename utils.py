###############################
##### AUXILIARY FUNCTIONS #####
###############################

import os, os.path
from statistics import mean
import itertools
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from emgFeatures import *
import tensorflow as tf

# Cross validation for sklearn models
def cvClassifiers(X, Y, list_Classifiers, k=10):

    ##### Input Parameters #####
    # X - Input Data
    # Y - Labels
    # list_Classifiers - List with all sklearn models
    # k - Number of folds

    ##### Outputs #####
    # train_scores - Average scores in training sets
    # val_scores - Average scores in test sets
    # val_cm - Confunsion matrix for test sets

    # Cross validation Kfold definition
    strtfdKFold = StratifiedKFold(n_splits=k)
    kfold = strtfdKFold.split(X, Y)

    # Train scores
    train_acc_scores = [[] for c in list_Classifiers]
    train_f1_scores = [[] for c in list_Classifiers]

    # Test scores
    val_acc_scores = [[] for c in list_Classifiers]
    val_f1_scores = [[] for c in list_Classifiers]

    # Ppredictions
    predicted_y = [[] for c in list_Classifiers]
    real_y = [[] for c in list_Classifiers]

    for train_index, val_index in kfold:
        for i, c in enumerate(list_Classifiers): 

            model = clone(c)

            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = Y[train_index], Y[val_index]

            model.fit(X_train_cv, y_train_cv)

            # Train evaluation
            py = model.predict(X_train_cv)

            train_acc_scores[i].append(model.score(X_train_cv,y_train_cv))
            train_f1_scores[i].append(f1_score(y_train_cv, py, average='macro'))

            # Test evaluation
            py = model.predict(X_val_cv)

            val_acc_scores[i].append(model.score(X_val_cv,y_val_cv))
            val_f1_scores[i].append(f1_score(y_val_cv, py, average='macro'))

            predicted_y[i] = list(itertools.chain(predicted_y[i],py.flatten().tolist()))
            real_y[i] = list(itertools.chain(real_y[i],y_val_cv.flatten().tolist()))

    # Average training scores
    train_accuracy = [mean(sc) for sc in train_acc_scores]
    train_f1 = [mean(sc) for sc in train_f1_scores]

    # Average test scores
    val_accuracy = [mean(sc) for sc in val_acc_scores]
    val_f1 = [mean(sc) for sc in val_f1_scores]
    val_cm = [confusion_matrix(real_y[i],predicted_y[i]) for i in range(0,len(list_Classifiers))]

    train_scores = [train_accuracy, train_f1]
    val_scores = [val_accuracy, val_f1]

    return train_scores, val_scores, val_cm

# Cross validation for keras models
def cvKeras(X, Y, model, k=10):

    strtfdKFold = StratifiedKFold(n_splits=k)
    kfold = strtfdKFold.split(X, Y)

    train_acc_scores = []
    train_f1_scores = []

    val_acc_scores = []
    val_f1_scores = []

    for train_index, val_index in kfold:

        m = model
        
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = Y[train_index], Y[val_index]

        y_train_cv_oh = tf.one_hot(y_train_cv, 5)
        y_val_cv_oh = tf.one_hot(y_val_cv, 5)

        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['Accuracy'])
        m.fit(x=X_train_cv, y=y_train_cv_oh, epochs=500)

        py = np.argmax(m.predict(X_train_cv), axis=1)
        
        train_acc_scores.append(m.evaluate(X_train_cv, y_train_cv_oh)[1])
        train_f1_scores.append(f1_score(y_train_cv, py, average='macro'))

        py = np.argmax(m.predict(X_val_cv), axis=1)
        
        val_acc_scores.append(m.evaluate(X_val_cv, y_val_cv_oh)[1])
        val_f1_scores.append(f1_score(y_val_cv, py, average='macro'))

    train_accuracy = mean(train_acc_scores)
    train_f1 = mean(train_f1_scores)

    val_accuracy = mean(val_acc_scores)
    val_f1 = mean(val_f1_scores)

    train_scores = [train_accuracy, train_f1]
    val_scores = [val_accuracy, val_f1]

    return train_scores, val_scores

# Load raw data in windows
def loadRawData(folder_path, w, w_inc):

    # folder_path - Direccion de los archivos
    # w - Tamaño de ventana
    # w_inc - Incremento de ventana

    files = [name for name in os.listdir(folder_path)]
    files.sort()

    dataset = []

    for file in files:

        print(file)

        data = pd.read_csv(folder_path + '/' + file, sep='\t', header=None).values[:,0:5]
        len_data = data.shape[0]

        for i in range(0, len_data - w + 1, w_inc):

            dataset.append(data[i:i + w,:])

    return dataset

# Cargar ventanas de datos realizando extracción de características
def loadFeatureData(folder_path, w, w_inc):

    # folder_path - Direccion de los archivos
    # w - Tamaño de ventana
    # w_inc - Incremento de ventana
  
    files = [name for name in os.listdir(folder_path)]
    files.sort()

    dataset = []
    error = 2
    angle_error_fe = 5
    angle_error_ps = 20

    for file in files:

        print(file)

        data = pd.read_csv(folder_path + '/' + file, sep='\t', header=None).values[:,0:5]
        len_data = data.shape[0]

        rest_angle_fe = np.mean(data[0:w,4])
        rest_angle_ps = ((max(data[:,4]) - min(data[:,4]))/2) + min(data[:,4])
        
        for i in range(0,len_data - w + 1,w_inc):

            d = data[i:i + w,:]
            d_rms = rms(d[:,0:4])
            d_mav = mav(d[:,0:4])
            d_zc = zeroCrossing(d[:,0:4], T=0.02)
            d_wl = wl(d[:,0:4])
            d_mdf = mdf(d[:,0:4])
            d_mnf = mnf(d[:,0:4])
            d_mom = mom(d[:,0:4], 4)

            d_features = np.concatenate((d_rms, d_mav, d_zc, d_wl, d_mdf, d_mnf, d_mom))

            if "FE" in file:
                mode = 0
                if d[-1,4] > rest_angle_fe + angle_error_fe:
                    c = "Flexion"
                elif d[-1,4] < rest_angle_fe - angle_error_fe:
                    c = "Extension"
                elif d[-1,4] > rest_angle_fe - angle_error_fe and d[-1,4] < rest_angle_fe + angle_error_fe:
                    c = "Quieto"
            elif "PS" in file:
                mode = 1
                if d[-1,4] > rest_angle_ps + angle_error_ps:
                    c = "Pronacion"
                elif d[-1,4] < rest_angle_ps - angle_error_ps:
                    c = "Supinacion"
                elif d[-1,4] > rest_angle_ps - angle_error_ps and d[-1,4] < rest_angle_ps + angle_error_ps:
                    c = "Quieto"
        
            dataset.append(np.append(d_features, [mode, c, d[-1,4]]))

    return dataset

# Moving average filter
def maf(data, p=20):

    # data - 
    # p - 

    data = list(data)
    maf_emg = []

    for w in data:

        d = w - np.mean(w, axis=0)
        d = np.absolute(d)

        for i in range(w.shape[1]-1):
            d[:,i] = np.convolve(d[:,i], np.ones(p)/p, mode='same')
            
        maf_emg.append(d)

    maf_emg = np.array(maf_emg)

    return maf_emg

# Extract EMG features
def EMGfeatures(data):

    d = list(data)

    feature_data = []

    for w in d:

        d_rms = rms(w)
        d_mav = mav(w)
        d_zc = zeroCrossing(w)
        d_wl = wl(w)
        d_mdf = mdf(w)
        d_mnf = mnf(w)
        d_mom = mom(w, 4)

        d_features = np.concatenate((d_rms, d_mav, d_zc, d_wl, d_mdf, d_mnf, d_mom))

        feature_data.append(d_features)
    
    emg_features_data = np.array(feature_data)

    return emg_features_data

# Plot model loss
def plotLoss(history):

    plt.figure(figsize=(10,6))
    plt.title('Model Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
