################################
##### FUCNIONES AUXILIARES #####
################################

import os, os.path
from statistics import mean
import itertools
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from emgFeatures import *

# Validacion cruzada para modelos de sklearn
def cvClassifiers(X, Y, list_Classifiers, k=10):

    # X - Datos de entrada
    # Y - Etiquetas
    # list_Classifiers - Lista de los modelos
    # k - Numero de folds

    strtfdKFold = StratifiedKFold(n_splits=k)
    kfold = strtfdKFold.split(X, Y)

    acc_scores = [[] for c in list_Classifiers]
    pre_scores = [[] for c in list_Classifiers]
    rec_scores = [[] for c in list_Classifiers]
    f1_scores = [[] for c in list_Classifiers]

    predicted_y = [[] for c in list_Classifiers]
    real_y = [[] for c in list_Classifiers]

    for train_index, test_index in kfold:
        for i, c in enumerate(list_Classifiers): 

            model = clone(c)

            X_train_cv, X_test_cv = X[train_index], X[test_index]
            y_train_cv, y_test_cv = Y[train_index], Y[test_index]

            model.fit(X_train_cv, y_train_cv)

            py = model.predict(X_test_cv)

            acc_scores[i].append(model.score(X_test_cv,y_test_cv))
            pre_scores[i].append(precision_score(y_test_cv, py, average='macro'))
            rec_scores[i].append(recall_score(y_test_cv, py, average='macro'))
            f1_scores[i].append(f1_score(y_test_cv, py, average='macro'))

            predicted_y[i] = list(itertools.chain(predicted_y[i],py.flatten().tolist()))
            real_y[i] = list(itertools.chain(real_y[i],y_test_cv.flatten().tolist()))

    accuracy = [mean(sc)*100 for sc in acc_scores]
    precision = [mean(sc) for sc in pre_scores]
    recall = [mean(sc) for sc in rec_scores]
    f1 = [mean(sc) for sc in f1_scores]
    cm = [confusion_matrix(real_y[i],predicted_y[i]) for i in range(0,len(list_Classifiers))]

    return accuracy, precision, recall, f1, cm

# Cargar ventanas de datos sin procesar
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

        for i in range(0,len_data - w + 1,w_inc):

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

# Filtro de media movil
def maf(data, p=20):

    # data - 
    # p - 

    dataset = []

    for w in data:

        d = w[:,0:4] - np.mean(w[:,0:4], axis=0)
        d = np.absolute(d)

        for i in range(w.shape[1]-1):
            d[:,i] = np.convolve(d[:,i], np.ones(p)/p, mode='same')
            
        dataset.append(np.concatenate((d, w[:,-1].reshape((w.shape[0],1))), axis=1))

    return dataset

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
