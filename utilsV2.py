###############################
##### AUXILIARY FUNCTIONS #####
###############################

from cmath import cos
from math import radians, sin, cos, pi
import os, os.path
from re import L
from statistics import mean
import itertools
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from emgFeatures import *
import tensorflow as tf
from scipy.signal import butter, lfilter

# Load EMG files
def loadFiles(folder_path):

    data = [name for name in os.walk(folder_path)]

    files = {}

    for i in range(1,len(data)):

        emgData = {}

        for j in range(0,len(data[i][2])):

            if data[i][2][j].split(".")[0] == 'subject_info':
                subject_info = {}
                d = pd.read_csv(data[i][0] + '\\' + data[i][2][j], header=None).values
                for item in d:
                    subject_info[item[0].split(':')[0]] = float(item[0].split(':')[1])
                emgData[data[i][2][j].split(".")[0]] = subject_info
            else:
                emgData[data[i][2][j].split(".")[0]] = pd.read_csv(data[i][0] + '\\' + data[i][2][j], sep='\t', header=None).values

        files[data[0][1][i-1]] = emgData

    return files

# Add torque data
def addTorque(files):

    torqueFiles = {}

    for subject, data in files.items():

        subject_info = data['subject_info']

        for test, values in data.items():

            if test != 'subject_info':

                test_params = test.split('_')
                m = test_params[1]
                w = float(test_params[3])/1000

                angle = values[:,4]

                torque = calTorque(angle, subject_info, w, m)
                
                values = np.concatenate((values, torque.reshape((torque.shape[0],1))), axis=1)

                data[test] = values

        torqueFiles[subject] = data


    return torqueFiles

# Filter EMG signal
def filterEMG(files, cutOffL, cutOffH, fs):

    filteredFiles = {}

    for subject, data in files.items():

        for test, values in data.items():

            if test != 'subject_info':

                b, a = butter(4, [cutOffH,cutOffL], fs=fs, btype='bandpass', analog=False)

                values[:,0:3] = lfilter(b,a,values[:,0:3],axis=0)

                data[test] = values[200:,:]

        filteredFiles[subject] = data


    return filteredFiles

# Sliding window
def slidingWindow(files, w, w_inc):

    segmentedFiles = {}

    for subject, data in files.items():

        for test, values in data.items():

            emg = []
            angle = []
            torque = []

            if test != 'subject_info':

                for i in range(0,values.shape[0] - w + 1,w_inc):

                    d = values[i:i + w,:]
                    emg.append(d[:,0:4])
                    angle.append(d[-1,4])
                    torque.append(d[-1,5])

                    data[test] = [emg, angle, torque]

        segmentedFiles[subject] = data


    return segmentedFiles

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

# Extract RMS EMG
def EMGRMS(data):

    d = list(data)

    feature_data = []

    for w in d:

        d_rms = rms(w)

        feature_data.append(d_rms)
    
    emg_features_data = np.array(feature_data)

    return emg_features_data

# Calculate torque
def calTorque(angle, subject_info, w, m):

    b, a = butter(4,2,fs=1024,btype='low',analog=False)

    angle = lfilter(b, a, angle)
    rad_angle = np.radians(angle)

    speed = np.gradient(rad_angle,1/1024)
    acel = np.gradient(speed,1/1024)

    speed = lfilter(b, a, speed)
    acel = lfilter(b, a, acel)

    l_arm = subject_info['arm_length']
    lcm_arm = (0.43*l_arm)
    l_hand = subject_info['hand_length']
    lcm_hand = (0.43*l_hand)
    arm_weight = 0.023*subject_info['weight']
    hand_weight = 0.0057*subject_info['weight']
    g = 9.81

    if m == 'flex':
        torque = ((arm_weight + w)*lcm_arm**2 + ((1/3)*(arm_weight + w)*l_arm**2))*acel + (arm_weight*lcm_arm + w*l_arm)*g*np.sin(rad_angle)
    elif m == 'ext':
        torque = - (((arm_weight + w)*lcm_arm**2 + ((1/3)*(arm_weight + w)*l_arm**2))*acel + (arm_weight*lcm_arm + w*l_arm)*g*np.sin(rad_angle))
    elif m == 'pron':
        torque = np.sin(rad_angle)
    elif m == 'sup':
        torque = np.sin(rad_angle)

    return torque

# Apply RMS to files
def featureExtraction(files, features=['rms']):

    featuredFiles = {}

    for subject, data in files.items():

        for test, values in data.items():

            if test != 'subject_info':

                d_rms = np.array([])
                d_mav = np.array([])
                d_zc = np.array([])
                d_wl = np.array([])
                d_mdf = np.array([])
                d_mnf = np.array([])
                d_mom = np.array([])

                featureData = []

                for w in values[0]:

                    if 'rms' in features:
                        d_rms = rms(w)
                    if 'mav' in features:
                        d_mav = mav(w)
                    if 'zc' in features:
                        d_zc = zeroCrossing(w)
                    if 'wl' in features:
                        d_wl = wl(w)
                    if 'mdf' in features:
                        d_mdf = mdf(w)
                    if 'mnf' in features:
                        d_mnf = mnf(w)
                    if 'mom' in features:
                        d_mom = mom(w, 4)

                    d_features = np.concatenate((d_rms, d_mav, d_zc, d_wl, d_mdf, d_mnf, d_mom))
                    featureData.append(d_features)

            data[test][0] = np.array(featureData)

        featuredFiles[subject] = data


    return featuredFiles




