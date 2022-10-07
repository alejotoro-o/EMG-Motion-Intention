###############################
##### EMG SIGNAL FEATURES #####
###############################

import numpy as np
from scipy.signal import welch
from scipy.stats import moment


### Time domain features ###

# Root Mean Square (RMS)
def rms(data):

    k = data.shape[0]
    rms_data = np.sqrt(np.sum(np.square(data)/k, axis=0))

    return rms_data

# Mean Absolute Value (MAV)
def mav(data):

    k = data.shape[0]
    mav_data = np.sum(np.absolute(data), axis=0)/k

    return mav_data  

# Zero Crosssing (ZC)
def zeroCrossing(data, T=0):

    k = data.shape[0]
    chs = data.shape[1]
    zc_data = np.zeros((chs))

    for i in range(chs):
        for j in range(k-1):
            if np.sign(data[j,i])*np.sign(data[j+1,i]) == -1 and (np.absolute(data[j,i]) >= T and np.absolute(data[j+1,i]) >= T):
                zc_data[i] = zc_data[i] + 1
            
    return zc_data

# Waveform Length (WL)
def wl(data):

    k = data.shape[0]
    diff = np.zeros((k-1,data.shape[1]))

    for i in range(k-1):

        diff[i,:] = data[i+1,:] - data[i,:]

    wl_data = np.sum(np.absolute(diff), axis=0)

    return wl_data

# Moments
def mom(data, moments=2):

    m_data = np.empty((0))

    for i in range(moments + 1):
        m_data = np.append(m_data, moment(data, moment=i, axis=0))

    return m_data

### Frequency domain features ###

# Median Frequency (MDF)
def mdf(data, fs = 1024):

    psd = welch(data, fs, axis=0)
    mdf_data = (1/2)*np.sum(psd[1],axis=0)

    return mdf_data

# Mean Frequency
def mnf(data, fs = 1024):

    psd = welch(data, fs, axis=0)
    mnf_data = np.divide(np.sum(np.multiply(psd[0].reshape((psd[0].shape[0],1)), psd[1]), axis=0), np.sum(psd[1],axis=0))

    return mnf_data