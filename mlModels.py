##########################################################
##### SKLEARN AND TENSORFLOW MACHINE LEARNING MODELS #####
##########################################################

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPool2D, Flatten, GRU, attention, Bidirectional

# Sklearn machine learning models
def sklearnModelsClassif():

    nb = GaussianNB()
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    lda = LinearDiscriminantAnalysis()
    lr = LogisticRegression(max_iter=500)
    mlp = MLPClassifier(max_iter=500)
    svm = SVC()
    bg = BaggingClassifier()
    ab = AdaBoostClassifier()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    clfs = [
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier())
    ]
    st = StackingClassifier(clfs, final_estimator=LogisticRegression())

    models_names_classif = ['Naive Bayes', 'Decision Tree', 'KNN', 'LDA', 'Logistic Regression', 'MLP', 'SVM', 'Bagging', 'AdaBoost', 
                    'Gradient Boosting', 'Random Forest', 'Stacking']
    models_classif = [nb, dt, knn, lda, lr, mlp, svm, bg, ab, gb, rf, st]

    return models_classif, models_names_classif

# Multi layer perceptron
def mlp():

    mlp_model = Sequential()
    mlp_model.add(Dense(32, activation='relu'))
    mlp_model.add(Dense(16, activation='relu'))
    mlp_model.add(Dense(5, activation='softmax'))

    return mlp_model

# Convolutional neural network
def cnn():

    cnn_model = Sequential()
    cnn_model.add(Conv2D(16, (4,4), activation='relu', padding='same'))
    cnn_model.add(MaxPool2D((2,2), padding='same'))
    cnn_model.add(Conv2D(8, (4,4), activation='relu', padding='same'))
    cnn_model.add(MaxPool2D((2,2), padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(8, activation='relu'))
    cnn_model.add(Dense(5, activation='softmax'))

    return cnn_model

# Recurrent neural network
def lstm_rnn():

    lstm_model = Sequential()
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(LSTM(16))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(5, activation='softmax'))

    return lstm_model

def gru_rnn():

    gru_model = Sequential()
    gru_model.add(Dense(32, activation='relu'))
    gru_model.add(GRU(16))
    gru_model.add(Dense(32, activation='relu'))
    gru_model.add(Dense(5, activation='softmax'))

    return gru_model
