import numpy as np
from numpy.lib.function_base import _parse_input_dimensions
import pandas as pd
from sklearn import svm
import sklearn
from sklearn import metrics
from sklearn import dummy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras import Sequential
from keras.layers import Dense
from sklearn.dummy import DummyClassifier


def load_data(fname):
    df = pd.read_excel(fname)
    df = df.dropna(subset=['BRAIN'])
    return df

def train_net(df):
    data = df[['D0A0', 'D0A6 and D2A0','D0S0', 'D0S6 and D2S0', 'D2A6', 'D2S6', 'D0A6']]
    # data = df[['D0A0', 'D0A6 and D2A0', 'D0S6 and D2S0', 'D2S6', 'D0A6']]
    data = data.to_numpy()
    labels = df['Y1A2PD3']
    labels=labels.to_numpy(dtype=int)
    # labels=np.maximum(labels,2)
    # labels=labels%2

    sc = StandardScaler()
    data = sc.fit_transform(data)
    labels_ohe = np.zeros((len(data), 3))
    for i in range(len(labels_ohe)):
        labels_ohe[i,labels[i]-1] = 1
    # print(labels_ohe)
    X_train, X_test, y_train, y_test = train_test_split(data, labels_ohe, test_size=0.2, random_state=5)
    numepochs=1000
    acc = np.zeros((4,7, numepochs))
    acc_test = np.zeros((4,7))
    for nl in [2,3,4,5]:
        for nn in range(8,22,2):
            print(f'Number of perceptrons per layer: {nn}')
            model=Sequential()
            model.add(Dense(nn, input_dim=7, activation='sigmoid'))
            model.add(Dense(nn, activation='sigmoid'))
            model.add(Dense(nn, activation='sigmoid'))
            model.add(Dense(nn, activation='sigmoid'))
            if nl>=5:
                model.add(Dense(nn, activation='sigmoid'))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history=model.fit(X_train, y_train, epochs=numepochs, batch_size=8, verbose=0)

            y_pred = model.predict(X_test)
            y_pred_bool = np.argmax(y_pred, axis=1)
            print(y_pred_bool)
            pred = []
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))

            test = []
            for i in range(len(y_test)):
                test.append(np.argmax(y_test[i]))
            a = accuracy_score(pred,test)
            print('Accuracy is:', a*100)

            print(classification_report(np.argmax(y_test, axis=1), y_pred_bool))
            acc[nl-2,nn//2-4] = history.history['accuracy']
            acc_test[nl-2, nn//2-4] = a
            # plt.plot(history.history['accuracy'])
            # plt.show()
    np.savez('results3.npz', acc=acc, acc_test=acc_test)

def train_classifier(df):
    data = df[['D0A0', 'D0A6 and D2A0','D0S0', 'D0S6 and D2S0', 'D2A6', 'D2S6', 'D0A6']]
    data = data.to_numpy()
    labels = df['Y1A2PD3']
    labels=labels.to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

    X_train = scale(X_train)
    X_test = scale(X_test)
    kernels = ['poly', 'rbf', 'linear', 'sigmoid']

    dc = DummyClassifier(strategy='constant', constant=3)
    dc.fit(X_train, y_train)
    dummies = np.zeros(100)
    for i in range (100):
        preds = dc.predict(X_test)
        # print(classification_report(y_test, preds))
        dummies[i]=precision_recall_fscore_support(y_test, preds, average='weighted')[2]
    print(dummies)
    print(np.std(dummies))
    print(np.mean(dummies))

    # for i in range(4):
    #     print(f'Kernel: {kernels[i]}')
    #     clf = svm.SVC(kernel=kernels[i])
    #     clf.fit(X_train, y_train)

    #     preds = clf.predict(X_test)
    #     # prob_preds = clf.predict_proba(X_test)
    #     # print(prob_preds)
    #     print(preds)
    #     print(y_test)
    #     print(classification_report(y_test, preds))

data=load_data('data.xlsx')
# train_classifier(data)
train_net(data)