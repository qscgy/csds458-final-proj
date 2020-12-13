import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def load_data(fname):
    df = pd.read_excel(fname)
    df = df.dropna(subset=['BRAIN'])
    return df

def train_classifier(df):
    data = df[['D0A0', 'D0A6 and D2A0','D0S0', 'D0S6 and D2S0', 'D2A6', 'D2S6', 'D0A6']]
    # data = df[['D0A0', 'D0A6 and D2A0', 'D0S6 and D2S0', 'D2S6', 'D0A6']]
    data = data.to_numpy()
    labels = df['Y1A2PD3']
    labels=labels.to_numpy(dtype=int)
    # labels=np.maximum(labels,2)
    # labels=labels%2

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=5)

    pca=PCA(n_components=5)
    # X_train = pca.fit_transform(X_train)
    X_train = scale(X_train)
    # print(np.cumsum(pca.explained_variance_ratio_))
    # plt.figure()
    # for i in range(3):
    #     plt.scatter(data[labels==i,0], data[labels==i,1])
    # plt.show()
    # print(data)
    # print(labels)
    
    # clf = svm.SVC(kernel='poly')
    clf = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=1000, learning_rate_init=1e-2, activation='logistic')
    clf.fit(X_train, y_train)
    # rs: 1,2

    # X_test = pca.transform(X_test)
    X_test = scale(X_test)
    preds = clf.predict(X_test)
    prob_preds = clf.predict_proba(X_test)
    print(prob_preds)
    print(preds)
    print(y_test)
    print(confusion_matrix(y_test, preds))

    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(data)
    # print(type(tsne_results))
    # plt.figure()
    # for i in range(3):
    #     plt.scatter(tsne_results[labels==i,0],tsne_results[labels==i,1])
    # plt.show()

data=load_data('data.xlsx')
train_classifier(data)