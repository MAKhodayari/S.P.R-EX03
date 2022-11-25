import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def open_bayesian():
    headers = ['X1', 'X2', 'y']

    train_1 = pd.read_csv('./dataset/BC-Train1.csv', names=headers)
    train_2 = pd.read_csv('./dataset/BC-Train2.csv', names=headers)

    test_1 = pd.read_csv('./dataset/BC-Test1.csv', names=headers)
    test_2 = pd.read_csv('./dataset/BC-Test2.csv', names=headers)

    return train_1, train_2, test_1, test_2


def calc_phi(labels):
    _, counts = np.unique(labels, return_counts=True)
    phi = counts / len(labels)
    return phi


def calc_mu(data):
    classes = np.unique(data.y)
    c_class = len(classes)
    _, n_feature = data.iloc[:, :-1].shape
    mu = np.zeros((c_class, n_feature))
    for i in classes:
        mu[i] = np.array(data[data.y == i].iloc[:, :-1].mean())
    return mu


#def calc_sigma(data):
#    sigma = np.cov(data.iloc[:, :-1], rowvar=False)
#   return sigma


def calc_sigma(data):
    x=data[['X1', 'X2']].values
    y=data[['y']].values
    mu=calc_mu(data)
    cov = np.zeros(shape=(x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        if y[i]==0:
            x_mu=x[i]-mu[0]
        else:
            x_mu=x[i]-mu[1]
            
        cov +=(x_mu).reshape(x.shape[1],1).dot((x_mu).reshape(1,x.shape[1]))
    cov = cov/x.shape[0]
    return cov

def Prediction(data, Mu, Sigma, Nclass):
    X=data[['X1','X2']].values
    y=data[['y']].values
    Probabilities = []
    p=calc_phi(y)
    for i in range(Nclass):
        Phi = ((p[i])**i * (1 - p[i])**(1 - i))
        mu = Mu[i, :]
        invSigma = np.linalg.pinv(Sigma)
        Probability = np.log(Phi) +(-0.5 * np.sum((X-mu).dot(invSigma)*(X-mu), axis=1))
        Probabilities.append(Probability)

    Classes = np.argmax(Probabilities, axis=0)
    return Classes, Probabilities


def calc_accuracy(y, yh):
    m_sample = len(y)
    correct = 0
    for i in range(m_sample):
        if yh[i] == y[i]:
            correct += 1
    acc = correct / m_sample
    return acc

    