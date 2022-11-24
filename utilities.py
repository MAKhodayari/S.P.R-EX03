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


def calc_sigma(data):
    sigma = np.cov(data.iloc[:, :-1], rowvar=False)
    return sigma
