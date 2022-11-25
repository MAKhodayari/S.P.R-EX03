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
    classes, counts = np.unique(data.y, return_counts=True)
    c_class = len(classes)
    _, n_feature = data.iloc[:, :-1].shape
    mu = np.zeros((c_class, n_feature))
    for i in classes:
        mu[i] = np.sum(np.array(data[data.y == i].iloc[:, :-1]), axis=0) / counts[i]
    return mu


def calc_sigma(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    m_sample, n_feature = X.shape
    mu = calc_mu(data)
    sigma = np.zeros((n_feature, n_feature))
    for i in range(m_sample):
        X_mu = X[i] - mu[y[i]]
        sigma += np.dot(X_mu.reshape(n_feature, -1), (X_mu.reshape(-1, n_feature)))
    sigma /= m_sample
    return sigma


def calc_params(data):
    phi = calc_phi(data.y)
    mu = calc_mu(data)
    sigma = calc_sigma(data)
    return phi, mu, sigma


def bayesian_prediction(data, phi, mu, sigma):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    c_class = len(np.unique(y))
    sigma_inv = np.linalg.inv(sigma)
    probs = []
    for x in X:
        prob = []
        for c in range(c_class):
            prob.append(- 0.5 * (np.dot(np.dot(x - mu[c], sigma_inv), x - mu[c]) + np.log(np.linalg.norm(sigma))) + np.log(phi[c]))
        probs.append(prob)
    yh = np.argmax(probs, axis=1)
    return yh


def calc_accuracy(y, yh):
    m_sample = len(y)
    correct = 0
    for i in range(m_sample):
        if yh[i] == y[i]:
            correct += 1
    acc = correct / m_sample
    return acc


def calc_scores(conf_mat):
    c_class = len(conf_mat)
    tp, tn, fp, fn = np.zeros(c_class, int), np.zeros(c_class, int), np.zeros(c_class, int), np.zeros(c_class, int)
    scores = np.zeros((c_class, 4))
    for i in range(c_class):
        tp[i] = conf_mat[i][i]
        tn[i] = np.sum(np.delete(np.delete(conf_mat, i, 0), i, 1))
        fp[i] = np.sum(np.delete(conf_mat[i, :], i))
        fn[i] = np.sum(np.delete(conf_mat[:, i], i, 0))
        scores[i][0] = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
        scores[i][1] = tp[i] / (tp[i] + fp[i])
        scores[i][2] = tp[i] / (tp[i] + fn[i])
        scores[i][3] = (2 * tp[i]) / ((2 * tp[i]) + fp[i] + fn[i])
    return scores


def confusion_score_matrix(label, pred):
    unique = np.unique(label)
    c_class = len(unique)
    label_index, pred_index = [], []
    conf_mat = np.zeros((c_class, c_class), int)
    for i in range(c_class):
        label_index.append(np.where(label == i)[0])
        pred_index.append(np.where(pred == i)[0])
    for i in range(c_class):
        for j in range(c_class):
            conf_mat[i][j] = len(np.intersect1d(pred_index[i], label_index[j]))
    score_mat = calc_scores(conf_mat)

    class_name = []
    for c in list(map(str, unique)):
        class_name.append('Class ' + c)

    conf_mat = pd.DataFrame(conf_mat, index=class_name, columns=class_name)
    score_mat = pd.DataFrame(score_mat, index=class_name, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    return conf_mat, score_mat


def generate_data(mu, sigma, c, size):
    X = np.random.multivariate_normal(mu, sigma, size)
    y = np.array([c for _ in range(size)])
    return X, y