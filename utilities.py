import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal


def normalize(X):
    NX = pd.DataFrame(columns=X.columns.values)
    for column in NX.columns:
        X_max = X[column].max()
        X_min = X[column].min()
        X_range = X_max - X_min
        if X_range != 0:
            NX[column] = (X[column] - X_min) / X_range
        else:
            NX[column] = X[column] / X_max
    return NX


def open_bayesian():
    headers = ['X1', 'X2', 'y']

    train_1 = pd.read_csv('./dataset/BC-Train1.csv', names=headers)
    test_1 = pd.read_csv('./dataset/BC-Test1.csv', names=headers)

    train_2 = pd.read_csv('./dataset/BC-Train2.csv', names=headers)
    test_2 = pd.read_csv('./dataset/BC-Test2.csv', names=headers)

    return train_1, test_1, train_2, test_2


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


def calc_cov(x, y):
    m_sample = len(x)
    mu_x = sum(x) / len(x)
    mu_y = sum(y) / len(y)
    normal_x = [i - mu_x for i in x]
    normal_y = [i - mu_y for i in y]
    cov = sum([normal_x[i] * normal_y[i] for i in range(m_sample)]) / m_sample
    return cov


def calc_sigma(data):
    c_class = len(np.unique(data.y))
    _, n_feature = data.iloc[:, :-1].shape
    sigma = np.zeros((c_class, n_feature, n_feature))
    for i in range(c_class):
        class_data = data[data.y == i].iloc[:, :-1].values.T
        sigma[i] = [[calc_cov(a, b) for a in class_data] for b in class_data]
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
            prob.append(- 0.5 * (np.dot(np.dot(x - mu[c], sigma_inv[c]), x - mu[c]) +
                                 np.log(np.linalg.det(sigma[c]))) + np.log(phi[c]))
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


def generate_clss_data(mu, sigma, c, size):
    X = np.random.multivariate_normal(mu, sigma, size)
    data = np.insert(X, 2, c, axis=1)
    return data


def generate_dataset(mu, sigma, c_class, c_size):
    dataset = []
    for i in range(c_class):
        dataset = np.append(dataset, generate_clss_data(mu[i], sigma[i], i, c_size))
    dataset = dataset.reshape((c_class * c_size, 3))
    dataset = pd.DataFrame(dataset, columns=['X1', 'X2', 'y'])
    dataset = dataset.astype({'y': 'int'})
    return dataset


def plot_linear_boundary(data, data_pred, phi, mu, sigma, ax, title):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    sigma_inv = np.linalg.inv(sigma)
    a = np.dot(sigma_inv[0], mu[1] - mu[0])
    b = (0.5 * (np.dot(np.dot(mu[0].T, sigma_inv[0]), mu[0]) - np.dot(np.dot(mu[1].T, sigma_inv[0]), mu[1]))) +\
        np.log(phi[0] / phi[1])
    decision_boundary = - (b + np.dot(a[0], X[:, 0])) / a[1]
    ax.scatter(X[(np.where((y == 0) & (data_pred == 0))), 0], X[(np.where((y == 0) & (data_pred == 0))), 1],
               marker='.', color='m', label='0 as 0')
    ax.scatter(X[(np.where((y == 1) & (data_pred == 1))), 0], X[(np.where((y == 1) & (data_pred == 1))), 1],
               marker='.', color='c', label='1 as 1')
    ax.scatter(X[(np.where((y == 0) & (data_pred == 1))), 0], X[(np.where((y == 0) & (data_pred == 1))), 1],
               marker='.', color='r', label='0 as 1')
    ax.scatter(X[(np.where((y == 1) & (data_pred == 0))), 0], X[(np.where((y == 1) & (data_pred == 0))), 1],
               marker='.', color='k', label='1 as 0')
    ax.plot(X[:, 0], decision_boundary)
    ax.set(xlabel='X[X1]', ylabel='X[X2]')
    ax.legend(loc='upper left')
    ax.set_title(title)
    return True


def plot_pdf(mu, sigma, ax, x_bound, y_bound, color, title, n=100):
    c_class, n_feature = mu.shape
    for i in range(c_class):
        x, y = np.meshgrid(np.linspace(x_bound[0], x_bound[1], n), np.linspace(y_bound[0], y_bound[1], n))
        two_pair = np.dstack((x, y))
        z = multivariate_normal.pdf(two_pair, mu[i], sigma[i])
        ax.contour3D(x, y, z, 100, cmap=color[i])
        ax.set_title(title)
    ax.set(xlabel='X[X1]', ylabel='X[X2]', zlabel='PDF')
    return True


def plot_contour(mu, sigma, ax, x_bound, y_bound, color, title, n=100):
    c_class, n_feature = mu.shape
    for i in range(c_class):
        x, y = np.meshgrid(np.linspace(x_bound[0], x_bound[1], n), np.linspace(y_bound[0], y_bound[1], n))
        two_pair = np.dstack((x, y))
        z = multivariate_normal.pdf(two_pair, mu[i], sigma[i])
        ax.contour(x, y, z, 10, cmap=color[i])
        ax.set_title(title)
    ax.set(xlabel='X[X1]', ylabel='X[X2]')
    return True


def plot_quadratic_boundary(data, data_pred, phi, mu, sigma, ax, title):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    c_class = len(np.unique(y))
    sigma_inv = np.linalg.inv(sigma)
    for i in range(c_class - 1):
        for j in range(i + 1, c_class):
            a = - 0.5 * (sigma_inv[i] - sigma_inv[j])
            b = np.dot(sigma_inv[i], mu[i]) - np.dot(sigma_inv[j], mu[j])
            c = 0.5 * (np.dot(np.dot(mu[j].T, sigma_inv[j]), mu[j]) -
                       np.dot(np.dot(mu[i].T, sigma_inv[i]), mu[i]) -
                       np.log(np.linalg.det(sigma[i]) / np.linalg.det(sigma[j]))) +\
                np.log(phi[i] / phi[j])
    pass


def density_estimate(data_train, method, x, h, sigma):
    N, D = data_train.shape
    const = 1 / (N * pow(h, D))
    sum = 0
    for i in data_train:
        prob = 1
        for j in range(D):
            if method == 'parzen_window':
                if np.abs((i[j] - x[j]) / h) <= 0.5:
                    prob *= 1
                else:
                    prob *= 0
            elif method == 'gaussian_kernel':
                prob *= (1 / (math.sqrt(2 * math.pi) * sigma)) *\
                        math.exp(-(((i[j] - x[j]) / h) ** 2) / (2 * sigma ** 2))
        sum += prob
    return const * sum


def plot_pdf_non_param(data, method, sigma, title):
    H = [0.09, 0.3, 0.6]
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
    x, y = np.meshgrid(x, y)
    prob = []
    for h in H:
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                pair = [x[i, j], y[i, j]]
                z[i, j] = density_estimate(data, method, pair, h, sigma)
        prob.append(z)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(title)
    for i in range(len(H)):
        ax = fig.add_subplot(1, len(H), i + 1, projection='3d')
        ax.plot_surface(x, y, prob[i], cmap='plasma')
        ax.set_title('H =  ' + str(H[i]))
        ax.set(xlabel='X[X1]', ylabel='X[X2]', zlabel='P(X)')
    fig.tight_layout()
    plt.show()
