import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal


# from scipy.optimize import curve_fit


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


def plot_linear_boundary(X, phi, mu, sigma, ax):
    sigma_inv = np.linalg.inv(sigma)
    a = np.dot(sigma_inv[0], (mu[1] - mu[0]))
    b = (0.5 * (np.dot(np.dot(mu[0].T, sigma_inv[0]), mu[0]) - np.dot(np.dot(mu[1].T, sigma_inv[0]), mu[1]))) + \
        np.log(phi[0] / phi[1])
    decision_boundary = - (b + np.dot(a[0], X[:, 0])) / a[1]
    ax.plot(X[:, 0], decision_boundary)


# def func(x, a, b, c):
#     return a * x * x + b * x + c


# def plot_quadratic_boundary(x_values, y_values, ax):
#     # non-linear least squares to fit func to data
#     p_opt, p_cov = curve_fit(func, x_values, y_values)
#     # these are the fitted values a, b, c
#     a, b, c = p_opt
#     # produce 100 values in the range we want to cover along x
#     x_fit = np.linspace(min(x_values), max(x_values), 100)
#     # compute fitted y values
#     y_fit = [func(x, a, b, c) for x in x_fit]
#     ax.plot(x_fit, y_fit)


# def plot_quadratic_boundary(data, data_pred, phi, mu, sigma, ax, title):
#     X = data.iloc[:, :-1].values
#     y = data.iloc[:, -1].values
#     c_class = len(np.unique(y))
#     sigma_inv = np.linalg.inv(sigma)
#     for i in range(c_class - 1):
#         for j in range(i + 1, c_class):
#             a = - 0.5 * (sigma_inv[i] - sigma_inv[j])
#             b = np.dot(sigma_inv[i], mu[i]) - np.dot(sigma_inv[j], mu[j])
#             c = 0.5 * (np.dot(np.dot(mu[j].T, sigma_inv[j]), mu[j]) -
#                        np.dot(np.dot(mu[i].T, sigma_inv[i]), mu[i]) -
#                        np.log(np.linalg.det(sigma[i]) / np.linalg.det(sigma[j]))) +\
#                 np.log(phi[i] / phi[j])
#     pass


def plot_raw_data(data, pred_label, method, phi, mu, sigma, ax, title):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if method == 'linear' or method == 'quadratic':
        ax.scatter(X[(np.where((y == 0) & (pred_label == 0))), 0], X[(np.where((y == 0) & (pred_label == 0))), 1],
                   marker='.', color='m', label='0 => 0')
        ax.scatter(X[(np.where((y == 1) & (pred_label == 1))), 0], X[(np.where((y == 1) & (pred_label == 1))), 1],
                   marker='.', color='c', label='1 => 1')
        ax.scatter(X[(np.where((y == 0) & (pred_label != 0))), 0], X[(np.where((y == 0) & (pred_label != 0))), 1],
                   marker='.', color='r', label='0 => ?')
        ax.scatter(X[(np.where((y == 1) & (pred_label != 1))), 0], X[(np.where((y == 1) & (pred_label != 1))), 1],
                   marker='.', color='k', label='1 => ?')
        if method == 'linear':
            plot_linear_boundary(X, phi, mu, sigma, ax)
    if method == 'quadratic':
        ax.scatter(X[(np.where((y == 2) & (pred_label == 2))), 0], X[(np.where((y == 2) & (pred_label == 2))), 1],
                   marker='.', color='green', label='2 => 2')
        ax.scatter(X[(np.where((y == 2) & (pred_label != 2))), 0], X[(np.where((y == 2) & (pred_label != 2))), 1],
                   marker='.', color='lime', label='2 => ?')
        # plot_quadratic_boundary(a[i][:, 0], a[i][:, 1], ax)
    ax.set(xlabel='X[X1]', ylabel='X[X2]')
    ax.legend(loc='upper left')
    ax.set_title(title)


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


def plot_contour(X, phi, mu, sigma, ax, x_bound, y_bound, color, method, title):
    c_class, n_feature = mu.shape
    for i in range(c_class):
        x, y = np.meshgrid(np.linspace(x_bound[0], x_bound[1], 100), np.linspace(y_bound[0], y_bound[1], 100))
        two_pair = np.dstack((x, y))
        z = multivariate_normal.pdf(two_pair, mu[i], sigma[i])
        ax.contour(x, y, z, 10, cmap=color[i])
        ax.set_title(title)
    if method == 'linear':
        plot_linear_boundary(X, phi, mu, sigma, ax)
    # elif method == 'quadratic':
    #     plot_quadratic_boundary(X[:, 0], X[:, 1], ax)
    ax.set(xlabel='X[X1]', ylabel='X[X2]')


def parzenWindowd(data_train, x, h, sigma, k):
    N, D = data_train.shape
    const = 1 / (N * pow(h, D))
    sum = 0
    for i in data_train:
        prob = 1
        for j in range(D):
            if np.abs((i[j] - x[j]) / h) <= 0.5:
                prob *= 1
            else:
                prob *= 0
        sum += prob
    return const * sum


def gaussiankernel(data_train, x, h, sigma, k):
    N, D = data_train.shape
    const = 1 / (N * pow(h, D))
    sum = 0
    for i in data_train:
        prob = 1
        for j in range(D):
            prob *= (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-(((i[j] - x[j]) / h) ** 2) / (2 * sigma ** 2))
        sum += prob
    return const * sum


def KNN(data_train, x, h, sigma, k):
    N, D = data_train.shape
    dist = np.array([math.dist(i, x) for i in data_train])
    R_k = np.sort(dist)[k - 1]
    v = math.pi * (R_k ** D)

    if v == 0:
        return 1
    else:
        return k / (N * v)


def histogram(data, h):
    X = data.iloc[:, :-1]
    # histogram
    X1_bin = np.arange(X['X1'].min(), X['X1'].max(), h)
    X2_bin = np.arange(X['X2'].min(), X['X2'].max(), h)
    hist, X1_bin, X2_bin = np.histogram2d(X['X1'], X['X2'], bins=(X1_bin, X2_bin))
    return hist, X1_bin, X2_bin


def plot_histogram(hist, X1_bin, X2_bin, title, ax, h):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(X1_bin[:-1] + 0.25, X2_bin[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title(title + '-h(' + str(h) + ')')
    # plt.show()
    return True


def plot_density1(test_data, train_samples, hist, X1_bin, X2_bin, titles, ax, h):
    XX = test_data['X1']
    YY = test_data['X2']
    XX, YY = np.meshgrid(XX, YY)
    Z = np.zeros(XX.shape)
    for i in range(XX.shape[0]):
        for j in range(YY.shape[0]):
            Z[i, j] = density(XX[i, j], YY[i, j], hist, X1_bin, X2_bin, h, train_samples)
    surf = ax.plot_surface(XX, YY, Z)
    ax.set_title(titles + '-h(' + str(h) + ')')
    return True


def density(x1, x2, hist, X1_bin, X2_bin, h, train_samples):
    X1_bin, X2_bin = X1_bin[1:], X2_bin[1:]
    for i, x in enumerate(X1_bin):
        if x1 < x:
            for j, y in enumerate(X2_bin):
                if x2 < y:
                    num_samples = hist[i, j]
                    bin_area = h * h
                    return 1 / train_samples * (num_samples / bin_area)


def plot_pdf_pw_gaus(data_train, func, sigma, title, flag):
    if func == 'gaussiankernel':
        func = gaussiankernel
    if func == 'parzenWindowd':
        func = parzenWindowd
    if func == 'KNN':
        func = KNN
    H = [0.09, 0.3, 0.6]
    k = [1, 9, 99]
    trn = data_train.iloc[:, :-1].values
    x = np.linspace(data_train['X1'].min(), data_train['X1'].max(), 40)
    y = np.linspace(data_train['X2'].min(), data_train['X2'].max(), 40)
    xx, yy = np.meshgrid(x, y)
    prob = []
    for h in range(len(H)):
        z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(yy.shape[0]):
                a = [xx[i, j], yy[i, j]]
                z[i, j] = func(trn, a, H[h], sigma, k[h])
        prob.append(z)
    pdf_c_fig = plt.figure(figsize=(10, 10))
    x = 1
    for i in range(len(prob)):
        ax = pdf_c_fig.add_subplot(2, 2, x, projection='3d')
        ax.plot_surface(xx, yy, prob[i], cmap='plasma')
        if flag == 0:
            ax.set_title(title + '-K(' + str(k[i]) + ')')
        else:
            ax.set_title(title + '-h(' + str(H[i]) + ')')
        ax.set(xlabel='X[X1]', ylabel='X[X2]', zlabel='P(x)')
        x += 1
    pdf_c_fig.tight_layout()
    plt.show()


def truePDF(data, mu, cov, phi):
    true_pdf = 0
    for i in range(len(mu)):
        true_pdf += (phi * multivariate_normal.pdf(data, mu[i], cov[i]))
    return true_pdf


def Mse(y,y_hat):
    diff=y-y_hat
    mse_pow=np.power(diff, 2,dtype='float64')
    mse = np.mean(mse_pow)
    return mse


def bset_h(data, mu, cov, phi, h_min, h_max, h_step, sigma):  
    data = data.iloc[:, :-1].values
    k_fold=np.array_split(data,5)
    min_error = 1e9
    best_h = 0
    for h in np.arange(h_min, h_max, h_step):
        print("h: ", h)
        folds_h = []
        MSE = []
        for fold in range(1, 6):
            print(fold)
            Range0 = (fold-1) * 300
            Range1 = fold * 300
            data_T=k_fold[fold-1]
            x = np.linspace(np.amin(data_T[:, 0]), np.amax(
            data_T[:, 0]), 20).reshape(-1, 1)
            y = np.linspace(np.amin(data_T[:, 1]), np.amax(
            data_T[:, 1]), 20).reshape(-1, 1)
            xx, yy = np.meshgrid(x, y)
            X_2d = np.concatenate(
                [xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
            
            data_train=np.delete(data, slice(Range0,Range1), axis=0)
            prob = []  
            for x in X_2d:
                px = gaussiankernel(data_train,x,h ,sigma,0)
                prob.append(px)
                
            true_density = truePDF(X_2d,mu, cov, phi)
            prob = np.array(prob)
            error  = Mse(true_density,prob)
            MSE.append(error)
        mse = np.sum(MSE) / 5        
        print("h =", round(h, 2), ": l2 =", mse)
        if mse < min_error:
            min_error = mse
            best_h = h  
    return best_h, min_error


def plot_truePDF(data,mu, cov, phi):
    data = data.iloc[:, :-1].values
    x = np.linspace(np.amin(data[:, 0]), np.amax(data[:, 0]), 20).reshape(-1, 1)
    y = np.linspace(np.amin(data[:, 1]), np.amax(data[:, 1]), 20).reshape(-1, 1)
    xx, yy = np.meshgrid(x, y)
    X_2d = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
    z = truePDF(X_2d, mu, cov, phi).reshape(xx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(xx, yy, z, cmap='plasma')
    ax.set_title('trupdf')
    fig.tight_layout()
    plt.show()
