import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn


def open_bayesian():
    headers = ['X1', 'X2', 'y']

    train_1 = pd.read_csv('./dataset/BC-Train1.csv', names=headers)
    test_1 = pd.read_csv('./dataset/BC-Test1.csv', names=headers)

    train_2 = pd.read_csv('./dataset/BC-Train2.csv', names=headers)
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
                                 np.log(np.linalg.norm(sigma[c]))) + np.log(phi[c]))
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

def plot_dec_boundary(data,prediction,phi,mu,sigma,title):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
   
    plt.plot(X[(np.where((y == 1) & (prediction==1)))[0],0],
            X[(np.where((y == 1) & (prediction==1)))[0],1], 'c.')
    plt.plot(X[(np.where((y == 0) & (prediction==0)))[0],0]
             ,X[(np.where((y == 0) & (prediction==0)))[0],1], 'm.')
    plt.plot(X[(np.where((y == 0) & (prediction==1)))[0],0]
            ,X[(np.where((y == 0) & (prediction==1)))[0],1], '.r')
    plt.plot(X[(np.where((y == 1) & (prediction==0)))[0],0]
            ,X[(np.where((y == 1) & (prediction==0)))[0],1], '.k')
    
    b0 = 0.5 * mu[0].T.dot(np.linalg.pinv(sigma[0])).dot(mu[0])
    b1 = -0.5 * mu[1].T.dot(np.linalg.pinv(sigma[1])).dot(mu[1])
    b = b0 + b1 + np.log(phi[0]/phi[1])
    a = np.linalg.pinv(sigma[0]).dot(mu[1] - mu[0])
    Decision_boundary= -(b + a[0]*X[:,0]) / a[1]
    plt.plot(X[:, 0], Decision_boundary)
    plt.title(title)
    plt.show()
    return

def Gaussian(X, mu, Sigma):
    const = 1/(np.sqrt(((np.pi)**2)*(np.linalg.det(Sigma))))
    Sigin = np.linalg.inv(Sigma)
    ans = const*np.exp(-0.5*(np.matmul((X-mu).T, np.matmul(Sigin,(X-mu)))))
    return ans  


def plot_PDF(data,phi,mu,sigma):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    c_class = len(np.unique(y))
    colors = [('b', 'plasma'), ('r', 'plasma')]   
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
   
    for i in range(c_class):
        Z=[]
        data=X[(np.where((y == i)))[0]]
        #XX, XY = np.meshgrid(data[:,0], data[:,1])
        [Z.append(Gaussian(j,mu[i],sigma)) for j in data]
        Z=np.array(Z)
        ax.scatter3D(X[y == i][:, 0], X[y == i][:, 1], np.ones(1) * -0.03, colors[i][0])
        ax.plot_trisurf(data[:,0],data[:,1],Z,cmap=colors[i][1],linewidth=2,alpha=0.9, shade=True)
    
    
    plt.title('3D PDFs ', fontsize=16)
    plt.show()
  
    
    

        
        
