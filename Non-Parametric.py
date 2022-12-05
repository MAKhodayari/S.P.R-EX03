import utilities as utl
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_classes = 3
    dataset_class_size = 500
    # Dataset  Data Preparation
    dataset_mu = [[2, 5], [8, 1], [5, 3]]
    dataset_sigma = [[[2, 0], [0, 2]], [[3, 1], [1, 3]], [[2, 1], [1, 2]]]

    dataset = utl.generate_dataset(dataset_mu, dataset_sigma, dataset_classes, dataset_class_size)
    #plot histogram

    train_1, test_1 = train_test_split(dataset, test_size=0.2)
    H = [0.09, 0.3, 0.6]
    x=1
    train_smapels=train_1.shape[0]
    pdf_c_fig = plt.figure(figsize=(10, 10))
    for h in H : 
        ax = pdf_c_fig.add_subplot(2, 2, x, projection='3d')
        hist, X1_bin,X2_bin=utl.histogram(train_1,h)
        utl.plot_histogram(hist, X1_bin,X2_bin,"histogram",ax,h)
        #utl.plot_density1(test_1,train_smapels,hist,X1_bin,X2_bin,"histogram",ax2,h)
        x +=1
    
    x=1
    train_smapels=train_1.shape[0]
    pdf_c_fig = plt.figure(figsize=(10, 10))
    for h in H : 
        ax2 = pdf_c_fig.add_subplot(2, 2, x, projection='3d')
        hist, X1_bin,X2_bin=utl.histogram(train_1,h)
        utl.plot_density1(test_1,train_smapels,hist,X1_bin,X2_bin,"histogram",ax2,h)
        x +=1
    pdf_c_fig.tight_layout()
    plt.show()
    
    # plot KNN

    utl.plot_pdf_pw_gaus(dataset,'KNN',0,'KNN ',0)

    # plot parzenWindowd
    utl.plot_pdf_pw_gaus(dataset, 'parzenWindowd', 0, 'parzenWindowd', 1)

    # plot gaussiankernel

    utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.2,'gaussiankernel-sigma(0.2)',1)
    utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.6,'gaussiankernel-sigma(0.6)',1)
    utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.9,'gaussiankernel-sigma(0.9)',1)