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

    # plot KNN

    # utl.plot_pdf_pw_gaus(dataset,'KNN',0,'KNN ',0)
    # plot parzenWindowd
    utl.plot_pdf_pw_gaus(dataset, 'parzenWindowd', 0, 'parzenWindowd', 1)

    # plot gaussiankernel

    # utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.2,'gaussiankernel-sigma(0.2)')
    # utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.6,'gaussiankernel-sigma(0.6)')
    # utl.plot_pdf_pw_gaus(dataset,'gaussiankernel',0.9,'gaussiankernel-sigma(0.9)')