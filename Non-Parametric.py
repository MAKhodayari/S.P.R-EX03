import utilities as utl
from threading import Thread
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Shared Info Between Datasets
    dataset_classes = 3
    dataset_class_size = 500

    # Dataset  Data Preparation
    dataset_mu = [[2, 5], [8, 1], [5, 3]]
    dataset_sigma = [[[2, 0], [0, 2]], [[3, 1], [1, 3]], [[2, 1], [1, 2]]]

    dataset = utl.generate_dataset(dataset_mu, dataset_sigma, dataset_classes, dataset_class_size)

    # Plots
    data = dataset.iloc[:, :-1].values

    # Parzen Window
    utl.plot_pdf_non_param(data, 'parzen_window', 0, 'Parzen Window')

    # Gaussian Kernel
    utl.plot_pdf_non_param(data, 'gaussian_kernel', 0.2, 'Gaussian Kernel (Standard Deviation = 0.2)')
    utl.plot_pdf_non_param(data, 'gaussian_kernel', 0.6, 'Gaussian Kernel (Standard Deviation = 0.6)')
    utl.plot_pdf_non_param(data, 'gaussian_kernel', 0.9, 'Gaussian Kernel (Standard Deviation = 0.9)')
