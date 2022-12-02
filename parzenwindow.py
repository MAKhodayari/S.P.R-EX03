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
    utl.plot_parzenWindowd(dataset)

    