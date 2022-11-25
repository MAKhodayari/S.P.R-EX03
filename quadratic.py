import utilities as utl
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X_11, y_11 = utl.generate_data([3, 6], [[1.5, 0], [0, 1.5]], 1, 500)
    X_12, y_12 = utl.generate_data([5, 4], [[2, 0], [0, 2]], 2, 500)
    X_13, y_13 = utl.generate_data([6, 6], [[1, 0], [0, 1]], 3, 500)

    X_train_11, X_test_11, y_train_11, y_test_11 = train_test_split(X_11, y_11, test_size=0.2)
    X_train_12, X_test_12, y_train_12, y_test_12 = train_test_split(X_12, y_12, test_size=0.2)
    X_train_13, X_test_13, y_train_13, y_test_13 = train_test_split(X_13, y_13, test_size=0.2)

    X_21, y_21 = utl.generate_data([3, 6], [[1.5, 0.1], [0.1, 0.5]], 1, 500)
    X_22, y_22 = utl.generate_data([5, 4], [[1, -0.2], [-0.2, 2]], 2, 500)
    X_23, y_23 = utl.generate_data([6, 6], [[2, -0.25], [-0.25, 1.5]], 3, 500)

    X_train_21, X_test_21, y_train_21, y_test_21 = train_test_split(X_21, y_21, test_size=0.2)
    X_train_22, X_test_22, y_train_22, y_test_22 = train_test_split(X_22, y_22, test_size=0.2)
    X_train_23, X_test_23, y_train_23, y_test_23 = train_test_split(X_23, y_23, test_size=0.2)
