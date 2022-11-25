import utilities as utl


if __name__ == '__main__':
    # Opening & Preparing Data
    train_1, train_2, test_1, test_2 = utl.open_bayesian()

    # Train Phase Of Dataset 1
    dataset_1_phi, dataset_1_mu, dataset_1_sigma = utl.calc_params(train_1)

    # Test Phase Of Dataset 1
    train_1_pred = utl.bayesian_prediction(train_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    train_1_acc = utl.calc_accuracy(train_1.y, train_1_pred)
    train_1_conf_mat, train_1_score_mat = utl.confusion_score_matrix(train_1.y, train_1_pred)

    test_1_pred = utl.bayesian_prediction(test_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    test_1_acc = utl.calc_accuracy(test_1.y, test_1_pred)
    test_1_conf_mat, test_1_score_mat = utl.confusion_score_matrix(test_1.y, test_1_pred)

    # Train Phase Of Dataset 2
    dataset_2_phi, dataset_2_mu, dataset_2_sigma = utl.calc_params(train_2)

    # Test Phase Of Dataset 1
    train_2_pred = utl.bayesian_prediction(train_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    train_2_acc = utl.calc_accuracy(train_2.y, train_2_pred)
    train_2_conf_mat, train_2_score_mat = utl.confusion_score_matrix(train_2.y, train_2_pred)

    test_2_pred = utl.bayesian_prediction(test_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    test_2_acc = utl.calc_accuracy(test_1.y, test_1_pred)
    test_2_conf_mat, test_2_score_mat = utl.confusion_score_matrix(test_2.y, test_2_pred)

    # Results
    print('Results:')

    print('─' * 50)

    print(f'BC-Train1 Confusion Matrix:\n{train_1_conf_mat.round(2)}\n')
    print(f'BC-Train1 Score Matrix:\n{train_1_score_mat.round(2)}\n')
    print(f'BC-Train1 Accuracy: {round(train_1_acc * 100, 2)}')

    print('─' * 50)

    print(f'BC-Test1 Confusion Matrix:\n{test_1_conf_mat.round(2)}\n')
    print(f'BC-Test1 Score Matrix:\n{test_1_score_mat.round(2)}\n')
    print(f'BC-Test1 Accuracy: {round(test_1_acc * 100, 2)}')

    print('─' * 50)

    print(f'BC-Train2 Confusion Matrix:\n{train_2_conf_mat.round(2)}\n')
    print(f'BC-Train2 Score Matrix:\n{train_2_score_mat.round(2)}\n')
    print(f'BC-Train2 Accuracy: {round(train_2_acc * 100, 2)}')

    print('─' * 50)

    print(f'BC-Test2 Confusion Matrix:\n{test_2_conf_mat.round(2)}\n')
    print(f'BC-Test2 Score Matrix:\n{test_2_score_mat.round(2)}\n')
    print(f'BC-Test2 Accuracy: {round(test_2_acc * 100, 2)}')
