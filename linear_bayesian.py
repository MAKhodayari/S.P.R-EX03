import utilities as utl
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Opening & Preparing Data
    train_1, test_1, train_2, test_2 = utl.open_bayesian()

    # Train Phase Of Dataset 1
    dataset_1_phi, dataset_1_mu, dataset_1_sigma = utl.calc_params(train_1)

    # Test Phase Of Dataset 1
    train_1_pred = utl.bayesian_prediction(train_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    train_1_conf_mat, train_1_score_mat = utl.confusion_score_matrix(train_1.y, train_1_pred)
    train_1_acc = utl.calc_accuracy(train_1.y, train_1_pred)

    test_1_pred = utl.bayesian_prediction(test_1, dataset_1_phi, dataset_1_mu, dataset_1_sigma)
    test_1_conf_mat, test_1_score_mat = utl.confusion_score_matrix(test_1.y, test_1_pred)
    test_1_acc = utl.calc_accuracy(test_1.y, test_1_pred)

    # Train Phase Of Dataset 2
    dataset_2_phi, dataset_2_mu, dataset_2_sigma = utl.calc_params(train_2)

    # Test Phase Of Dataset 2
    train_2_pred = utl.bayesian_prediction(train_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    train_2_conf_mat, train_2_score_mat = utl.confusion_score_matrix(train_2.y, train_2_pred)
    train_2_acc = utl.calc_accuracy(train_2.y, train_2_pred)

    test_2_pred = utl.bayesian_prediction(test_2, dataset_2_phi, dataset_2_mu, dataset_2_sigma)
    test_2_conf_mat, test_2_score_mat = utl.confusion_score_matrix(test_2.y, test_2_pred)
    test_2_acc = utl.calc_accuracy(test_2.y, test_2_pred)

    # Plots
    # Data Based On Label & Being Correct Or Not
    db_fig, db_axs = plt.subplots(2, 2, figsize=(10.5, 7))
    db_fig.suptitle('Bayesian Classifier With Linear Boundary')
    utl.plot_raw_data(train_1, train_1_pred, 'linear', dataset_1_phi, dataset_1_mu, dataset_1_sigma,
                      db_axs[0, 0], "BC-Train1")
    utl.plot_raw_data(test_1, test_1_pred, 'linear', dataset_1_phi, dataset_1_mu, dataset_1_sigma,
                      db_axs[1, 0], "BC-Test1")
    utl.plot_raw_data(train_2, train_2_pred, 'linear', dataset_2_phi, dataset_2_mu, dataset_2_sigma,
                      db_axs[0, 1], "BC-Train2")
    utl.plot_raw_data(test_2, test_2_pred, 'linear', dataset_2_phi, dataset_2_mu, dataset_2_sigma,
                      db_axs[1, 1], "BC-Test2")
    db_fig.tight_layout()

    # PDF & Contour
    dataset_1_x_bound, dataset_1_y_bound = [-3, 9], [-5, 11]
    dataset_2_x_bound, dataset_2_y_bound = [-4, 7], [-4, 7]
    color_map = ['summer', 'autumn']

    pdf_c_fig = plt.figure(figsize=(10, 10))
    pdf_c_fig.suptitle('PDF & Contour Plots')

    pdf_c_axs_pdf1 = pdf_c_fig.add_subplot(2, 2, 1, projection='3d')
    utl.plot_pdf(dataset_1_mu, dataset_1_sigma, pdf_c_axs_pdf1, dataset_1_x_bound, dataset_1_y_bound,
                 color_map, 'Dataset 1 (BC1) PDF')

    pdf_c_axs_pdf2 = pdf_c_fig.add_subplot(2, 2, 2, projection='3d')
    utl.plot_pdf(dataset_2_mu, dataset_2_sigma, pdf_c_axs_pdf2, dataset_2_x_bound, dataset_2_y_bound,
                 color_map, 'Dataset 2 (BC2) PDF')

    pdf_c_axs_c1 = pdf_c_fig.add_subplot(2, 2, 3)
    utl.plot_contour(test_1.iloc[:, :-1].values, dataset_1_phi, dataset_1_mu, dataset_1_sigma, pdf_c_axs_c1,
                     dataset_1_x_bound, dataset_1_y_bound,
                     color_map, 'linear', 'Dataset 1 (BC1) Contour')

    pdf_c_axs_c2 = pdf_c_fig.add_subplot(2, 2, 4)
    utl.plot_contour(test_2.iloc[:, :-1].values, dataset_2_phi, dataset_2_mu, dataset_2_sigma, pdf_c_axs_c2,
                     dataset_2_x_bound, dataset_2_y_bound,
                     color_map, 'linear', 'Dataset 2 (BC2) Contour')

    pdf_c_fig.tight_layout()

    plt.show()

    # Results
    print('─' * 50)

    print('Results:')

    print('─' * 50)

    print('Dataset 1 Parameters:\n')
    print(f'Phi:\n{dataset_1_phi}\n\nMu:\n{dataset_1_mu}\n\nSigma:\n{dataset_1_sigma}')

    print('─' * 50)

    print(f'BC-Train1 Confusion Matrix:\n{train_1_conf_mat}\n')
    print(f'BC-Train1 Score Matrix:\n{train_1_score_mat}\n')
    print(f'BC-Train1 Accuracy: {train_1_acc}')

    print('─' * 50)

    print(f'BC-Test1 Confusion Matrix:\n{test_1_conf_mat}\n')
    print(f'BC-Test1 Score Matrix:\n{test_1_score_mat}\n')
    print(f'BC-Test1 Accuracy: {test_1_acc}')

    print('─' * 50)

    print('Dataset 2 Parameters:\n')
    print(f'Phi:\n{dataset_2_phi}\n\nMu:\n{dataset_2_mu}\n\nSigma:\n{dataset_2_sigma}')

    print('─' * 50)

    print(f'BC-Train2 Confusion Matrix:\n{train_2_conf_mat}\n')
    print(f'BC-Train2 Score Matrix:\n{train_2_score_mat}\n')
    print(f'BC-Train2 Accuracy: {train_2_acc}')

    print('─' * 50)

    print(f'BC-Test2 Confusion Matrix:\n{test_2_conf_mat}\n')
    print(f'BC-Test2 Score Matrix:\n{test_2_score_mat}\n')
    print(f'BC-Test2 Accuracy: {test_2_acc}')

    print('─' * 50)
