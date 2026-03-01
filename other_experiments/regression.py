from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import argparse

def get_correlation(ground_truth, predicted):
    ground_truth = np.array(ground_truth).flatten()
    predicted = np.array(predicted).flatten()
    return np.corrcoef(ground_truth, predicted)[0][1]

# Train the Ridge regression model with grid search on regularization parameter alpha
def ridge_regression(train_x, train_y, dev_x, dev_y):
    """
    :param train_x: training data features
    :param train_y: training data labels
    :param dev_x: development data features
    :param dev_y: development data labels
    :return: Ridge regression model, optimal alpha
    
    """
    alpha_range = np.arange(-12, 4, 0.5)
    alpha_range = 10.0 ** alpha_range
    alpha_range = [0.0] + list(alpha_range)
    # low_mse, optimal_alpha = float('inf'), None
    high_corr, optimal_alpha = float('-inf'), None
    for i, alpha in enumerate(alpha_range):
        ridge_reg = Ridge(alpha=alpha)  # You can adjust the alpha parameter for regularization strength
        ridge_reg.fit(train_x, train_y)
        train_pred = ridge_reg.predict(train_x)
        dev_pred = ridge_reg.predict(dev_x)
        dev_mse = mean_squared_error(dev_y, dev_pred)
        # if dev_mse < low_mse:
        #     low_mse = dev_mse
        #     optimal_alpha = alpha
        dev_corr = get_correlation(dev_y, dev_pred)
        if dev_corr > high_corr:
            high_corr = dev_corr
            optimal_alpha = alpha

    ridge_reg = Ridge(alpha=optimal_alpha)
    data_x = pd.concat([train_x, dev_x], ignore_index = True)
    data_y = pd.concat([train_y, dev_y], ignore_index = True)
    ridge_reg.fit(data_x, data_y)
    return ridge_reg, high_corr

def evaluate_regression(test_y, test_pred, hard_idxes, easy_idxes):
    human_scores = np.array(test_y).flatten()
    predicted_scores = np.array(test_pred).flatten()

    print('Pearson correlation between Predicted and Human scores =', '{:.6f}'.format(np.corrcoef(predicted_scores, human_scores)[0][1]))
    print('MSE error =', '{:.6f}'.format(np.mean((predicted_scores - human_scores) ** 2)))

    hard_subset_human = human_scores[hard_idxes]
    hard_subset_test_predicted = predicted_scores[hard_idxes]

    print('\n On hard subset: \n')

    print('Pearson correlation between Predicted and Human scores =', '{:.6f}'.format(np.corrcoef(hard_subset_human, hard_subset_test_predicted)[0][1]))
    print('MSE error =', '{:.6f}'.format(np.mean((hard_subset_human - hard_subset_test_predicted) ** 2)))


    easy_subset_human_scores = human_scores[easy_idxes]
    easy_idxes_subset_gpt_scores = predicted_scores[easy_idxes]
    print('\nOn Easy subset:\n')
    print('Pearson correlation between GPT and Human scores =', '{:.6f}'.format(np.corrcoef(easy_idxes_subset_gpt_scores, easy_subset_human_scores)[0][1]))

    print('MSE error =', '{:.6f}'.format(np.mean((easy_idxes_subset_gpt_scores - easy_subset_human_scores) ** 2)))

if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='../data/moral_data/dyadic/dyadic_train.tsv', required=True)
    parser.add_argument('--dev-file', type=str, default='../data/moral_data/dyadic/dyadic_dev.tsv', required=True)
    parser.add_argument('--test-file', type=str, default='../data/moral_data/dyadic/dyadic_test.tsv', required=True)

    args = parser.parse_args()
    tr_df = pd.read_csv(args.train_file, sep='\t')
    dev_df = pd.read_csv(args.dev_file, sep='\t')
    test_df = pd.read_csv(args.test_file, sep='\t')

    # Specify your feature columns and target column for regression
    feature_columns = ['vulnerable_score', 'intentional_score', 'harm_score', 'help_score']
    target_column = ['human_score_normalized [-4,4]']

    # Get the indices corresponding to hard and easy scenarios
    hard_idxes = list(test_df[test_df['dataset'] == 'Lotto'].index)
    easy_idxes = list(test_df[(test_df['dataset'] != 'Lotto') & (test_df['dataset'] != 'Kruepke')].index)
    

    train_x, train_y = tr_df[feature_columns], tr_df[target_column]
    dev_x, dev_y = dev_df[feature_columns], dev_df[target_column]
    test_x, test_y = test_df[feature_columns], test_df[target_column]
    ridge_reg, high_corr = ridge_regression(train_x, train_y, dev_x, dev_y)
    regression_predictions = ridge_reg.predict(test_x)

    print('Regression Evaluation:\n')

    evaluate_regression(test_y, regression_predictions, hard_idxes, easy_idxes)