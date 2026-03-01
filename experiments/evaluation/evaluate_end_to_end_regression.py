import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0)
import glob
from copy import deepcopy
import os
import argparse


DATASETS = [
    'Clifford',
    'Effron',
    'Mickelberg',
    'Cook',
    'Grizzard',
    'Kruepke',
    'Lotto'
]

FILE_METADATA = json.load(open('metadata.json', 'r'))

NUM_TRAIN_SCENARIOS = 96
NUM_VAL_SCENARIOS = 50
NUM_TEST_SCENARIOS = 503

GPT_DF_END_TO_END = None

def get_pearson_correlation(ground_truth, predicted):
    ground_truth = np.array(ground_truth).flatten()
    predicted = np.array(predicted).flatten()
    return np.corrcoef(ground_truth, predicted)[0][1]

def evaluate_end_to_end_all(file_path):
    df = pd.read_csv(file_path, sep='\t')

    file_name = file_path.split('/')[-1][:-4]

    print('\n' * 5)
    print('Evaluating file:', file_name, '\n\n\n\n\n\n\n\n')
    file_metadata = FILE_METADATA["end_to_end"]
    human_score_col = file_metadata['human_score_col']
    moral_score_col = file_metadata['moral_score_col']
    feature_cols = file_metadata['feature_cols']

    gpt_predictions = evaluate_llm(df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS], human_score_col, moral_score_col)

    feature_cols = [moral_score_col]

    train_x, train_y, val_x, val_y, test_x, test_y = prepare_data(df, feature_cols, human_score_col)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('\n\n\n\n\nPerforming regression...\n')
    regression_predictions = perform_regression(train_x, train_y, val_x, val_y, test_x, test_y, df, feature_cols)

    return gpt_predictions


def prepare_data(df, feature_cols, human_score_col):
    train_df = df[df['id'] <= NUM_TRAIN_SCENARIOS].dropna(subset=feature_cols)
    val_df = df[(NUM_TRAIN_SCENARIOS < df['id']) & (df['id'] <= NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS)].dropna(subset=feature_cols)
    test_df = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)

    train_x, train_y = train_df[feature_cols].values, train_df[[human_score_col]].values
    val_x, val_y = val_df[feature_cols].values, val_df[[human_score_col]].values
    test_x, test_y = test_df[feature_cols].values, test_df[[human_score_col]].values

    print('Number of training data points', len(train_x))
    print('Number of validation data points', len(val_x))
    print('Number of test data points', len(test_x))
    print('train data output shape', train_y.shape)
    return train_x, train_y, val_x, val_y, test_x, test_y

    
def evaluate_llm(df, human_moral_score_col, gpt_moral_score_col):
    filtered_df = df.dropna(subset=[human_moral_score_col, gpt_moral_score_col])
    gpt_scores, human_scores = filtered_df[gpt_moral_score_col].values, filtered_df[human_moral_score_col].values
    dataset_names = filtered_df['dataset'].values
    evaluate(gpt_scores, human_scores, dataset_names)
    return gpt_scores


def evaluate(gpt_scores, human_scores, dataset_names):
    print(f"Len of gpt scores: {len(gpt_scores)}")
    print(f"Len of human scores: {len(human_scores)}")
    gpt_scores = np.array(gpt_scores).flatten()
    human_scores = np.array(human_scores).flatten()

    print("Len of gpt scores", len(gpt_scores))
    print("Len of human scores", len(human_scores))
    # print("gpt scores", gpt_scores)
    # print("human scores", human_scores)
    print('Overall Pearson correlation between GPT and Human scores =', '{:.6f}'.format(np.corrcoef(gpt_scores, human_scores)[0][1]))

    print('Overall MSE error =', '{:.6f}'.format(np.mean((gpt_scores - human_scores) ** 2)))

    new_df = pd.DataFrame({'gpt_scores': gpt_scores, 'human_scores': human_scores, 'dataset': dataset_names})
    for dataset in DATASETS:
        dataset_df = new_df[new_df['dataset'] == dataset]
        gpt_dataset_scores = dataset_df['gpt_scores'].values
        human_dataset_scores = dataset_df['human_scores'].values
        print(f'\nOn {dataset} dataset:')
        print('Pearson correlation between GPT and Human scores =', '{:.6f}'.format(np.corrcoef(gpt_dataset_scores, human_dataset_scores)[0][1]))
        print('MSE error =', '{:.6f}'.format(np.mean((gpt_dataset_scores - human_dataset_scores) ** 2)))

def perform_regression(train_x, train_y, dev_x, dev_y, test_x, test_y, df, feature_cols):

    ridge_reg, low_mse = ridge_regression(train_x, train_y, dev_x, dev_y)
    regression_predictions = ridge_reg.predict(test_x)

    dataset_names = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['dataset'].values
    evaluate(regression_predictions, test_y, dataset_names)
    return regression_predictions

# Train the Ridge regression model with grid search on regularization parameter alpha
def ridge_regression(train_x, train_y, dev_x, dev_y):
    alpha_range = np.arange(-12, 4, 0.5)
    alpha_range = 10.0 ** alpha_range
    alpha_range = [0.0] + list(alpha_range)
    low_mse, optimal_alpha = float('inf'), None
    for i, alpha in enumerate(alpha_range):
        ridge_reg = Ridge(alpha=alpha)  # You can adjust the alpha parameter for regularization strength
        ridge_reg.fit(train_x, train_y)
        train_pred = ridge_reg.predict(train_x)
        dev_pred = ridge_reg.predict(dev_x)
        dev_mse = mean_squared_error(dev_y, dev_pred)
        if dev_mse < low_mse:
            low_mse = dev_mse
            optimal_alpha = alpha

    ridge_reg = Ridge(alpha=optimal_alpha)
    data_x = np.concatenate((train_x, dev_x), axis=0)
    data_y = np.concatenate((train_y, dev_y), axis=0)
    ridge_reg.fit(data_x, data_y)
    return ridge_reg, low_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Moral Score Prediction')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use for moral score prediction')

    args = parser.parse_args()
    model = args.model.replace('/', '_')
    file_path = f'../results/{model}/end_to_end_all.tsv'

    evaluate_end_to_end_all(file_path)
