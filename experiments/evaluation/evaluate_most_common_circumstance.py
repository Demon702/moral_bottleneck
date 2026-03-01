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


NUM_TRAIN_SCENARIOS = 96
NUM_VAL_SCENARIOS = 50
NUM_TEST_SCENARIOS = 503

from glob import glob
feature_cols= [
    "vulnerable_score",
    "intentional_score",
    "harm_score",
    "help_score"
]


def evaluate_file(file_path):
    df = pd.read_csv(file_path, sep='\t')

    model = file_path.split('/')[-2]

    file_name = file_path.split('/')[-1][:-4]
    human_score_col = 'human_score'
    moral_score_col = 'moral_score'

    feature_cols= [
        "vulnerable_score",
        "intentional_score",
        "harm_score",
        "help_score"
    ]

    if 'end_to_end' in file_name:
        evaluate_llm(df, human_score_col, moral_score_col)
        return
    

    gpt_predictions = evaluate_llm(df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS], human_score_col, moral_score_col)


    train_x, train_y, val_x, val_y, test_x, test_y = prepare_data(df, feature_cols, human_score_col)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('\n\n\n\n\nPerforming regression...\n')
    regression_predictions = perform_regression(train_x, train_y, val_x, val_y, test_x, test_y, df, feature_cols)


    print('\n\n\n\n\nTwo layer mlp...\n')

    model_save_path = f'weights/{model}/{file_name}_two_layer_mlp.pth'
    os.makedirs('weights', exist_ok=True)
    mlp_predictions = perform_two_layer_mlp(train_x, train_y, val_x, val_y, test_x, test_y, df, model_save_path, device, feature_cols)

    gpt_predictions = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)[moral_score_col].values
    test_ids = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['id'].values
    perform_ensemble(gpt_predictions, regression_predictions, mlp_predictions, test_y.flatten(), df, feature_cols)
    return gpt_predictions


def evaluate_llm(df, human_moral_score_col, gpt_moral_score_col):
    filtered_df = df.dropna(subset=[human_moral_score_col, gpt_moral_score_col])
    gpt_scores, human_scores = filtered_df[gpt_moral_score_col].values, filtered_df[human_moral_score_col].values
    evaluate(gpt_scores, human_scores)
    return gpt_scores

def evaluate(gpt_scores, human_scores):
    gpt_scores = np.array(gpt_scores).flatten()
    human_scores = np.array(human_scores).flatten()
    print('Overall Pearson correlation between GPT and Human scores =', '{:.6f}'.format(np.corrcoef(gpt_scores, human_scores)[0][1]))

    print('Overall MSE error =', '{:.6f}'.format(np.mean((gpt_scores - human_scores) ** 2)))


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

def perform_regression(train_x, train_y, dev_x, dev_y, test_x, test_y, df, feature_cols):

    ridge_reg, low_mse = ridge_regression(train_x, train_y, dev_x, dev_y)
    regression_predictions = ridge_reg.predict(test_x)

    dataset_names = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['dataset'].values
    evaluate(regression_predictions, test_y)
    return regression_predictions

def perform_two_layer_mlp(train_x, train_y, dev_x, dev_y, test_x, test_y, df, model_save_path, device, feature_cols):
    best_combination, best_loss = two_layer_mlp_grid_search(train_x, train_y, dev_x, dev_y, model_save_path, device)
    hidden_size, learning_rate, batch_size = best_combination
    input_dim = train_x.shape[1]
    best_model = MLP(hidden_size=hidden_size, inp_size=input_dim).to(device)
    test_x = torch.from_numpy(test_x).float().to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    mlp_predictions = best_model(test_x).detach().numpy()
    dataset_names = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['dataset'].values
    evaluate(mlp_predictions, test_y)
    return mlp_predictions

def perform_ensemble(gpt_scores, regression_redictions, mlp_predictions, human_scores, df, feature_cols):
    gpt_scores = np.array(gpt_scores).flatten()
    regression_redictions = np.array(regression_redictions).flatten()
    mlp_predictions = np.array(mlp_predictions).flatten()

    print('\n\nEnsemble evaluation without end_to_end:\n\n')
    avg_score = (regression_redictions + mlp_predictions + gpt_scores) / 3
    evaluate(avg_score, human_scores)


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

class MLP(nn.Module):
    def __init__(self, hidden_size=4, inp_size=4):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(inp_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, inp):
        out = self.mlp1(inp)
        out = self.relu(out)
        out = self.mlp2(out)
        return out


def train_with_hyperparams(train_x, train_y, dev_x,  dev_y, hidden_size: int, learning_rate: float, batch_size: int, inp_size: int = 4, device: str ='cuda:0', num_epochs=1000):
    torch.manual_seed(0)
    model = MLP(hidden_size=hidden_size, inp_size=inp_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_dataloader = DataLoader(train_dataset, batch_size = int(batch_size), shuffle = True)
    num_epochs = 1000
    least_dev_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        for inputs, targets in train_dataloader:
            inputs = inputs.float().to(device)
            targets = targets.to(device)
            predicted = model(inputs)
            loss = criterion(predicted, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy().item()
#         print(f'Epoch {epoch}')
#         print(f'Train Loss: {(total_loss / len(train_dataloader)):.3f}')
    model.eval()
    with torch.no_grad():
        pred_dev = model(dev_x.float())
        dev_loss = criterion(pred_dev, dev_y).item()
#             print(f'Dev Loss: {dev_loss:.3f}')
        if dev_loss < least_dev_loss:
            least_dev_loss = dev_loss
            best_model = deepcopy(model)
    return least_dev_loss, best_model


def two_layer_mlp_grid_search(train_x, train_y, dev_x, dev_y, model_save_path, device):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dev_x = torch.from_numpy(dev_x).to(device)
    dev_y = torch.from_numpy(dev_y).to(device)

    # Specifying hyperparameter search space
    HIDDEN_SIZES = [4, 8, 16, 32]
    LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
    BATCH_SIZES = 2 ** np.arange(4, 8)
    best_combination = None
    best_loss = float('inf')

    input_dim = train_x.shape[1]
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    for hidden_size in HIDDEN_SIZES:
        for learning_rate in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                dev_loss, model = train_with_hyperparams(train_x, train_y, dev_x, dev_y, hidden_size, learning_rate, batch_size, inp_size=input_dim, device=device)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_combination = (hidden_size, learning_rate, batch_size)
                    torch.save(model.state_dict(), model_save_path)
                print(f'Combination: hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size = {batch_size}, loss: {dev_loss:.3f}')
    print(f'Best Combination: hidden_size={best_combination[0]}, learning_rate={best_combination[1]}, batch_size = {best_combination[2]}')
    print(f'Best loss: {best_loss:.3f}')
    return best_combination, best_loss

if __name__ == '__main__':
    
    file_paths = glob('../results/*/*most_common_circumstance*.tsv')

    for file_path in file_paths:
        model_name = file_path.split('/')[-2]

        method_name = file_path.split('/')[-1].split('_')[0]
        print('\n\n\nEvaluating ', model_name, method_name)
        evaluate_file(file_path)