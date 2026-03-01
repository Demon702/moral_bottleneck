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

def evaluate_file(file_path, model, add_reg_scores=False):
    df = pd.read_csv(file_path, sep='\t')

    file_name = file_path.split('/')[-1][:-4]

    print('\n' * 5)
    print('Evaluating file:', file_name, '\n\n\n\n\n\n\n\n')
    file_metadata = FILE_METADATA[file_name]
    human_score_col = file_metadata['human_score_col']
    moral_score_col = file_metadata['moral_score_col']
    feature_cols = file_metadata['feature_cols']

    if file_name == 'end_to_end':
        gpt_predictions = evaluate_llm(df, human_score_col, moral_score_col)
    else:
        gpt_predictions = evaluate_llm(df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS], human_score_col, moral_score_col)

    if not feature_cols:
        return df

    train_x, train_y, val_x, val_y, test_x, test_y = prepare_data(df, feature_cols, human_score_col)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('\n\n\n\n\nPerforming regression...\n')
    regression_predictions = perform_regression(train_x, train_y, val_x, val_y, test_x, test_y, df, feature_cols)

    # add a column in the df with the predictions
    if add_reg_scores:

        # Add the split column
        df["split"] = [None] * len(df)
        df.loc[df['id'] <= NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS, "split"] = "train"
        df.loc[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS, "split"] = "test"

        # Add the regression score column
        df["regression_score"] = [None] * len(df)
        df.loc[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS, "regression_score"] = regression_predictions

    print('\n\n\n\n\nTwo layer mlp...\n')

    model_save_path = f'weights/{model}/{file_name}_two_layer_mlp.pth'
    os.makedirs('weights', exist_ok=True)
    mlp_predictions = perform_two_layer_mlp(train_x, train_y, val_x, val_y, test_x, test_y, df, model_save_path, device, feature_cols)


    # add a column in the df with the predictions
    if add_reg_scores:
        df["mlp_score"] = [None] * len(df)
        df.loc[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS, "mlp_score"] = mlp_predictions

    gpt_predictions = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)[moral_score_col].values
    test_ids = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['id'].values
    # gpt_scores_end_to_end = GPT_DF_END_TO_END[GPT_DF_END_TO_END['id'].isin(test_ids)]['moral_score'].values
    # perform_ensemble(gpt_predictions, regression_predictions, mlp_predictions, gpt_scores_end_to_end, test_y.flatten(), df, feature_cols)
    return df


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
    print('train data input shape', train_x.shape)
    print('train data output shape', train_y.shape)
    print('val data input shape', val_x.shape)
    print('val data output shape', val_y.shape)
    print('test data input shape', test_x.shape)
    print('test data output shape', test_y.shape)
    return train_x, train_y, val_x, val_y, test_x, test_y

    
def evaluate_llm(df, human_moral_score_col, gpt_moral_score_col):
    filtered_df = df.dropna(subset=[human_moral_score_col, gpt_moral_score_col])
    gpt_scores, human_scores = filtered_df[gpt_moral_score_col].values, filtered_df[human_moral_score_col].values
    dataset_names = filtered_df['dataset'].values
    evaluate(gpt_scores, human_scores, dataset_names)
    return gpt_scores


def evaluate(gpt_scores, human_scores, dataset_names):
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

def perform_two_layer_mlp(train_x, train_y, dev_x, dev_y, test_x, test_y, df, model_save_path, device, feature_cols):
    best_combination, best_loss = two_layer_mlp_grid_search(train_x, train_y, dev_x, dev_y, model_save_path, device)
    hidden_size, learning_rate, batch_size = best_combination
    input_dim = train_x.shape[1]
    best_model = MLP(hidden_size=hidden_size, inp_size=input_dim).to(device)
    test_x = torch.from_numpy(test_x).float().to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    mlp_predictions = best_model(test_x).detach().numpy()
    dataset_names = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['dataset'].values
    evaluate(mlp_predictions, test_y, dataset_names)
    np.save(f'predictions/{model_save_path.split("/")[-1][:-4]}.npy', mlp_predictions)
    return mlp_predictions

def perform_ensemble(gpt_scores, regression_redictions, mlp_predictions, gpt_scores_end_to_end, human_scores, df, feature_cols):
    gpt_scores = np.array(gpt_scores).flatten()
    regression_redictions = np.array(regression_redictions).flatten()
    mlp_predictions = np.array(mlp_predictions).flatten()
    gpt_scores_end_to_end = np.array(gpt_scores_end_to_end).flatten()

    dataset_names = df[df['id'] > NUM_TRAIN_SCENARIOS + NUM_VAL_SCENARIOS].dropna(subset=feature_cols)['dataset'].values
    print('\n\nEnsemble evaluation without end_to_end:\n\n')
    avg_score = (regression_redictions + mlp_predictions) / 2
    evaluate(avg_score, human_scores, dataset_names)

    # print('Ensemble evaluation with end_to_end:\n')
    # avg_score = (regression_redictions + mlp_predictions + gpt_scores_end_to_end) / 3
    # evaluate(avg_score, human_scores, dataset_names)




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
# ## Ensemble with dyadic
# """

# gpt_scores_dyadic = np.array(gpt_scores_dyadic).flatten()
# dyadic_mlp_redictions = np.array(dyadic_mlp_redictions).flatten()
# dyadic_regression_predictions = np.array(dyadic_regression_predictions).flatten()
# gpt_scores_end_to_end = np.array(gpt_scores_end_to_end).flatten()

# avg_dyadic_score = (gpt_scores_dyadic + dyadic_mlp_redictions + dyadic_regression_predictions) / 3

# print('Ensemble with Dyadic Frames evaluation:\n')
# evaluate_regression(avg_dyadic_score, human_scores)

# """## Ensemble with Dyadic and end to end (no theory)

# We can also consider the GPT end to end (without any theory) scores in our ensemble scores
# """

# avg_dyadic_score = (gpt_scores_end_to_end + dyadic_mlp_redictions + dyadic_regression_predictions) / 3

# print('Ensemble with Dyadic Frames and End to End (no theory) evaluation:\n')

# evaluate_regression(avg_dyadic_score, human_scores)

# """## Ensemble with MFT"""

# gpt_scores_mft = np.array(gpt_scores_mft).flatten()
# mft_mlp_redictions = np.array(mft_mlp_redictions).flatten()
# mft_regression_predictions = np.array(mft_regression_predictions).flatten()
# gpt_scores_end_to_end = np.array(gpt_scores_end_to_end).flatten()

# avg_mft_score = (gpt_scores_mft + mft_mlp_redictions + mft_regression_predictions) / 3

# print('Ensemble with MFT Frames evaluation:\n')
# evaluate_regression(avg_mft_score, human_scores)

# """## Ensemble with MFT and end to end (no theory)

# """

# avg_mft_score = (gpt_scores_end_to_end + mft_mlp_redictions + mft_regression_predictions) / 3

# print('Ensemble with mft Frames and End to End (no theory) evaluation:\n')

# evaluate_regression(avg_mft_score, human_scores)

# """## Mixture pf Experts"""

# class MOE(nn.Module):
#     def __init__(self):
#         super(MOE, self).__init__()
#         self.mlp_dyadic = MLP(hidden_size=8, inp_size=4)
#         self.mlp_mft = MLP(hidden_size=32, inp_size=6)
#         self.gate_dyadic = nn.Linear(4, 1)
#         self.gate_mft = nn.Linear(6, 1)


#     def forward(self, x):
#         dyadic_score, mft_score = x[:, :4], x[:, 4:]
#         dyadic_pred = self.mlp_dyadic(dyadic_score)
#         mft_pred = self.mlp_mft(mft_score)
#         weight_dyadic = self.gate_dyadic(dyadic_score)
#         weight_mft = self.gate_mft(mft_score)
#         weight_combined = torch.cat((weight_dyadic, weight_mft), dim=1)
#         normalized_wts = weight_combined.softmax(dim=-1)
#         self.weights = normalized_wts
#         combined_scores = torch.cat((dyadic_pred, mft_pred), dim = 1)
#         final_scores = (normalized_wts * combined_scores).sum(dim = -1)
#         return final_scores.unsqueeze(1)

# def train_moe(train_x, train_y, dev_x, dev_y, learning_rate, batch_size):
#     torch.manual_seed(0)
#     model = MOE().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
#     train_dataloader = DataLoader(train_dataset, batch_size = int(batch_size), shuffle = True)
#     num_epochs = 1000
#     least_dev_loss = float('inf')
#     for epoch in range(num_epochs):
#         total_loss = 0.0

#         model.train()
#         for inputs, targets in train_dataloader:
#             inputs = inputs.float().to(device)
#             targets = targets.to(device)
#             predicted = model(inputs)
#             # print(predicted)
#             loss = criterion(predicted, targets.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.detach().cpu().numpy().item()
#             # if epoch % 20 == 0:
#             #     print(f'Epoch {epoch}')
#             #     print(f'Train Loss: {(total_loss / len(train_dataloader)):.3f}')
#         model.eval()
#         with torch.no_grad():
#             pred_dev = model(dev_x.float())
#             dev_loss = criterion(pred_dev, dev_y).item()
#             # print(f'Dev Loss: {dev_loss:.3f}')
#             if dev_loss < least_dev_loss:
#                 least_dev_loss = dev_loss
#                 best_model = deepcopy(model)
#     return least_dev_loss, best_model

# # HIDDEN_SIZES = [4, 8, 16, 32]

# train_moe_x = np.concatenate((dyadic_train_x.values, mft_train_x.values), axis=1)
# train_moe_y = dyadic_train_y.values
# dev_moe_x = torch.from_numpy(np.concatenate((dyadic_dev_x, mft_dev_x), axis=1)).to(device)
# dev_moe_y = torch.from_numpy(dyadic_dev_y.values).to(device)
# # LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
# LEARNING_RATES = [0.01]
# BATCH_SIZES = [32]

# best_combination = None
# best_loss = float('inf')

# # for hidden_size in HIDDEN_SIZES:
# for learning_rate in LEARNING_RATES:
#     for batch_size in BATCH_SIZES:
#         dev_loss, model = train_moe(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, learning_rate, batch_size)
#         if dev_loss < best_loss:
#             best_loss = dev_loss
#             best_combination = (learning_rate, batch_size)
#             torch.save(model.state_dict(), 'weights/best_model_hyperparams_moe.pth')
#         print(f'Combination: learning_rate={learning_rate}, batch_size = {batch_size}, loss: {dev_loss:.3f}')
# print(f'Best Combination: learning_rate={best_combination[0]}, batch_size = {best_combination[1]}')
# print(f'Best loss: {best_loss:.3f}')
# # Best combination = learning_rate=0.01, batch_size = 32, loss: 1.149

# best_moe_model = MOE()
# best_moe_model.load_state_dict(torch.load('weights/best_model_hyperparams_moe.pth'))

# test_x = torch.from_numpy(np.concatenate((dyadic_test_x.values, mft_test_x.values), axis=1)).float()
# test_y = mft_test_y.to_numpy()
# criterion = nn.MSELoss()
# moe_predictions = best_moe_model(test_x).detach().numpy()


# print('MOE Evaluation:\n')

# evaluate_regression(test_y, moe_predictions)

# """## Analysis of MOE Weights"""

# weights = best_moe_model.weights
# print('Average dyadic weight:', weights[:, 0].mean().detach().item())
# print('Average mft weight:', weights[:, 1].mean().detach().item())

# avg_moe_score = (gpt_scores_end_to_end + moe_predictions.flatten()) / 2
# print('Ensemble with moe and End to End (no theory) evaluation:\n')

# evaluate_regression(avg_moe_score, human_scores)

# correlations = [0.8667, 0.8411, 0.8130, 0.8576, 0.5375, 0.8671, 0.6059, 0.8790 ,0.7996, 0.9025, 0.8407, 0.8856, 0.9043]
# mses = [2.5886,2.7395, 3.7102, 2.2067, 3.0018, 1.1019, 2.7154, 0.9603, 1.5192, 0.7890, 1.2623, 0.9121, 1.0150]

# import matplotlib.pyplot as plt

# plt.scatter(correlations, mses)

# plt.xlabel('Correlation')
# plt.ylabel('MSE')

# """## Mixture of experts with scores"""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Expert(nn.Module):
#     def __init__(self, input_size=10, output_size=1):
#         super(Expert, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, output_size),
#         )

#     def forward(self, x):
#         return self.network(x)

# class GatingMechanism(nn.Module):
#     def __init__(self, input_size, num_experts):
#         super(GatingMechanism, self).__init__()
#         self.num_experts = num_experts
#         self.gate = nn.Linear(input_size, num_experts)

#     def forward(self, x):
#         return self.gate(x).softmax(dim=1)

# class MixtureOfExperts(nn.Module):
#     def __init__(self, input_size, output_size, num_experts=4):
#         super(MixtureOfExperts, self).__init__()
#         self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(num_experts)])
#         self.gate = GatingMechanism(input_size, num_experts)

#     def forward(self, x):
#         # Get outputs from all experts for the input
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         # Get gating weights
#         gate_outputs = self.gate(x)
#         # Weighted sum of expert outputs
#         output = (gate_outputs.unsqueeze(2) * expert_outputs).sum(dim = 1)
#         return output

# # Example usage
# input_size = 10  # Size of your input vector
# output_size = 1  # Size of your output
# num_experts = 4  # Number of experts

# model = MixtureOfExperts(input_size, output_size, num_experts)

# def train_moe_wth_features(train_x, train_y, dev_x, dev_y, learning_rate, batch_size, num_experts):
#     torch.manual_seed(0)
#     model = MixtureOfExperts(input_size=10, output_size=1, num_experts=num_experts).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
#     train_dataloader = DataLoader(train_dataset, batch_size = int(batch_size), shuffle = True)
#     num_epochs = 1000
#     least_dev_loss = float('inf')
#     for epoch in range(num_epochs):
#         total_loss = 0.0

#         model.train()
#         for inputs, targets in train_dataloader:
#             inputs = inputs.float().to(device)
#             targets = targets.to(device)
#             predicted = model(inputs)
#             # print(predicted)
#             loss = criterion(predicted, targets.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.detach().cpu().numpy().item()
#             # if epoch % 20 == 0:
#             #     print(f'Epoch {epoch}')
#             #     print(f'Train Loss: {(total_loss / len(train_dataloader)):.3f}')
#         model.eval()
#         with torch.no_grad():
#             pred_dev = model(dev_x.float())
#             dev_loss = criterion(pred_dev, dev_y).item()
#             # print(f'Dev Loss: {dev_loss:.3f}')
#             if dev_loss < least_dev_loss:
#                 least_dev_loss = dev_loss
#                 best_model = deepcopy(model)
#     return least_dev_loss, best_model

# # LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
# # BATCH_SIZES =  2 ** np.arange(2, 7)
# # NUM_EXPERTS_CHOICES = 2 ** np.arange(1, 5)

# LEARNING_RATES = [0.01]
# BATCH_SIZES =  [4]
# NUM_EXPERTS_CHOICES = [4]

# best_combination = None
# best_loss = float('inf')

# for learning_rate in LEARNING_RATES:
#     for batch_size in BATCH_SIZES:
#         for num_experts in NUM_EXPERTS_CHOICES:
#             dev_loss, model = train_moe_wth_features(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, learning_rate, batch_size, num_experts)
#             if dev_loss < best_loss:
#                 best_loss = dev_loss
#                 best_combination = (learning_rate, batch_size, num_experts)
#                 torch.save(model.state_dict(), 'weights/best_model_hyperparams_moe_with_features.pth')
#             print(f'Combination: learning_rate={learning_rate}, batch_size = {batch_size}, num_experts = {num_experts}, loss: {dev_loss:.3f}')
# print(f'Best Combination: learning_rate={best_combination[0]}, batch_size = {best_combination[1]}, num_experts = {best_combination[2]}')
# print(f'Best loss: {best_loss:.3f}')
# # Best combination = learning_rate=0.01, batch_size = 8, num_experts = 4 loss: 1.149

# best_moe_with_features_model = MixtureOfExperts(input_size= 10, output_size = 1, num_experts = 4)
# best_moe_with_features_model.load_state_dict(torch.load('weights/best_model_hyperparams_moe_with_features.pth'))

# test_x = torch.from_numpy(np.concatenate((dyadic_test_x.values, mft_test_x.values), axis=1)).float()
# test_y = mft_test_y.to_numpy()

# moe_with_features_predictions = best_moe_with_features_model(test_x).detach().numpy()


# print('MOE with features Evaluation:\n')

# evaluate_regression(test_y, moe_with_features_predictions)

# for idx, expert in enumerate(best_moe_with_features_model.experts):
#     print(f"Expert {idx}:")
#     for name, param in expert.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values : \n{param.data}\n")

# """## Mixture of experts with features top k"""

# class MixtureOfExpertsTopk(nn.Module):
#     def __init__(self, input_size, output_size, num_experts=4):
#         super(MixtureOfExpertsTopk, self).__init__()
#         self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(num_experts)])
#         self.gate = GatingMechanism(input_size, num_experts)

#     def forward(self, x):
#         # Get outputs from all experts for the input
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
#         # Get gating weights
#         gate_outputs = self.gate(x)
#         # Weighted sum of expert outputs
#         # Select top 2 experts based on gating mechanism
#         top2_values, top2_indices = torch.topk(gate_outputs, 2, dim=1)
#         # print('top2 values', top2_values)

#         top2_values = top2_values.softmax(dim=1)

#         # print('gated_outputs', gate_outputs)
#         # print('top2 values after softmax', top2_values)
#         # print('top 2 indices', top2_indices)

#         # print('expert outputs', expert_outputs)

#         # Use only the outputs of the top 2 experts

#         selected_outputs = expert_outputs.gather(dim = 1, index = top2_indices.unsqueeze(2))
#         # print('selected outputs', selected_outputs)
#         output = (top2_values.unsqueeze(2) * selected_outputs).sum(dim = 1)
#         return output
#         # output = torch.zeros(x.size(0), output_size, device=expert_outputs.device)
#         # for i in range(x.size(0)):
#         #     output[i] = torch.sum(top2_values[i].unsqueeze(1) * expert_outputs[i, top2_indices[i]], dim=0)

#         return output


# input_size = 10  # Size of your input vector
# output_size = 1  # Size of your output
# num_experts = 3  # Number of experts

# model = MixtureOfExpertsTopk(input_size, output_size, num_experts)

# # Example input
# x = torch.rand(5, input_size)  # Batch size of 5, input size of 10

# output = model(x)

# def train_moe_wth_features(train_x, train_y, dev_x, dev_y, learning_rate, batch_size, num_experts):
#     torch.manual_seed(0)
#     model = MixtureOfExpertsTopk(input_size=10, output_size=1, num_experts=num_experts).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
#     train_dataloader = DataLoader(train_dataset, batch_size = int(batch_size), shuffle = True)
#     num_epochs = 1000
#     least_dev_loss = float('inf')
#     for epoch in range(num_epochs):
#         total_loss = 0.0

#         model.train()
#         for inputs, targets in train_dataloader:
#             inputs = inputs.float().to(device)
#             targets = targets.to(device)
#             predicted = model(inputs)
#             # print(predicted)
#             loss = criterion(predicted, targets.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.detach().cpu().numpy().item()
#             # if epoch % 20 == 0:
#             #     print(f'Epoch {epoch}')
#             #     print(f'Train Loss: {(total_loss / len(train_dataloader)):.3f}')
#         model.eval()
#         with torch.no_grad():
#             pred_dev = model(dev_x.float())
#             dev_loss = criterion(pred_dev, dev_y).item()
#             # print(f'Dev Loss: {dev_loss:.3f}')
#             if dev_loss < least_dev_loss:
#                 least_dev_loss = dev_loss
#                 best_model = deepcopy(model)
#     return least_dev_loss, best_model

# LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
# BATCH_SIZES =  2 ** np.arange(3, 6)
# NUM_EXPERTS_CHOICES = 2 ** np.arange(1, 5)

# best_combination = None
# best_loss = float('inf')

# for learning_rate in LEARNING_RATES:
#     for batch_size in BATCH_SIZES:
#         for num_experts in NUM_EXPERTS_CHOICES:
#             dev_loss, model = train_moe_wth_features(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, learning_rate, batch_size, num_experts)
#             if dev_loss < best_loss:
#                 best_loss = dev_loss
#                 best_combination = (learning_rate, batch_size, num_experts)
#                 torch.save(model.state_dict(), 'weights/best_model_hyperparams_moe_with_features_top_k.pth')
#             print(f'Combination: learning_rate={learning_rate}, batch_size = {batch_size}, num_experts = {num_experts}, loss: {dev_loss:.3f}')
# print(f'Best Combination: learning_rate={best_combination[0]}, batch_size = {best_combination[1]}, num_experts = {best_combination[2]}')
# print(f'Best loss: {best_loss:.3f}')

# best_moe_with_features_top_k_model = MixtureOfExpertsTopk(input_size= 10, output_size = 1, num_experts = 16)
# best_moe_with_features_top_k_model.load_state_dict(torch.load('weights/best_model_hyperparams_moe_with_features_top_k.pth'))

# test_x = torch.from_numpy(np.concatenate((dyadic_test_x.values, mft_test_x.values), axis=1)).float()
# test_y = mft_test_y.to_numpy()

# moe_with_features_predictions = best_moe_with_features_top_k_model(test_x).detach().numpy()


# print('MOE with features Evaluation:\n')

# evaluate_regression(test_y, moe_with_features_predictions)
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Moral Score Prediction')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use for moral score prediction')
    parser.add_argument('--theory', type=str, default='utilitarianism', help='Moral Theory to evaluate')
    parser.add_argument('--add_reg_scores', action='store_true', help='Add regression scores to the evaluation')

    args = parser.parse_args()
    model = args.model.replace('/', '_')

    FILE_NAMES = [args.theory]
    for file_name in FILE_NAMES:
        file_path = f'../results/{model}/{file_name}.tsv'
        df = evaluate_file(f'../results/{model}/{file_name}.tsv', model, add_reg_scores=args.add_reg_scores)
        if args.add_reg_scores:
            df.to_csv(file_path.replace(".tsv", "_with_reg_scores.tsv"), index=False, sep="\t")