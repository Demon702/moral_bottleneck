import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import argparse
import pandas as pd
torch.manual_seed(0)

from copy import deepcopy


def evaluate(test_y, test_pred, hard_idxes, easy_idxes):
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

class Expert(nn.Module):
    def __init__(self, input_size=10, output_size=1):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        return self.network(x)

class GatingMechanism(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingMechanism, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_size, num_experts)

    def forward(self, x):
        return self.gate(x).softmax(dim=1)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, output_size, num_experts=4):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(num_experts)])
        self.gate = GatingMechanism(input_size, num_experts)

    def forward(self, x):
        # Get outputs from all experts for the input
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Get gating weights
        gate_outputs = self.gate(x)
        # Weighted sum of expert outputs
        output = (gate_outputs.unsqueeze(2) * expert_outputs).sum(dim = 1)
        return output



def train_moe_wth_features(train_x, train_y, dev_x, dev_y, learning_rate, batch_size, num_experts):
    torch.manual_seed(0)
    model = MixtureOfExperts(input_size=10, output_size=1, num_experts=num_experts).to(device)
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
            # print(predicted)
            loss = criterion(predicted, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().numpy().item()
            # if epoch % 20 == 0:
            #     print(f'Epoch {epoch}')
            #     print(f'Train Loss: {(total_loss / len(train_dataloader)):.3f}')
        model.eval()
        with torch.no_grad():
            pred_dev = model(dev_x.float())
            dev_loss = criterion(pred_dev, dev_y).item()
            # print(f'Dev Loss: {dev_loss:.3f}')
            if dev_loss < least_dev_loss:
                least_dev_loss = dev_loss
                best_model = deepcopy(model)
    return least_dev_loss, best_model

def moe_with_features_grid_search(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, model_save_path='weights/best_model_hyperparams_moe_with_features.pth'):
    LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
    BATCH_SIZES =  2 ** np.arange(2, 7)
    NUM_EXPERTS_CHOICES = 2 ** np.arange(1, 5)

    best_combination = None
    best_loss = float('inf')

    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for num_experts in NUM_EXPERTS_CHOICES:
                dev_loss, model = train_moe_wth_features(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, learning_rate, batch_size, num_experts)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_combination = (learning_rate, batch_size, num_experts)
                    torch.save(model.state_dict(), model_save_path)
                print(f'Combination: learning_rate={learning_rate}, batch_size = {batch_size}, num_experts = {num_experts}, loss: {dev_loss:.3f}')
    print(f'Best Combination: learning_rate={best_combination[0]}, batch_size = {best_combination[1]}, num_experts = {best_combination[2]}')
    print(f'Best loss: {best_loss:.3f}')
    return best_combination


if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dyadic-train_file', type=str, default='../data/moral_data/dyadic/dyadic_train.tsv', required=True)
    parser.add_argument('--dyadic-dev-file', type=str, default='../data/moral_data/dyadic/dyadic_dev.tsv', required=True)
    parser.add_argument('--dyadic-test_file', type=str, default='../data/moral_data/dyadic/dyadic_test.tsv', required=True)

    parser.add_argument('--mft-train_file', type=str, default='../data/moral_data/mft/mft_train.tsv', required=True)
    parser.add_argument('--mft-dev-file', type=str, default='../data/moral_data/mft/mft_dev.tsv', required=True)
    parser.add_argument('--mft-test_file', type=str, default='../data/moral_data/mft/mft_test.tsv', required=True)
        
    parser.add_argument('--model-save-path', type=str, default='weights/best_model_hyperparams_moe_with_theories.pth')

    args = parser.parse_args()
    dyadic_tr_df = pd.read_csv(args.dyadic_train_file, sep='\t')
    dyadic_dev_df = pd.read_csv(args.dyadic_dev_file, sep='\t')
    dyadic_test_df = pd.read_csv(args.dyadic_test_file, sep='\t')

    mft_tr_df = pd.read_csv(args.mft_train_file, sep='\t')
    mft_dev_df = pd.read_csv(args.mft_dev_file, sep='\t')
    mft_test_df = pd.read_csv(args.mft_test_file, sep='\t')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Get the indices corresponding to hard and easy scenarios
    hard_idxes = list(dyadic_test_df[dyadic_test_df['dataset'] == 'Lotto'].index)
    easy_idxes = list(dyadic_test_df[(dyadic_test_df['dataset'] != 'Lotto') & (dyadic_test_df['dataset'] != 'Kruepke')].index)
    
    dyadic_feature_columns = ['vulnerable_score', 'intentional_score', 'harm_score', 'help_score']
    dyadic_target_column = ['human_score_normalized [-4,4]']

    dyadic_train_x, dyadic_train_y = dyadic_tr_df[dyadic_feature_columns], dyadic_tr_df[dyadic_target_column]
    dyadic_dev_x, dyadic_dev_y = dyadic_dev_df[dyadic_feature_columns], dyadic_dev_df[dyadic_target_column]
    dyadic_test_x, dyadic_test_y = dyadic_test_df[dyadic_feature_columns], dyadic_test_df[dyadic_target_column]
    

    mft_feature_columns = ['Harm/ Care', 'Cheating/ Fairness', 'Betrayal/ Loyalty', 'Subversion/ Authority', 'Degradation/ Sanctity', 'Oppression/ Liberty']
    mft_target_column = ['Human Score Normalized [-4,4]']

    mft_train_x, mft_train_y = mft_tr_df[mft_feature_columns], mft_tr_df[mft_target_column]
    mft_dev_x, mft_dev_y = mft_dev_df[mft_feature_columns], mft_dev_df[mft_target_column]
    mft_test_x, mft_test_y = mft_test_df[mft_feature_columns], mft_test_df[mft_target_column]


    train_moe_x = np.concatenate((dyadic_train_x.values, mft_train_x.values), axis=1)
    train_moe_y = dyadic_train_y.values
    dev_moe_x = torch.from_numpy(np.concatenate((dyadic_dev_x, mft_dev_x), axis=1)).to(device)
    dev_moe_y = torch.from_numpy(dyadic_dev_y.values).to(device)

    best_combination = moe_with_features_grid_search(train_moe_x, train_moe_y, dev_moe_x, dev_moe_y, model_save_path=args.model_save_path)

    best_moe_with_features_model = MixtureOfExperts(input_size=10, output_size=1, num_experts=best_combination[2])
    best_moe_with_features_model.load_state_dict(torch.load(args.model_save_path))

    test_x = torch.from_numpy(np.concatenate((dyadic_test_x.values, mft_test_x.values), axis=1)).float()
    test_y = mft_test_y.to_numpy()

    moe_with_features_predictions = best_moe_with_features_model(test_x).detach().numpy()


    print('MOE with features Evaluation:\n')

    evaluate(test_y, moe_with_features_predictions, hard_idxes, easy_idxes)