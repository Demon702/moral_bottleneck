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


def train_with_hyperparams(train_x, train_y, dev_x,  dev_y, hidden_size: int, learning_rate: float, batch_size: int, inp_size=4):
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

def two_layer_mlp_grid_search(train_x, train_y, dev_x, dev_y, inp_size=4, model_save_path='weights/best_model_hyperparams_two_layer.pth'):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    dev_x = torch.from_numpy(dev_x.to_numpy()).to(device)
    dev_y = torch.from_numpy(dev_y.to_numpy()).to(device)

    # Specifying hyperparameter search space
    HIDDEN_SIZES = [4, 8, 16, 32]
    LEARNING_RATES = 10 ** np.linspace(-2, -4, 5)
    BATCH_SIZES = 2 ** np.arange(1, 7)
    best_combination = None
    best_loss = float('inf')


    for hidden_size in HIDDEN_SIZES:
        for learning_rate in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                dev_loss, model = train_with_hyperparams(train_x, train_y, dev_x, dev_y, hidden_size, learning_rate, batch_size, inp_size)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_combination = (hidden_size, learning_rate, batch_size)
                    torch.save(model.state_dict(), model_save_path)
                print(f'Combination: hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size = {batch_size}, loss: {dev_loss:.3f}')
    print(f'Best Combination: hidden_size={best_combination[0]}, learning_rate={best_combination[1]}, batch_size = {best_combination[2]}')
    print(f'Best loss: {best_loss:.3f}')

    return best_combination


if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='../data/moral_data/dyadic/dyadic_train.tsv', required=True)
    parser.add_argument('--dev-file', type=str, default='../data/moral_data/dyadic/dyadic_dev.tsv', required=True)
    parser.add_argument('--test_file', type=str, default='../data/moral_data/dyadic/dyadic_test.tsv', required=True)
    parser.add_argument('--model-save-path', type=str, default='weights/best_model_hyperparams_two_layer.pth')

    args = parser.parse_args()
    tr_df = pd.read_csv(args.train_file, sep='\t')
    dev_df = pd.read_csv(args.dev_file, sep='\t')
    test_df = pd.read_csv(args.test_file, sep='\t')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Specify your feature columns and target column for mlp input and output
    feature_columns = ['vulnerable_score', 'intentional_score', 'harm_score', 'help_score']
    target_column = ['human_score_normalized [-4,4]']

    # Get the indices corresponding to hard and easy scenarios
    hard_idxes = list(test_df[test_df['dataset'] == 'Lotto'].index)
    easy_idxes = list(test_df[(test_df['dataset'] != 'Lotto') & (test_df['dataset'] != 'Kruepke')].index)
    

    train_x, train_y = tr_df[feature_columns], tr_df[target_column]
    dev_x, dev_y = dev_df[feature_columns], dev_df[target_column]
    test_x, test_y = test_df[feature_columns], test_df[target_column]
    best_combination = two_layer_mlp_grid_search(train_x, train_y, dev_x, dev_y, inp_size=len(feature_columns), model_save_path=args.model_save_path)

    best_model = MLP(hidden_size=best_combination[0], inp_size=len(feature_columns)).to(device)
    best_model.load_state_dict(torch.load(args.model_save_path))

    test_x = torch.from_numpy(test_x.to_numpy()).float()
    test_y = test_y.to_numpy()
    criterion = nn.MSELoss()
    dyadic_mlp_redictions = best_model(test_x).detach().numpy()


    print('Two Layer MLP with Dyadic Theory Features Evaluation:\n')

    evaluate(test_y, dyadic_mlp_redictions, hard_idxes, easy_idxes)