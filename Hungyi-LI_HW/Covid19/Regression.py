#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time   : 2023/3/30 17:56
# @Author : Lixinqian Yu
# @E-mail : yulixinqian805@gmail.com
# @File   : Regression.py
# @Project: NLP
import sys

import torch, os, math, time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    INFO("Execute Train_Valid_split!")
    valid_len = int(valid_ratio*len(data_set))
    train_len = int(len(data_set)-valid_len)
    train_set, valid_set = random_split(data_set, [train_len, valid_len],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def select_feature(dataset, select_all=True, select_list=None):
    '''useful feature will be chosen'''
    feature = dataset[:, :-1]
    target = dataset[:, -1]
    if select_all:
        feature_idx = list(range(feature.shape[1]))
    else:
        if select_list:
            INFO("Function is still building!")
        else:
            INFO("Error!")

    return feature[:, feature_idx], target


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # the random number for numpy's methods is fixed
    torch.manual_seed(seed)  # the random number for CPU is fixed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # the rand number for all GPU are fixed


class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.
    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True, file=sys.stdout)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


def INFO(text: str):
    print("[INFO]-->"+text)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # hyper-parameters for model
    config = {
        'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
        'select_all': True,  # Whether to use all features.
        'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
        'n_epochs': 300,  # Number of epochs.
        'batch_size': 256,
        'learning_rate': 1e-5,
        'early_stop': 400,  # If model has not improved for this many consecutive ep ochs, stop training.
        'save_path': 'models/model.ckpt'  # Your model will be saved here.
    }
    INFO(f"Training device --> {device}")
    INFO(f"Model will be saved in directory --> {os.path.join(os.getcwd(), config.get('save_path'))}")

    # Set seed for reproducibility
    same_seed(config['seed'])

    train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
    INFO(f"Train_data --> {train_data.shape}   "+f"Valid_data --> {valid_data.shape}")
    INFO(f"Test_data --> {test_data.shape}")
    x_train, y_train = select_feature(train_data, config['select_all'])
    x_valid, y_valid = select_feature(valid_data, config['select_all'])
    x_test, y_test = select_feature(test_data, config['select_all'])

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                                 COVID19Dataset(x_valid, y_valid), \
                                                 COVID19Dataset(x_test)
    INFO("COVID19Datasets are all created!")
    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    INFO("Dataloaders are all created!")

    model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)





