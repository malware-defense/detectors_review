from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

class DrebinNnTorch:
    def __init__(self, mode='train', filename="nn_drebin_torch.h5", normalize_mean=False, epochs=50, batch_size=128):
        self.mode = mode #train or load
        self.filename = filename
        self.normalize_mean = normalize_mean
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================
        self.num_classes = 2
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_drebin_data()
        if normalize_mean:
            self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        # else: # linear 0-1
        #     self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)     #todo：zyy这里的normalize_linear是否有必要，fix；如何归一化[0,1]

        #convert labels to one_hot
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)

        # train_loader、test_loader
        self.train_loader = DataLoader(TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float()), shuffle=True, batch_size=10)
        self.test_loader = DataLoader(TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float()), shuffle=True, batch_size=10)

        #====================== Model =============================
        self.input_shape = self.x_train.shape[1:]
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_state_dict(torch.load("{}{}".format(checkpoints_dir, self.filename)))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")


    def build_model(self):
        #================= The final model ============================
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11 * 11, 128), nn.ReLU(),
            nn.Linear(128, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, self.num_classes))
        return model
    
    def train(self, model):
        #================= Settings =========================
        learning_rate = 0.001

        #================= Train =========================
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(self.epochs):
            for i, (x, y) in tqdm(enumerate(self.train_loader)):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                logit = model(x)
                loss = F.cross_entropy(logit, (torch.max(y, dim = 1)[1]))
                loss.backward()
                optimizer.step()
                if i % 1000 == 0:
                    print('第{}个数据，loss值等于{}'.format(i, loss))

        # ================= Save model and history =========================
        torch.save(model.state_dict(), "{}{}".format(checkpoints_dir, self.filename))

        return model
