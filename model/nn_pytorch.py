from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class NnTorch:
    def __init__(self, dataset, model_save_dir, filename, mode='train', normalize_mean=False, epochs=50, batch_size=128):
        self.mode = mode  # train or load, main.py中根据参数决定
        self.model_save_dir = model_save_dir
        self.filename = filename
        self.normalize_mean = normalize_mean
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 2

        # ====================== Model =============================
        self.input_shape = dataset.x_train.shape[1]
        self.model = self.build_model()

        if mode == 'train':
            self.model = self.train(self.model, dataset)
        elif mode == 'load':
            self.model.load_state_dict(torch.load("{}{}".format(self.model_save_dir, self.filename)))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def build_model(self):
        # ================= The final model ============================
        model = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.input_shape, 128), nn.ReLU(),
            nn.Linear(128, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, self.num_classes))
        return model

    def train(self, model, dataset):
        # ================= Settings =========================
        learning_rate = 0.001

        # ================= Train =========================
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(self.epochs):
            for i, (x, y) in tqdm(enumerate(dataset.train_loader)):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                logit = model(x)
                loss = F.cross_entropy(logit, (torch.max(y, dim = 1)[1]))
                loss.backward()
                optimizer.step()
                if i % 1000 == 0:
                    print('第{}个数据，loss值等于{}'.format(i, loss))

        # ================= Save model and history =========================
        torch.save(model.state_dict(), "{}{}".format(self.model_save_dir, self.filename))

        return model
