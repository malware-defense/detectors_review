from settings import config
import numpy as np
import os
from common.util import *
from setup_paths import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import logging

class ApkData:
    """The Dataset for training the malware detection methods"""

    def __init__(self, detection, classifier, attacker, base_clf_dir=config['base_clf_dir'], data_source=config['data_source']):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_adv = None
        self.y_adv = None
        self.detection = detection
        self.classifier = classifier
        self.attacker = attacker
        self.data_source = data_source

        self.base_clf_name = "_".join([detection, 'apg', classifier]) + ".model"
        self.base_clf_dir = base_clf_dir
        self.base_clf_path = None
        self.base_clf = None
        self.base_clf_predict_y_train = None

        self.load_data(detection, classifier, attacker, data_source)

        # train_loader、test_loader
        self.train_loader = DataLoader(TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float()), shuffle=True, batch_size=10)
        self.test_loader = DataLoader(TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float()), shuffle=True, batch_size=10)


    def load_data(self, detection, classifier, attacker, data_source, isOneHotEncoding = True):
        x_train = np.load(os.path.join(data_source, attacker, 'train_sample', detection, detection + '_' + classifier + '_train.npy'))
        y_train = np.load(os.path.join(data_source, attacker, 'train_sample', detection, detection + '_' + classifier + '_train_label.npy'))
        x_test = np.load(os.path.join(data_source, attacker, 'test_sample', detection, detection + '_' + classifier + '_test.npy'))
        y_test = np.load(
            os.path.join(data_source, attacker, 'test_sample', detection, detection + '_' + classifier + '_test_label.npy'))
        x_adv = np.load(os.path.join(data_source, attacker, 'attack_sample', detection, detection + '_' + classifier + '_attack.npy'))
        y_adv = np.load(os.path.join(data_source, attacker, 'attack_sample', detection, detection + '_' + classifier + '_attack_label.npy'))

        x_train = np.reshape(x_train, (-1, 11, 11, 1))    # 这里的size需要改变
        x_test = np.reshape(x_test, (-1, 11, 11, 1))
        x_adv = np.reshape(x_adv, (-1, 11, 11, 1))

        #convert labels to one_hot
        if isOneHotEncoding:
            self.num_classes = 2
            y_adv = y_adv.astype(int)
            self.y_train = to_onehot_encode(y_train, self.num_classes)
            self.y_test = to_onehot_encode(y_test, self.num_classes)
            self.y_adv = to_onehot_encode(y_adv, self.num_classes)

        self.x_train = x_train
        self.x_test = x_test
        self.x_adv = x_adv


        self.base_clf_path = os.path.join(self.base_clf_dir, self.base_clf_name)
        if os.path.exists(self.base_clf_path + ".clf"):
            logging.debug(blue('Loading model from {}...'.format(self.base_clf_path + ".clf")))
            with open(self.base_clf_path + ".clf", "rb") as f:
                self.base_clf = pickle.load(f)

        base_clf_predict_y_train = self.base_clf.predict(np.reshape(x_train, (x_train.shape[0], -1)))
        self.base_clf_predict_y_train = to_onehot_encode(base_clf_predict_y_train, self.num_classes)

    def mixData(self):
        return





