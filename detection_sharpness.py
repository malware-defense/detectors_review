from __future__ import division, absolute_import, print_function
import argparse

import numpy as np
import seaborn
from datasets.apks import ApkData
import logging
import datetime
import matplotlib.pyplot as plt
from androguard.core.androconf import show_logging
import random

from settings import config
from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fs.datasets.datasets_utils import calculate_accuracy
from sharpness.attack.attack_by_sharpness import attack_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset



def normalize_criteria(criteria):
    outmap_min, _ = torch.min(criteria, dim = 0, keepdim=True)
    outmap_max, _ = torch.max(criteria, dim = 0, keepdim=True)
    outmap = (criteria - outmap_min) / (outmap_max - outmap_min)
    return outmap


def model_test(X_all, threshold):
    Y_pred = X_all > threshold
    return Y_pred

def main():
    args = parse_args()
    show_logging(logging.INFO)

    dataset = ApkData(args.detection, args.classifier, args.attacker)
    if args.Implement_way == 'pytorch':
        from model.nn_pytorch import NnTorch as myModel
        model_class = myModel(dataset, config['model_save_dir'], filename='nn_{}_{}_pytorch.h5'.format(args.detection, args.classifier),
                              mode = args.mode)
        model = model_class.model
    else:
        from model.nn_pytorch import NnTorch as myModel

    x_train, y_train, x_test, y_test, x_adv, y_adv = \
        dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, dataset.x_adv, dataset.y_adv


    if args.Random_sample:
        random.seed(42)
        sample_num = len(x_adv)
        sample_list = [i for i in range(len(x_test))]
        sample_list = random.sample(sample_list, sample_num)
        x_test = [x_test[i] for i in sample_list]
        y_test = [y_test[i] for i in sample_list]
        test_loader = DataLoader(TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float()),
                                      shuffle=True, batch_size=10)
    else:
        test_loader = dataset.test_loader

    adv_loader = DataLoader(TensorDataset(torch.tensor(x_adv).float(), torch.tensor(y_adv).float()),
                             shuffle=True,batch_size=10)

    logging.info(green('Test the target model...'))
    y_pred = model(torch.tensor(x_test).float()).detach().numpy()
    accuracy_all = calculate_accuracy(y_pred, y_test)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))

    if args.Detect_sharpness:
        # ======================  Generate perturbation data ========================
        # Settings
        adv_init_mag = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  #, 0.02, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
        adv_lr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]   # , 0.02, 0.03, 0.04, 0.01, 0.06, 0.07, 0.08, 0.09, 0.1
        perturbation_steps = 25
        detect_type = 'loss'


        detection_classifier_attacker_dir = os.path.join(config['sharpness_result_dir'], "_".join([args.detection, args.classifier, args.attacker]))
        if not os.path.exists(detection_classifier_attacker_dir):
            os.makedirs(detection_classifier_attacker_dir, exist_ok=True)
        log_save_path = os.path.join(detection_classifier_attacker_dir, 'results_{}.log'.format(datetime.datetime.now()))
        max_F1 = 0
        for init_mag in adv_init_mag:
            for lr in adv_lr:
                with open(log_save_path, 'a+') as f1:
                    f1.write(f'init_mag:{init_mag}, lr:{lr}\n')

                logging.info(magenta('init_mag: %2f, lr: %2f' % (init_mag, lr)))
                logging.info(blue('Begin Detect ------- Building the gradient attack'))

                # perturb on test sample
                perturbation_step_list, test_label_not_flip_rate, test_criteria_all = \
                    attack_label(model, test_loader, init_mag, lr, perturbation_steps, detect_type)

                # perturb on adv sample
                _, adv_label_not_flip_rate, adv_criteria_all =\
                    attack_label(model, adv_loader, init_mag, lr, perturbation_steps, detect_type)

                if args.Save_perturb_figure:
                    fig_save_path = os.path.join(detection_classifier_attacker_dir, 'perturb_figure', 'lr_{}_init_mag_{}'.format(lr, init_mag))
                    if not os.path.exists(fig_save_path):
                        os.makedirs(fig_save_path, exist_ok=True)
                    draw_flip_rate_fig(adv_label_not_flip_rate, init_mag, lr, perturbation_step_list, test_label_not_flip_rate, fig_save_path)


                logging.info(blue('Begin Detect ------- Detect sample after attack'))
                for step in range(perturbation_steps):
                    test_criteria = test_criteria_all[step]
                    adv_criteria = adv_criteria_all[step]
                    X_all = np.concatenate([test_criteria.detach().numpy(), adv_criteria.detach().numpy()])
                    X_all = normalize_criteria(torch.tensor(X_all)).detach().numpy()
                    Y_all = np.concatenate([np.zeros(len(test_criteria), dtype=bool), np.ones(len(adv_criteria), dtype=bool)])



                    fprs_all, roc_auc_all, thresholds_all, tprs_all = save_roc_auc_figure(X_all, Y_all, init_mag,
                                                                                          lr, step, fig_save_path)
                    draw_loss_figure(X_all, init_mag, lr, step, test_criteria, fig_save_path)


                    maxindex = (tprs_all - fprs_all).tolist().index(max(tprs_all - fprs_all))
                    threshold = thresholds_all[maxindex]
                    Y_all_pred = model_test(X_all, threshold=threshold)
                    accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)

                    if F1 >= max_F1:
                        max_F1 = F1
                        max_accuracy = accuracy
                        max_precision = precision
                        max_recall = recall
                        max_auc = roc_auc_all
                        max_args = 'init_mag: %.2f, lr: %.2f, step: %d' % (init_mag, lr, step)


                    with open(f'{log_save_path}', 'a+') as f1:
                        f1.write('INFO: ----------------args----------------\n')
                        f1.write(f'扰动次数 step: {step}\n')
                        f1.write(f'取最优的阈值: {threshold}\n')
                        f1.write('INFO: ----------------results----------------\n')
                        f1.write(f'AUC: {roc_auc_all}\n')
                        f1.write(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {F1}\n')
                        f1.write('\n')


        # save best result
        with open(f'{log_save_path}', 'a+') as f1:
            f1.write(f'max_args: {max_args}\n')
            f1.write(f'max_auc: {max_auc}\n')
            f1.write(f'max_accuracy: {max_accuracy}\nmax_precision: {max_precision}\nmax_recall: {max_recall}\nmax_F1: {max_F1}\n')
            f1.write('\n')
        logging.info(blue('Done!'))


def draw_loss_figure(X_all, init_mag, lr, step, test_criteria, fig_save_path):
    vars_criteria = {'adv': X_all[len(test_criteria):],
                     'test': X_all[0: len(test_criteria)]}
    seaborn.displot(vars_criteria)
    plt.xlabel(f'Criteria: detect type is loss, step is {step}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, 'loss_step_{}.png'.format(step)))
    plt.cla()


def save_roc_auc_figure(X_all, Y_all, init_mag, lr, step, fig_save_path):
    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, X_all)
    roc_auc_all = auc(fprs_all, tprs_all)
    plt.plot(fprs_all, tprs_all, lw=1.5, label="ROC, AUC=%.3f)" % roc_auc_all)
    plt.xlabel("FPR", fontsize=15)
    plt.ylabel("TPR", fontsize=15)
    plt.title(f'adv_lr is {lr}, adv_init_mag is {init_mag}')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, 'roc_{}.png'.format(step)))
    plt.cla()
    return fprs_all, roc_auc_all, thresholds_all, tprs_all


def draw_flip_rate_fig(adv_label_not_flip_rate, init_mag, lr, perturbation_step_list, test_label_not_flip_rate, fig_save_path):
    l1 = plt.plot(perturbation_step_list, test_label_not_flip_rate, 's-', color='r', label='normal_example')
    l2 = plt.plot(perturbation_step_list, adv_label_not_flip_rate, 'o-', color='g', label='adv_example')
    plt.ylabel("label not flip rate")
    plt.xlabel("perturbation steps")
    plt.title(f'adv_lr is {lr}, adv_init_mag is {init_mag}')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, 'flip_rate.png'))
    plt.cla()


def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('--mode', type=str, default="load", help='Train or load the model.')

    # Choose the target android dataset
    p.add_argument('--dataset', type=str, default="apg", help='The target malware dataset.')

    # Choose the target feature extraction method
    p.add_argument('--detection', type=str, default="mamadroid", help='The target malware feature extraction method.')

    # Choose the target classifier
    p.add_argument('--classifier', type=str, default="rf", help='The target malware classifier.')

    # Choose the attack method
    p.add_argument('--attacker', type=str, default="AdvDroidZero", help='The attack method.')

    # Choose the detect type
    p.add_argument('--detect_type', type=str, default="loss", help='Train or load the model.')

    # Attackers
    p.add_argument('--ADZ_attack', action='store_true', help='The AdvDroidZero Attack.')
    p.add_argument('-N', '--attack_num', type=int, default=100, help='The query budget.')
    p.add_argument('-S', '--attack_sample_num', type=int, default=100, help='The random sample num.')
    p.add_argument('--Save_feature_data', action='store_true', help='Save feature data')
    p.add_argument('--Implement_way', type=str, default="pytorch", help='Model implement way')
    p.add_argument('--Random_sample', action='store_true', help='randomly sampled test data or not.')
    p.add_argument('--Detect_sharpness', action='store_true', help='The Detect Sharpness.')
    p.add_argument('--Save_perturb_figure', action='store_true', help='The Detect Sharpness.')
    p.add_argument('--Save_ROC_AUC_figure', action='store_true', help='The Detect Sharpness.')

    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')

    args = p.parse_args()

    return args


if __name__ == '__main__':
    main()