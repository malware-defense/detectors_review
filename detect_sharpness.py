from __future__ import division, absolute_import, print_function
import argparse

import numpy as np
import seaborn

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

def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    assert os.path.isfile('{}nn_{}_th.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'drebin':
        from baselineCNN.nn.nn_drebin_th import DrebinNnTorch as myModel
        model_class = myModel(mode='load', filename='nn_{}_th.h5'.format(args.dataset))
        model=model_class.model
        # import joblib
        # model = joblib.load('./model/model.m')  # load

    # Load the dataset
    X_train_all, Y_train_all, X_test_all, Y_test_all, train_loader, test_loader = \
        model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test, model_class.train_loader, model_class.test_loader

    #--------------
    # Evaluate the trained model.
    # Refine the normal and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    print ("Evaluating the pre-trained model...")
    Y_pred_all = model(torch.tensor(X_test_all).float()).detach().numpy()
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
    X_test = X_test_all[inds_correct]
    Y_test = Y_test_all[inds_correct]
    Y_pred = Y_pred_all[inds_correct]
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float()),
                             shuffle=True,batch_size=10)


    # ======================  Generate perturbation data ========================
    # Settings
    adv_init_mag = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  #, 0.02, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1
    adv_lr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]   # , 0.02, 0.03, 0.04, 0.01, 0.06, 0.07, 0.08, 0.09, 0.1
    perturbation_steps = 25
    detect_type = 'loss'


    for attack in ATTACKS:
        # load adversarial samples
        X_test_adv = np.load('{}{}.npy'.format(adv_data_dir, 'mama_family_testCW_data'))
        X_test_adv = np.reshape(X_test_adv, (-1, 11, 11, 1))
        Y_test_adv = np.load('{}{}.npy'.format(adv_data_dir, 'mama_family_testCW_label'))

        # filter adversarial samples according test example
        adv_inds_correct = inds_correct[np.where(inds_correct < len(Y_test_adv))]
        X_test_adv = X_test_adv[adv_inds_correct]
        Y_test_adv = Y_test_adv[adv_inds_correct]

        # filter adversarial samples【attack failure】
        # Y_pred_adv = model.predict(X_test_adv)
        # Y_pred_adv = model(torch.tensor(X_test_adv).float()).detach().numpy()
        # attack_success = np.where(Y_pred_adv != Y_test_adv)[0]
        # X_test_adv = X_test_adv[attack_success]
        # Y_test_adv = Y_test_adv[attack_success]


        test_adv_loader = DataLoader(TensorDataset(torch.tensor(X_test_adv).float(), torch.tensor(Y_test_adv).float()),
                                 shuffle=True, batch_size=10)

        max_F1 = 0
        for init_mag in adv_init_mag:
            for lr in adv_lr:

                print('init_mag: %4f, lr: %4f', (init_mag, lr))
                with open(f'{os.path.join("./results/sharpness", "results.log")}', 'a+') as f1:
                    f1.write(f'################################ init_mag:{init_mag},lr:{lr} ################################\n')
                    f1.write(f'################################ init_mag:{init_mag},lr:{lr} ################################\n')

                path = f'./results/sharpness/perturb_figure/lr_{lr}_init_mag_{init_mag}'
                folder = os.path.exists(path)
                if not folder:
                    os.makedirs(path)

                # perturb on test sample
                perturbation_step_list, test_label_not_flip_rate, test_criteria_all = \
                    attack_label(model, test_loader, init_mag, lr, perturbation_steps, detect_type)

                # perturb on adv sample
                _, adv_label_not_flip_rate, adv_criteria_all =\
                    attack_label(model, test_adv_loader, init_mag, lr, perturbation_steps, detect_type)

                # draw flip_rate [lr, init_mag]
                l1 = plt.plot(perturbation_step_list, test_label_not_flip_rate, 's-', color='r', label='normal_example')
                l2 = plt.plot(perturbation_step_list, adv_label_not_flip_rate, 'o-', color='g', label='adv_example')
                plt.ylabel("label not flip rate")
                plt.xlabel("perturbation steps")
                plt.title(f'adv_lr is {lr}, adv_init_mag is {init_mag}')
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(f'./results/sharpness/perturb_figure/lr_{lr}_init_mag_{init_mag}/flip_rate.png')
                plt.cla()


                # prepare X and Y for detectors
                for step in range(perturbation_steps):
                    test_criteria = test_criteria_all[step]
                    adv_criteria = adv_criteria_all[step]
                    X_all = np.concatenate([test_criteria.detach().numpy(), adv_criteria.detach().numpy()])
                    X_all = normalize_criteria(torch.tensor(X_all)).detach().numpy()
                    Y_all = np.concatenate([np.zeros(len(test_criteria), dtype=bool), np.ones(len(adv_criteria), dtype=bool)])

                    # calculate ROC、AUC
                    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, X_all)
                    roc_auc_all = auc(fprs_all, tprs_all)
                    plt.plot(fprs_all, tprs_all, lw=1.5, label="ROC, AUC=%.3f)" % roc_auc_all)
                    plt.xlabel("FPR", fontsize=15)
                    plt.ylabel("TPR", fontsize=15)
                    plt.title(f'adv_lr is {lr}, adv_init_mag is {init_mag}')
                    plt.legend(loc="best")
                    plt.tight_layout()
                    plt.savefig(f'./results/sharpness/perturb_figure/lr_{lr}_init_mag_{init_mag}/roc_{step}.png')
                    plt.cla()


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

                    # draw loss
                    vars_criteria = {'adv': X_all[len(test_criteria) :],
                                     'test': X_all[0: len(test_criteria)]}
                    seaborn.displot(vars_criteria)
                    plt.xlabel(f'Criteria: detect type is loss, step is {step}')
                    plt.tight_layout()
                    plt.savefig(
                        f'./results/sharpness/perturb_figure/lr_{lr}_init_mag_{init_mag}/loss_step_{step}.png')
                    plt.cla()


                    # save result
                    with open(f'{os.path.join("./results/sharpness", "results.log")}', 'a+') as f1:
                        f1.write('INFO: ----------------args----------------\n')
                        f1.write(f'扰动次数 step: {step}\n')
                        f1.write(f'取最优的阈值: {threshold}\n')
                        f1.write('INFO: ----------------results----------------\n')
                        f1.write(
                            f'AUC: {roc_auc_all}\n'
                        )
                        f1.write(
                            f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {F1}\n'
                        )
                        f1.write('\n')


        # save best result
        with open(f'{os.path.join("./results/sharpness", "results.log")}', 'a+') as f1:
            f1.write(f'args: {max_args}\n')
            f1.write(
                f'AUC: {max_auc}\n'
            )
            f1.write(
                f'Accuracy: {max_accuracy}\nPrecision: {max_precision}\nRecall: {max_recall}\nF1: {max_F1}\n'
            )
            f1.write('\n')
    print('Done!')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)