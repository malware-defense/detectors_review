from __future__ import division, absolute_import, print_function
import argparse

import numpy as np

from common.util import *
from setup_paths import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from lid.util import (random_split, block_split, train_lr, compute_roc, get_lids_random_batch, get_noisy_samples)
from keras import optimizers
from keras.metrics import categorical_crossentropy
from fs.datasets.datasets_utils import calculate_accuracy

from settings import config
import logging
import datetime
from androguard.core.androconf import show_logging
from datasets.apks import ApkData

#method from the original paper gitub code available on /lid folder
def get_lid(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Get local intrinsic dimensionality
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy, X_test_adv, dataset, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels


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
        from model.nn_tensorflow import NnTensorflow as myModel
        model_class = myModel(dataset, config['model_save_dir'], filename='nn_{}_{}_tensorflow.h5'.format(args.detection, args.classifier),
                              mode = args.mode)
        model=model_class.model


    x_train, y_train, x_test, y_test, x_adv, y_adv = \
        dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, dataset.x_adv, dataset.y_adv

    logging.info(green('Test the target model...'))
    y_pred = model.predict(x_test)
    accuracy_all = calculate_accuracy(y_pred, y_test)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))

    if args.detection == "drebin" or args.detection == "apigraph":
        x_test = x_test.toarray()
        x_adv = x_adv.toarray()

    if args.Random_sample:
        random.seed(42)
        sample_num = len(x_adv)
        sample_list = [i for i in range(len(x_test))]
        sample_list = random.sample(sample_list, sample_num)
        x_test = [x_test[i] for i in sample_list]
        y_test = [y_test[i] for i in sample_list]


    x_test = np.asarray(x_test)
    x_adv = np.asarray(x_adv)

    # as there are some parameters to tune for noisy example, so put the generation
    # step here instead of the adversarial step which can take many hours
    print('Crafting %s noisy samples. ' % args.dataset)
    X_test_noisy = get_noisy_samples(x_test, x_adv, args.dataset, 'fgsm_0.03125')


    detection_classifier_attacker_dir = os.path.join(config['lid_result_dir'],
                                                     "_".join([args.detection, args.classifier, args.attacker]))
    if not os.path.exists(detection_classifier_attacker_dir):
        os.makedirs(detection_classifier_attacker_dir, exist_ok=True)

    lid_file_X = os.path.join(detection_classifier_attacker_dir, '{}_{}_{}_lid_X.npy'.format(args.detection, args.classifier, args.attacker))
    lid_file_Y = os.path.join(detection_classifier_attacker_dir, '{}_{}_{}_lid_Y.npy'.format(args.detection, args.classifier, args.attacker))
    if os.path.isfile(lid_file_X) & os.path.isfile(lid_file_Y):
        X = np.load(lid_file_X)
        Y = np.load(lid_file_Y)
    else:
        X, Y = get_lid(model, x_test, X_test_noisy, x_adv, k_nn[DATASETS.index(args.dataset)], 100, args.dataset)
        np.save(lid_file_X, X)
        np.save(lid_file_Y, Y)
    
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X) # standarization

    print("LID: [characteristic shape: ", X.shape, ", label shape: ", Y.shape)
    # test attack is the same as training attack
    x_lr_train, y_lr_train, x_lr_test, y_lr_test = block_split(X, Y)
    print("Train data size: ", x_lr_train.shape)
    print("Test data size: ", x_lr_test.shape)

    ## Build detector
    print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" % (args.dataset, args.attacker, 'None'))
    lr = train_lr(x_lr_train, y_lr_train)
    
    
    #Split
    n_samples = int(len(x_lr_test)/3)
    x_lr_normal=x_lr_test[:n_samples]
    x_lr_noise=x_lr_test[n_samples:n_samples*2]
    x_lr_adv=x_lr_test[n_samples*2:]
    x_test_all = np.concatenate([x_lr_normal, x_lr_adv])

    y_lr_normal=y_lr_test[:n_samples]
    y_lr_noise=y_lr_test[n_samples:n_samples*2]
    y_lr_adv=y_lr_test[n_samples*2:]
    y_test_all = np.concatenate([y_lr_normal, y_lr_adv])

    ## Evaluate detector on adversarial attack
    y_pred_all = lr.predict_proba(x_test_all)[:, 1]
    y_pred_all_label = lr.predict(x_test_all)

    results_all = []
    #for Y_all
    acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(y_test_all[:][:,0], y_pred_all_label)
    fprs_all, tprs_all, thresholds_all = roc_curve(y_test_all[:][:,0], y_pred_all)
    roc_auc_all = auc(fprs_all, tprs_all)
    print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

    accuracy, precision, recall, F1 = detection_evalulate_metric2(y_test_all[:][:,0], y_pred_all_label)



    log_save_path = os.path.join(detection_classifier_attacker_dir, 'results_{}.log'.format(datetime.datetime.now()))
    with open(f'{log_save_path}', 'a+') as f1:
        f1.write(f'AUC: {roc_auc_all}\n')
        f1.write(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {F1}\n')
        f1.write('\n')
    logging.info(blue('Done!'))


def detection_evalulate_metric2(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred)
    precision = precision_score(Y_detect_test, Y_detect_pred)
    recall = recall_score(Y_detect_test, Y_detect_pred)
    F1 = float(2 * precision * recall / (precision + recall))
    return accuracy, precision, recall, F1



def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('--mode', type=str, default="train", help='Train or load the model.')

    # Choose the target android dataset
    p.add_argument('--dataset', type=str, default="drebin", help='The target malware dataset.')

    # Choose the target feature extraction method
    p.add_argument('--detection', type=str, default="mamadroid", help='The target malware feature extraction method.')

    # Choose the target classifier
    p.add_argument('--classifier', type=str, default="rf", help='The target malware classifier.')

    # Choose the attack method
    p.add_argument('--attacker', type=str, default="AdvDroidZero", help='The attack method.')

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




if __name__ == "__main__":
    main()
