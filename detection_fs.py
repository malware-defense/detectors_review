from __future__ import division, absolute_import, print_function
import argparse

from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fs.datasets.datasets_utils import calculate_accuracy
from fs.utils.squeeze import reduce_precision_py, bit_depth_py, median_filter_py, non_local_means_color_py
# from fs.utils.output import write_to_csv
# from fs.robustness import evaluate_robustness
from fs.detections.base import evalulate_detection_test
from keras import optimizers
from keras.metrics import categorical_crossentropy
from torch.utils.data import DataLoader,TensorDataset

from settings import config
import logging
import datetime
from androguard.core.androconf import show_logging
from datasets.apks import ApkData

# from tensorflow.python.platform import flags
# FLAGS = flags.FLAGS
# flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')

def get_distance(model, dataset, X1):
    X1_pred = model.predict(X1)
    vals_squeezed = []

    X1_seqeezed_bit = bit_depth_py(X1, 1)
    vals_squeezed.append(model.predict(X1_seqeezed_bit))
    X1_seqeezed_filter_median = median_filter_py(X1, 2)
    vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
    # X1_seqeezed_filter_local = non_local_means_color_py(X1, 6, 3, 2)
    # vals_squeezed.append(model.predict(X1_seqeezed_filter_local))

    dist_array = []
    for val_squeezed in vals_squeezed:
        dist = np.sum(np.abs(X1_pred - val_squeezed), axis=tuple(range(len(X1_pred.shape))[1:]))
        dist_array.append(dist)

    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)

def train_fs(model, dataset, X1, train_fpr):
    distances = get_distance(model, dataset, X1)
    selected_distance_idx = int(np.ceil(len(X1) * (1-train_fpr)))
    threshold = sorted(distances)[selected_distance_idx-1]
    threshold = threshold
    print ("Threshold value: %f" % threshold)
    return threshold

def model_test(model, dataset, X, threshold):
    distances = get_distance(model, dataset, X)
    Y_pred = distances > threshold
    return Y_pred, distances

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

    if args.Random_sample:
        random.seed(42)
        sample_num = len(x_adv)
        sample_list = [i for i in range(len(x_test))]
        sample_list = random.sample(sample_list, sample_num)
        x_test = [x_test[i] for i in sample_list]
        y_test = [y_test[i] for i in sample_list]


    # prepare X and Y for detectors
    X_all = np.concatenate([x_test, x_adv])
    Y_all = np.concatenate([np.zeros(len(x_test), dtype=bool), np.ones(len(x_adv), dtype=bool)])



    Y_all_pred_score = get_distance(model, args.dataset, X_all)
    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
    roc_auc_all = auc(fprs_all, tprs_all)
    maxindex = (tprs_all - fprs_all).tolist().index(max(tprs_all - fprs_all))
    threshold = thresholds_all[maxindex]


    Y_all_pred, Y_all_pred_score = model_test(model, args.dataset, X_all, threshold)
    accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)

    detection_classifier_attacker_dir = os.path.join(config['fs_result_dir'],
                                                     "_".join([args.detection, args.classifier, args.attacker]))
    if not os.path.exists(detection_classifier_attacker_dir):
        os.makedirs(detection_classifier_attacker_dir, exist_ok=True)
    log_save_path = os.path.join(detection_classifier_attacker_dir, 'results_{}.log'.format(datetime.datetime.now()))
    with open(f'{log_save_path}', 'a+') as f1:
        f1.write(f'AUC: {roc_auc_all}\n')
        f1.write(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {F1}\n')
        f1.write('\n')
    logging.info(blue('Done!'))


def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('--mode', type=str, default="train", help='Train or load the model.')

    # Choose the target android dataset
    p.add_argument('--dataset', type=str, default="apg", help='The target malware dataset.')

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