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

# from tensorflow.python.platform import flags
# FLAGS = flags.FLAGS
# flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')

def get_distance(model, dataset, X1):
    X1_pred = model.predict(X1)
    vals_squeezed = []

    if dataset == 'mnist':
        X1_seqeezed_bit = bit_depth_py(X1, 1)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
    elif dataset == 'drebin':
        X1_seqeezed_bit = bit_depth_py(X1, 1)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
        # X1_seqeezed_filter_local = non_local_means_color_py(X1, 6, 3, 2)
        # vals_squeezed.append(model.predict(X1_seqeezed_filter_local))
    else:
        X1_seqeezed_bit = bit_depth_py(X1, 5)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
        X1_seqeezed_filter_local = non_local_means_color_py(X1, 13, 3, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_local))

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

def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    assert os.path.isfile('{}nn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    if args.dataset == 'drebin':
        from baselineCNN.nn.nn_drebin import DREBINNN as myModel
        model_class = myModel(mode='load', filename='nn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    # Load the dataset
    X_train_all, Y_train_all, X_test_all, Y_test_all = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test
    
    #--------------
    # Evaluate the trained model.
    # Refine the normal and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    print ("Evaluating the pre-trained model...")
    Y_pred_all = model.predict(X_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    X_test = X_test_all[inds_correct]
    Y_test = Y_test_all[inds_correct]
    Y_pred = Y_pred_all[inds_correct]

    # split valid and test
    indx_valid = random.sample(range(len(X_test)), int(len(X_test)/2))
    indx_test = list(set(range(0, len(X_test)))-set(indx_valid))

    x_valid = X_test[indx_valid]
    y_valid = Y_test[indx_valid]
    x_test = X_test
    y_test = Y_test
    #compute thresold - use test data to compute that
    threshold = train_fs(model, args.dataset, x_valid, 0.05)


    ## Evaluate detector
    #on adversarial attack
    for attack in ATTACKS:
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}.npy'.format(adv_data_dir, 'mama_family_testCW_data'))
        X_test_adv = np.reshape(X_test_adv, (-1, 11, 11, 1))
        Y_test_adv = np.load('{}{}.npy'.format(adv_data_dir, 'mama_family_testCW_label'))

        # filter adversarial samples according test example
        adv_inds_correct = inds_correct[np.where(inds_correct < len(Y_test_adv))]
        X_test_adv = X_test_adv[adv_inds_correct]
        Y_test_adv = Y_test_adv[adv_inds_correct]

        # prepare X and Y for detectors
        X_all = np.concatenate([x_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(x_test), dtype=bool), np.ones(len(X_test_adv), dtype=bool)])


        # 默认参数
        Y_all_pred, Y_all_pred_score = model_test(model, args.dataset, X_all, threshold)
        accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)


        #for Y_all
        # if attack == ATTACKS[0]:
        Y_all_pred_score = get_distance(model, args.dataset, X_all)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
        roc_auc_all = auc(fprs_all, tprs_all)
        maxindex = (tprs_all - fprs_all).tolist().index(max(tprs_all - fprs_all))
        threshold = thresholds_all[maxindex]


        Y_all_pred, Y_all_pred_score = model_test(model, args.dataset, X_all, threshold)
        accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)



        with open(f'{os.path.join("./results/fs", "results.log")}', 'a+') as f1:
            f1.write(
                f'AUC: {roc_auc_all}\n'
            )
            f1.write(
                f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {F1}\n'
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