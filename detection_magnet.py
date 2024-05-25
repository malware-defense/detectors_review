from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sklearn.metrics import accuracy_score, precision_score, recall_score
from magnet.defensive_models import DenoisingAutoEncoder as DAE
from magnet.worker import *
from keras import optimizers, Model
from keras.metrics import categorical_crossentropy

from settings import config
import logging
import datetime
from androguard.core.androconf import show_logging
from datasets.apks import ApkData
from fs.datasets.datasets_utils import calculate_accuracy

def test(dic, X, thrs):
    dist_all = []
    pred_labels = []
    for d in dic:
        m = dic[d].mark(X)#m = np.reshape(dic[d].mark(X), (len(X),1))
        dist_all.append(m)
        pred_labels.append(m>thrs[d])
    
    #idx_pass = np.argwhere(marks < thrs[name])
    labels = pred_labels[0]
    for i in range(1, len(pred_labels)):
        labels = labels | pred_labels[i]
    
    # dist = dist_all[0]
    # for i in range(1, len(dist_all)):
    #     dist = np.max(np.concatenate((dist, dist_all[i]), axis=1), axis=1)
    
    return labels, dist_all 


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


    clip_min, clip_max = 0, 1
    v_noise = 0.1
    p1 = 2
    p2 = 1
    type = 'error'
    t = 10
    drop_rate = {"I": 0.1, "II": 0.1}
    epochs = 10


    num_samples = x_train.shape[0]
    val_size = int(num_samples / 10)

    x_dae_val = x_train[:val_size, :]
    y_dae_val = y_train[:val_size]
    x_dae_train = x_train[val_size:, :]
    y_dae_train = y_train[val_size:]

    #Train detector -- if already trained, load it    key steps
    detection_classifier_attacker_dir = os.path.join(config['magnet_result_dir'],
                                                     "_".join([args.detection, args.classifier, args.attacker]))
    if not os.path.exists(detection_classifier_attacker_dir):
        os.makedirs(detection_classifier_attacker_dir, exist_ok=True)

    detector_i_filename = '{}_{}_{}_magnet_detector_i.h5'.format(args.detection, args.classifier, args.attacker)
    detector_ii_filename = '{}_{}_{}_magnet_detector_ii.h5'.format(args.detection, args.classifier, args.attacker)

    if args.detection == "mamadroid":
        x_dae_train.resize(x_dae_train.shape[0], 11, 11, 1)
        x_test.resize(x_test.shape[0], 11, 11, 1)
        x_dae_val.resize(x_dae_val.shape[0], 11, 11, 1)
        x_adv.resize(x_adv.shape[0], 11, 11, 1)
    elif args.detection == "drebin":
        x_dae_train.resize(x_dae_train.shape[0], 11, 11, 1)
        x_test.resize(x_test.shape[0], 11, 11, 1)
        x_dae_val.resize(x_dae_val.shape[0], 11, 11, 1)
        x_adv.resize(x_adv.shape[0], 11, 11, 1)
    elif args.detection == "apigraph":
        x_dae_train.resize(x_dae_train.shape[0], 11, 11, 1)
        x_test.resize(x_test.shape[0], 11, 11, 1)
        x_dae_val.resize(x_dae_val.shape[0], 11, 11, 1)
        x_adv.resize(x_adv.shape[0], 11, 11, 1)

    im_dim = [x_dae_train.shape[1], x_dae_train.shape[2], x_dae_train.shape[3]]
    detector_I = DAE(im_dim, [3, "average", 3], v_noise=v_noise, activation="sigmoid", model_dir=detection_classifier_attacker_dir, reg_strength=1e-9)
    detector_II = DAE(im_dim, [3], v_noise=v_noise, activation="sigmoid", model_dir=detection_classifier_attacker_dir, reg_strength=1e-9)
    if os.path.exists((os.path.join(detection_classifier_attacker_dir, detector_i_filename))):
        detector_I.load(detector_i_filename)
    else:
        detector_I.train(x_dae_train, x_test, detector_i_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)
    if os.path.exists((os.path.join(detection_classifier_attacker_dir, detector_ii_filename))):
        detector_II.load(detector_ii_filename)
    else:
        detector_II.train(x_dae_train, x_test, detector_ii_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)


    #Make AEs ready
    classifier = Classifier(model, model_class.num_classes)
    if type=='error':
        if args.dataset=='cifar':
            detect_I = AEDetector(detector_I.model, p=p1)
            detect_II = AEDetector(detector_I.model, p=p2)
            reformer = SimpleReformer(detector_II.model)
        else:
            detect_I = AEDetector(detector_I.model, p=p1)
            detect_II = AEDetector(detector_II.model, p=p2)
            reformer = SimpleReformer(detector_I.model)
        detector_dict = dict()
        detector_dict["I"] = detect_I
        detector_dict["II"] = detect_II
    elif type=='prob':
        reformer = SimpleReformer(detector_I.model)
        reformer2 = SimpleReformer(detector_II.model)
        detect_I = DBDetector(reformer, reformer2, classifier, T=t)
        detector_dict = dict()
        detector_dict["I"] = detect_I


    operator = Operator(x_dae_val, x_test, y_test, classifier, detector_dict, reformer)


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


    # --- get thresholds per detector
    testAttack = AttackData(x_adv, np.argmax(y_test, axis=1), args.attacker)
    evaluator = Evaluator(operator, testAttack)
    thrs = evaluator.operator.get_thrs(drop_rate)

    #For Y_all
    Y_all_pred, Y_all_pred_score = test(detector_dict, X_all, thrs)
    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score[0])
    roc_auc_all = auc(fprs_all, tprs_all)
    maxindex = (tprs_all - fprs_all).tolist().index(max(tprs_all - fprs_all))
    threshold = thresholds_all[maxindex]


    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score[1])
    roc_auc_all1 = auc(fprs_all, tprs_all)
    maxindex = (tprs_all - fprs_all).tolist().index(max(tprs_all - fprs_all))
    threshold1 = thresholds_all[maxindex]

    accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)

    # thrs['I'] = threshold
    # thrs['II'] = threshold1
    Y_all_pred, Y_all_pred_score = test(detector_dict, X_all, thrs)
    accuracy, precision, recall, F1 = detection_evalulate_metric(Y_all, Y_all_pred)

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