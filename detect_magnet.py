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
        modelx = Model(inputs=model.get_layer('l_0').output, outputs=model.get_layer('classification_head_before_activation').output)
        clip_min, clip_max = 0,1
        v_noise=0.1
        p1=2
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.001, "II": 0.001}
        epochs=10

    elif args.dataset == 'drebin':
        from baselineCNN.nn.nn_drebin import DREBINNN as myModel
        model_class = myModel(mode='load', filename='nn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_layer('l_0').output,
                       outputs=model.get_layer('classification_head_before_activation').output)
        clip_min, clip_max = 0, 1
        v_noise = 0.1
        p1 = 2
        p2 = 1
        type = 'error'
        t = 10
        drop_rate = {"I": 0.001, "II": 0.001}
        epochs = 10

    elif args.dataset == 'cifar':
        from baselineCNN.nn.nn_drebin import DREBINNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_softmax').output)
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_activation').output)
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='prob'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_activation').output)
        # clip_min, clip_max = -2.117904,2.64
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test
    val_size = 1000
    x_val = X_train[:val_size, :, :, :]
    y_val = Y_train[:val_size]
    X_train = X_train[val_size:, :, :, :]
    Y_train = Y_train[val_size:]

    #Train detector -- if already trained, load it    key steps
    detector_i_filename = '{}_magnet_detector_i.h5'.format(args.dataset)
    detector_ii_filename = '{}_magnet_detector_ii.h5'.format(args.dataset)
    im_dim = [X_train.shape[1], X_train.shape[2], X_train.shape[3]]
    detector_I = DAE(im_dim, [3, "average", 3], v_noise=v_noise, activation="sigmoid", model_dir=magnet_results_dir, reg_strength=1e-9)
    detector_II = DAE(im_dim, [3], v_noise=v_noise, activation="sigmoid", model_dir=magnet_results_dir, reg_strength=1e-9)
    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_i_filename)):
        detector_I.load(detector_i_filename)
    else:
        detector_I.train(X_train, X_test, detector_i_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)
    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_ii_filename)):
        detector_II.load(detector_ii_filename)
    else:
        detector_II.train(X_train, X_test, detector_ii_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    print("X_test: ", X_test.shape)

    #Make AEs ready
    classifier = Classifier(modelx, model_class.num_classes)
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

    operator = Operator(x_val, X_test, Y_test, classifier, detector_dict, reformer)

    ## Evaluate detector
    #on adversarial attack
    Y_test_copy=Y_test
    X_test_copy=X_test
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
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
        X_all = np.concatenate([X_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(X_test), dtype=bool), np.ones(len(X_test_adv), dtype=bool)])


        # --- get thresholds per detector
        testAttack = AttackData(X_test_adv, np.argmax(Y_test, axis=1), attack)
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



        with open(f'{os.path.join("./results/magnet", "results.log")}', 'a+') as f1:
            f1.write(
                f'AUC: {roc_auc_all}\n'
            )
            f1.write(
                f'AUC1: {roc_auc_all1}\n'
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
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use train the discriminator; either  {}".format(ATTACK),
    #     required=False, type=str
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )

    # parser.set_defaults(batch_size=100)
    # parser.set_defaults(attack="cwi")
    args = parser.parse_args()
    main(args)