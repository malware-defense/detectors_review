from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *
from keras.layers import *

from keras.callbacks import *
from keras import optimizers, Model
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

class NnTensorflow:
    def __init__(self, dataset, model_save_dir, filename, mode='train', normalize_mean=False, epochs=50, batch_size=128):
        self.mode = mode  # train or load, main.py中根据参数决定
        self.model_save_dir = model_save_dir
        self.filename = filename
        self.normalize_mean = normalize_mean
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 2

        #====================== Model =============================
        self.input_shape = dataset.x_train.shape[1:]
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model, dataset)
        elif mode=='load':
            self.model.load_weights("{}{}".format(model_save_dir, self.filename))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def build_model(self):
        #================= Settings =========================
        weight_decay = 0.0005
        basic_dropout_rate = 0.3

        #================= Input ============================
        input = Input(shape=self.input_shape, name='l_0')

        #================= Dense ============================
        # task0 = Flatten(name='l_1')(input)
        task0 = Dense(128, kernel_regularizer=regularizers.l2(weight_decay), name='l_2')(input)
        task0 = Activation('relu', name='l_3')(task0)
        task0 = Dense(100, kernel_regularizer=regularizers.l2(weight_decay), name='l_4')(task0)
        task0 = Activation('relu', name='l_5')(task0)
        task0 = Dropout(basic_dropout_rate + 0.2, name='l_6')(task0)

        #================= Output - classification head ============================
        classification_output = Dense(self.num_classes, name="classification_head_before_activation")(task0)
        classification_output = Activation('softmax', name="classification_head")(classification_output)

        #================= The final model ============================
        model = Model(inputs=input, outputs=classification_output)
        return model
    
    def train(self, model, dataset):
        #================= Settings =========================
        learning_rate = 0.001

        weights_file = "{}{}".format(checkpoints_dir, self.filename)
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit(dataset.x_train, y=dataset.y_train, batch_size=self.batch_size, epochs=self.epochs)
        
        #================= Save model and history =========================
        with open("{}{}_history.pkl".format(checkpoints_dir, self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model.save_weights(weights_file)

        return model
