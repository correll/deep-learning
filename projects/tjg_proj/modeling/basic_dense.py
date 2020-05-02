import tensorflow as tf
import keras
import pickle
import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, LSTM
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adam

# For running on GPU.
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.model_selection import train_test_split


class BasicDense(object):
    def __init__(self, data_path="../exports/as_pkl/data_dict_left-right.pkl"):
        self.data = pickle.load(open(data_path, "rb"))
        self.params = self.get_param_dict()
        self.set_up_session()
        X_train, X_test, y_train, y_test = self.organize_train_test_sets()
        self.make_model(X_train, X_test, y_train, y_test, self.params)


    def make_model(self, X_train, X_test, y_train, y_test, params):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=5, input_shape=params["INPUT_SHAPE"]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=5))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(20, 20)))

        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation("relu"))

        model.add(Dense(params["CLASSES"]))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer='rmsprop',
                      metrics=["accuracy"])
        model.summary()

        history = model.fit(X_train, y_train,batch_size=params["BATCH_SIZE"],
                            epochs=params["EPOCHS"], verbose=1,
                            validation_split=params["VALIDATION_SPLIT"])
        score = model.evaluate(X_test, y_test, verbose=1)
        print(score)


    def get_param_dict(self):

        EPOCHS = 50
        BATCH_SIZE = 16
        VERBOSE = 1
        OPTIMIZER = Adam()
        VALIDATION_SPLIT=0.15
        IMG_ROWS, IMG_COLS = 460, 96 # input image dimensions
        CLASSES = 3 # number of outputs = number of digits
        INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

        param_dict = {"EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
                      "VERBOSE": VERBOSE, "OPTIMIZER": Adam(),
                      "VALIDATION_SPLIT": VALIDATION_SPLIT, "IMG_ROWS": IMG_ROWS,
                      "IMG_COLS": IMG_COLS, "CLASSES": CLASSES,
                      "INPUT_SHAPE": INPUT_SHAPE}

        return param_dict


    def set_up_session(self):

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)


    def organize_train_test_sets(self):

        samples = []
        labels = []
        for val in self.data.values():
            for k, v in val.items():
                if len(v) == 2:
                    for elm in v:
                        print(len(elm))

        for sample in samples:
            print(sample.shape)

        # samples = samples.astype("float32")
        for i, sample in enumerate(samples):
            if len(sample) != 460:
                samples.remove(sample)
                labels.pop(i)
            else:
                sample = sample.astype("float32")

        for label in labels:
            label = label.astype("float32")


        samples = np.array([np.array(xi) for xi in samples])
        labels = np.array([np.array(xi) for xi in labels])

        samples = np.stack(samples)
        labels = np.stack(labels)

        print(samples[1].shape)

        return train_test_split(samples, labels, shuffle=True, test_size=0.15)




BasicDense()
