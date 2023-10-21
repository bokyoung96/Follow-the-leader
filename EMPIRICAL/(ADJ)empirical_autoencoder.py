"""
Article: Follow the leader: Index tracking with factor models

Topic: Empirical Analysis
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from empirical_loader import *


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

dir_name = "./ae_callback/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


class AutoEncoder(DataLoader):
    """
    <DESCRIPTION>
    Construct AutoEncoder under Tensorflow Keras.
    """

    def __init__(self,
                 mkt:  str = 'KOSPI200',
                 date: str = 'Y10',
                 test_size: int = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100):
        super().__init__(mkt, date)
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.idx, self.stocks = self.as_empirical(idx_weight='EQ')
        self.stocks_ret = self.stocks.pct_change(axis=1).fillna(0)

    @property
    def data_split(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Classify data into training & validation set.
        """
        data = self.stocks_ret
        x_train, x_val = train_test_split(data,
                                          test_size=self.test_size,
                                          random_state=42)
        return x_train, x_val

    def auto_encoder(self, output_dim: int = 25):
        """
        <DESCRIPTION>
        Construct AutoEncoder.
        """
        # DATA
        x_train, x_val = self.data_split

        # SHAPES & LAYERS
        input_dim = x_train.shape[1]
        layers = round(input_dim / 2)

        layer_sizes = []
        for layer in range(layers):
            layer_size = max(output_dim, int(input_dim / (2**layer)))
            if layer_size <= output_dim:
                break
            layer_sizes.append(layer_size)

        # INPUT LAYER
        input_layer = Input((input_dim, ))

        # ENCODE LAYER
        encode_layer = Dense(
            layer_sizes[1], activation='relu')(input_layer)
        for size in layer_sizes[2:]:
            encode_layer = Dense(size, activation='relu')(encode_layer)

        # LATENT LAYER
        latent_layer = Dense(
            output_dim, activation='sigmoid')(encode_layer)

        # DECODE LAYER
        layer_sizes.reverse()
        decode_layer = Dense(
            layer_sizes[0], activation='relu')(latent_layer)
        for size in layer_sizes[1:-1]:
            decode_layer = Dense(size, activation='relu')(decode_layer)
        decode_layer = Dense(
            layer_sizes[-1], activation='relu')(decode_layer)

        # MODEL
        model = Model(inputs=input_layer,
                      outputs=decode_layer)
        adam = Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False)
        model.compile(optimizer=adam, loss='mse')
        model.summary()

        # CALLBACKS
        checkpoints = ModelCheckpoint(filepath='./ae_callback/ae_callback_lr_{}.hdf5'.format(self.learning_rate),
                                      monitor='val_loss',
                                      save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.00001,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        callbacks = [checkpoints, early_stopping]

        # FITTING
        res = model.fit(x_train,
                        x_train,
                        epochs=self.epochs,
                        batch_size=32,
                        validation_data=(x_val, x_val),
                        callbacks=callbacks)

        stopped_epoch = early_stopping.stopped_epoch
        return model, res, stopped_epoch

    def auto_encoder_plot(self, res, stopped_epoch):
        """
        <DESCRIPTION>
        Plot graph for training & validation loss in AutoEncoder.
        """
        epoch = np.arange(1, len(res.history['loss'])+1)
        plt.figure(figsize=(15, 5))
        plt.plot(epoch, res.history['loss'],
                 linewidth=2, label='Training loss')
        plt.plot(epoch, res.history['val_loss'],
                 linewidth=2, label='Validation loss')
        plt.axvline(x=stopped_epoch, color='r', linestyle='--',
                    label='Early stopping epoch')
        plt.title('AutoEncoder performance')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    loader = AutoEncoder()
    model, res, stopped_epoch = loader.auto_encoder(output_dim=10)
    loader.auto_encoder_plot(res, stopped_epoch)
