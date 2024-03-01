import keras
from keras.utils import to_categorical
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import codecs
import gc
import numpy as np
import pandas as pd
import time
import os
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import roc_auc_score



class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space use [unused1]
            else:
                R.append('[UNK]')  # rest string [UNK]
        return R


    def f1_loss(y_true, y_pred):
        loss = 2 * tf.reduce_sum(y_true * y_pred) / tf.reduce_sum(y_true + y_pred) + K.epsilon()
        return -loss


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1


    def __len__(self):
        return self.steps


    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))


            if self.shuffle:
                np.random.shuffle(idxs)


            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                # indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
                x1, x2 = tokenizer.encode(first=text)
                y = np.float32(d[1])
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # print('Y', Y)
                    yield [X1, X2], Y[:, 0]
                    [X1, X2, Y] = [], [], []




def build_bert(nclass, selfloss, lr, is_train):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)


    for l in bert_model.layers:
        l.trainable = is_train


    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))


    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, :])(x)


    avg_pool_3 = GlobalAveragePooling1D()(x)
    max_pool_3 = GlobalMaxPooling1D()(x)
    # https://www.cnpython.com/pypi/keras-self-attention
    # source https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_self_attention.py
    attention_3 = SeqSelfAttention(attention_activation='softmax')(x)
    attention_3 = Lambda(lambda x: x[:, 0])(attention_3)


    x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
    p = Dense(nclass, activation='sigmoid')(x)


    model = Model([x1_in, x2_in], p)
    model.compile(loss=selfloss,
                  optimizer=Adam(lr),
                  metrics=['acc'])
    print(model.summary())
    return model


def run_cv(nfold, data, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=2020).split(data)
    train_model_pred = np.zeros((len(data), 1))
    test_model_pred = np.zeros((len(data_test), 1))


    lr = 1e-7  # 1e-5
    # categorical_crossentropy (option：'binary_crossentropy', f1_loss)
    selfloss = f1_loss
    is_train = False  # True False


    for i, (train_fold, test_fold) in enumerate(kf):
        print('***************%d-th****************' % i)
        t = time.time()
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]


        model = build_bert(1, selfloss, lr, is_train)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('/home/codes/news_classify/comment_classify/expriments/' + str(i) + '_2.hdf5', monitor='val_acc',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=False)


        batch_size = 16
        train_D = data_generator(X_train, batch_size=batch_size, shuffle=True)
        valid_D = data_generator(X_valid, batch_size=batch_size, shuffle=False)
        test_D = data_generator(data_test, batch_size=batch_size, shuffle=False)


        model.load_weights('/home/codes/news_classify/comment_classify/expriments/' + str(i) + '.hdf5')


        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=8,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )


        # return model
        train_model_pred[test_fold] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)


        del model
        gc.collect()
        K.clear_session()


        print('time:', time.time()-t)


    return train_model_pred, test_model_pred


