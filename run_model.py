from __future__ import division, print_function, absolute_import
import tflearn as tf
import speech_data
import numpy as np
from sklearn.cross_validation import train_test_split

def score_model(X, y):
    y_predicted = np.array(model.predict(X))
    bool_arr = np.argmax(y_predicted,axis=1) == np.argmax(np.array(y),axis=1)
    bool_sum = np.sum(bool_arr)
    return ('model accuracy: {}'.format(round(float(bool_sum)/bool_arr.shape[0],2)))


LEARNING_RATE = 0.0001
BATCH_SIZE = 64
WIDTH = 20  # mfcc features
HEIGHT = 80  # (max) length of utterance
CLASSES = 10  # digits

data_set = speech_data.mfcc_batch_generator(2400)
X, Y = next(data_set)
X, Y = np.array(X), np.array(Y)

# get train, test, validation split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2,
                                                                 random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.2, random_state=0)
# Network building
net = tf.input_data([None, WIDTH, HEIGHT])
net = tf.lstm(net, 128, dropout=0.8)
net = tf.fully_connected(net, CLASSES, activation='softmax')
net = tf.regression(net, optimizer='adam', learning_rate=LEARNING_RATE,
                                                loss='categorical_crossentropy')
model = tf.DNN(net, tensorboard_verbose=0)

# Training

# model.load("saved_model/epoch_2000.tfl")

# EPOCHS = 20
# epochs_performed = 2000
# for _ in xrange(50):
#     # Fit model
#     model.fit(X_train, y_train, n_epoch=EPOCHS, validation_set=(X_val, y_val),
#                                         show_metric=True, batch_size=BATCH_SIZE)
#     # Save model
#     epochs_performed += 20
#     model_name = "saved_model/epoch_{}.tfl".format(epochs_performed)
#     model.save(model_name)

# model evaluation
model.load("saved_model/epoch_3000.tfl")

print (score_model(X_test, y_test))
