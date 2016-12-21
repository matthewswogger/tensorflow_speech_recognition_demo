from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf

LEARNING_RATE = 0.0001
# training_iters = 300000  # steps
BATCH_SIZE = 64

WIDTH = 20  # mfcc features
HEIGHT = 80  # (max) length of utterance
CLASSES = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(BATCH_SIZE)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

# Network building
net = tflearn.input_data([None, WIDTH, HEIGHT])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, CLASSES, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=LEARNING_RATE,
                                                loss='categorical_crossentropy')

# Training

# add this "fix" for tensorflow version errors
# col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# for x in col:
#     tf.add_to_collection(tf.GraphKeys.VARIABLES, x )


model = tflearn.DNN(net, tensorboard_verbose=0)
###################################################################
# while 1: # training_iters
#     model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY),
#                                         show_metric=True, batch_size=BATCH_SIZE)
#     _y=model.predict(X)
#
# model.save("tflearn.lstm.model")
# print (_y)
# print (y)

#################################################################
# Training
EPOCHS = 20
epochs_performed = 0
for _ in xrange(100):
    # Fit model
    model.fit(trainX, trainY, n_epoch=EPOCHS, validation_set=(testX, testY),
                                        show_metric=True, batch_size=BATCH_SIZE)
    # Save model
    epochs_performed += 20
    save = "saved_model/epoch_{}.tfl".format(epochs_performed)
    # print (save)
    model.save(save)
    # model.save("saved_model/20_epoch.tfl")
