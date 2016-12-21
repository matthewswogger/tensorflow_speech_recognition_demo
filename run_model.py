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

# Training
# EPOCHS = 20
# epochs_performed = 0
# for _ in xrange(100):
#     # Fit model
#     model.fit(trainX, trainY, n_epoch=EPOCHS, validation_set=(testX, testY),
#                                         show_metric=True, batch_size=BATCH_SIZE)
#     # Save model
#     epochs_performed += 20
#     model_name = "saved_model/epoch_{}.tfl".format(epochs_performed)
#     model.save(model_name)

model.load("saved_model/epoch_2000.tfl")
_y=model.predict(X)
print (_y[0])
print (Y[0])
