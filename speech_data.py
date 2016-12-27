"""
Utilities for downloading and providing data from openslr.org,
libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies.
"""
# TODO! see https://github.com/pannous/caffe-speech-recognition for some data sources

import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  # scikit-image
import librosa
import matplotlib

# import extensions as xx
from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


SOURCE_URL = 'http://pannous.net/files/' #spoken_numbers.tar'
DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/" # 8 bit
wav_path = "data/spoken_numbers_wav/" # 16 bit s16le
path = pcm_path
CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification

# http://pannous.net/files/spoken_numbers_pcm.tar
class Source:  # labels
  DIGIT_WAVES = 'spoken_numbers_pcm.tar'
  DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
  NUMBER_WAVES = 'spoken_numbers_wav.tar'
  NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
  WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width,height=512# todo: sliding window!
  TEST_INDEX = 'test_index.txt'
  TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3
  word_phonemes=4
  word=5#characters=5
  sentence=6
  sentiment=7
  first_letter=8


def progresshook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def maybe_download(file, work_directory):
  """Download the data from Pannous's website, unless it's already here."""
  print("Looking for data %s in %s"%(file,work_directory))
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, re.sub('.*\/','',file))
  if not os.path.exists(filepath):
    if not file.startswith("http"): url_filename = SOURCE_URL + file
    else: url_filename=file
    print('Downloading from %s to %s' % (url_filename, filepath))
    filepath, _ = urllib.request.urlretrieve(url_filename, filepath, progresshook)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
    # os.system('ln -s '+work_directory)
  if os.path.exists(filepath):
    print('Extracting %s to %s' % ( filepath, work_directory))
    os.system('tar xf %s -C %s' % ( filepath, work_directory))
    print('Data ready!')
  return filepath.replace(".tar","")


def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
  maybe_download(source, DATA_DIR)
  if target == Target.speaker:
      speakers = get_speakers()
  batch_features = []
  labels = []
  files = os.listdir(path)
  while True:
    print("loaded batch of %d files" % len(files))
    shuffle(files)
    for wav in files:
      if not wav.endswith(".wav"):
          continue
      wave, sr = librosa.load(path+wav, mono=True)

      if target==Target.speaker:
          label=one_hot_from_item(speaker(wav), speakers)
      elif target==Target.digits:
          label=dense_to_one_hot(int(wav[0]),10)
      elif target==Target.first_letter:
          label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
      else:
          raise Exception("todo : labels for Target!")
      labels.append(label)
      mfcc = librosa.feature.mfcc(wave, sr)
      # print(np.array(mfcc).shape)
      mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
      batch_features.append(np.array(mfcc))
      if len(batch_features) >= batch_size:
        # print(np.array(batch_features).shape)
        # yield np.array(batch_features), labels
        yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
        batch_features = []  # Reset for next batch
        labels = []


def one_hot_from_item(item, items):
  # items=set(items) # assure uniqueness
  x=[0]*len(items)# numpy.zeros(len(items))
  i=items.index(item)
  x[i]=1
  return x


def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
  output_shape = tf.reshape(concat, [2])
  sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
  return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  return numpy.eye(num_classes)[labels_dense]


if __name__ == "__main__":
  print("downloading speech datasets")
  maybe_download( Source.DIGIT_SPECTROS)
  maybe_download( Source.DIGIT_WAVES)
  maybe_download( Source.NUMBER_IMAGES)
  maybe_download( Source.NUMBER_WAVES)
