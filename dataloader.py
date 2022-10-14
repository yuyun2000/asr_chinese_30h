import tensorflow as tf
import numpy as np
import os
import cv2
from prepro import compute_log_mel_fbank
from vocab import *


def load_list(list_path='./data/thchs_train.txt'):
    wav = []
    labels = []
    seqlen = []
    labelslen = []
    with open(list_path, 'r', encoding='utf8') as f:
        for line in f:
            name, pinyin, _ = line.split('\t')
            pinyin_list = pinyin.split(' ')
            res = []
            for x in pinyin_list:
                if len(x)>1 and x[-1].isdigit():
                    x = x[:-1]
                if x in label_list:
                    res.append(label_list.index(x))
            labelslen.append(len(res))
            seqlen.append(88)
            res = np.pad(res, (0, 47 - len(res)))#.astype(np.float32)
            labels.append(res)
            wav.append(name)
    labels = np.array(labels)

    labels = tf.constant(labels, dtype=tf.int32)
    labelslen = tf.constant(labelslen, dtype=tf.int32)
    seqlen = tf.constant(seqlen, dtype=tf.int32)
    return wav, labels,labelslen,seqlen

def load_image(wav_path, label,labellen,seqlen):

    # out = compute_log_mel_fbank(wav_path.numpy().decode())
    # img = cv2.resize(out, (80, 1400),interpolation=cv2.INTER_AREA).astype(np.float32)
    # img = img.reshape(1400, 80, 1)

    file = wav_path.numpy().decode()
    img = np.load('./data_thchs30/trainpro/%s.npy'%file[19:-4])

    # labellen = tf.constant(labellen,dtype=tf.int64)
    # seqlen = tf.constant(seqlen,dtype=tf.int64)
    return img, label,labellen,seqlen


def train_iterator():
    images, labels,labelslen,seqlen = load_list()

    dataset = tf.data.Dataset.from_tensor_slices((images, labels,labelslen,seqlen)).shuffle(len(images))
    dataset = dataset.map(lambda x, y,z1,z2: tf.py_function(load_image, inp=[x, y,z1,z2], Tout=[tf.float32, tf.int32, tf.int32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(50).prefetch(1)
    it = dataset.__iter__()
    return it

if __name__ == '__main__':
    it = train_iterator()
    images, labels ,labellen,sqelen= it.next()
    print(labels[0],labellen[0])




