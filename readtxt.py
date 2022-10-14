import numpy as np

from vocab import *
from prepro import compute_log_mel_fbank
import cv2
vocab = []
with open('./data/thchs_train.txt','r', encoding='utf8') as f:
    for line in f:
        name,pinyin,_ = line.split('\t')
        pinyin_list = pinyin.split(' ')
        res = []
        for x in pinyin_list:
            if len(x)>1 and x[-1].isdigit():
                x = x[:-1]
            if x in label_list:
                res.append(label_list.index(x))
        res = np.pad(res,(0,47-len(res)))
        # print(res)
        vocab.append(res)

        # out = compute_log_mel_fbank(name)
        # img = cv2.resize(out,(80,1400))
        # img = img.reshape(1400,80,1)

x = []
for i in range(len(vocab)):
    x.append(len(vocab[i]))
print(np.mean(x))
