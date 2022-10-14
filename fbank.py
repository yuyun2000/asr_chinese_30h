import numpy as np
import cv2
from prepro import compute_log_mel_fbank
import os
dir = os.listdir('./data_thchs30/train')

for i in range(len(dir)):
    if 'WAV' not in dir[i]:
        continue
    file = './data_thchs30/train/%s' % dir[i]
    out = compute_log_mel_fbank(file)
    img = cv2.resize(out, (80, 1400),interpolation=cv2.INTER_AREA).astype(np.float32)
    img = img.reshape(1400, 80, 1)
    np.save('./data_thchs30/trainpro/%s'%dir[i][:-4],img)


