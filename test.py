import numpy
import tensorflow as tf
import cv2
import numpy as np
from prepro import compute_log_mel_fbank
from vocab import *



model = tf.keras.models.load_model("./h5/zw-60.h5")

# out = compute_log_mel_fbank('./data/wav/test/D11/D11_751.wav')

out = compute_log_mel_fbank('data_thchs30/train/A11_137.wav')
img = cv2.resize(out, (80, 1400), interpolation=cv2.INTER_AREA).astype(np.float32)
# img = img / 255
img = img.reshape(1, 1400, 80, 1)

out = model(img)
out = np.argmax(out, axis=-1).reshape(88)
# print(out)
mask = out != 0
out = out[mask]

txet = []
for i in range(len(out)):
    txet.append(label_list[out[i]])
for i in range(len(txet)-1):
    if txet[i] != txet[i+1]:
        print(txet[i],end=' ')

