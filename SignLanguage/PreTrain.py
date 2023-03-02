import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import Video as V

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(V.actions)}

sequences, labels = [], []
for action in V.actions:
    for sequence in np.array(os.listdir(os.path.join(V.DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(V.sequence_length):
            res = np.load(os.path.join(V.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

