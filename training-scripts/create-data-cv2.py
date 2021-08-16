import numpy as np
import os
import cv2
from PIL import Image
dir = './data/Train'
data = []
label = []

for labels in os.listdir(dir):
    print("Loading label: {}".format(labels))
    for y in os.listdir(dir+"/"+str(labels)):
        img = cv2.imread(dir+"/"+str(labels)+"/"+y, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32 ,32))
        img = np.array(img)
        data.append(img)
        label.append(labels)

data = np.array(data)
label = np.array(label)

np.save('./data-cv2',data)
np.save('./label-cv2',label)