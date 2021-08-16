import numpy as np
import os
import cv2
from PIL import Image
dir = '../data/Train'
data = []
label = []

for labels in os.listdir(dir):
    print("Loading label: {}".format(labels))
    for y in os.listdir(dir+"/"+str(labels)):
        img = Image.open(dir+"/"+str(labels)+"/"+y)
        img = img.resize((32 ,32))
        img = np.array(img)
        data.append(img)
        label.append(labels)

data = np.array(data)
label = np.array(label)

np.save('./data',data)
np.save('./label',label)