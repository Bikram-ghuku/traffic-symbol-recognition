from PIL import Image
from detect import get_data
import cv2
import os
import time

video = cv2.VideoCapture(0)
try:
    os.mkdir('./cur-img')
    print("Made cur image folder")
except FileExistsError:
    print("Folder already exists, skipped")
while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (32,32))
    cv2.imwrite('./cur-img/image.png', frame)
    image = Image.open('./cur-img/image.png')
    x = get_data(image)
    print(x)
    os.remove('./cur-img/image.png')
