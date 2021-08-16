import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime

data = []
labels = []

data = np.load('./arrays/data-cv2.npy')
labels = np.load('./arrays/label-cv2.npy')



X_train, X_validate, Y_train, Y_validate = train_test_split(data, labels)

print(X_train.shape, X_validate.shape, Y_train.shape, Y_validate.shape)

Y_validate = to_categorical(Y_validate, 43)
Y_train = to_categorical(Y_train, 43)

print(X_train.shape, X_validate.shape, Y_train.shape, Y_validate.shape)
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_validate.reshape(X_validate.shape[0], 28, 28, 1)

def myModel():
    model = Sequential()
        
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=X_train.shape[1:], activation=tf.nn.relu))
    model.add(Conv2D(64, kernel_size=(5, 5), activation=tf.nn.relu))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu))
    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dropout(rate=0.5))

    model.add(Dense(43, activation=tf.nn.softmax))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = myModel()
print(model.summary())
model.fit(X_train, Y_train, epochs=3, validation_data=(X_validate, Y_validate),callbacks=[tensorboard_callback])
model.save('./models/model-cv2.h5')

