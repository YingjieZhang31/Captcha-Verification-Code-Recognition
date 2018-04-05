import os
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
###########################################################################
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data

images = []
for fn in os.listdir('./images'):
    if fn.endswith('.png'):
        fd = os.path.join('./images',fn)
        images.append(read_image(fd))
print('load success!')
X = np.array(images)
print (X.shape)

y = np.loadtxt('out.txt')
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 30)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

print("Changing format......")

X_train = X_train.reshape(-1, 1,60, 160)/255.
X_test = X_test.reshape(-1, 1,60, 160)/255.
y_train = np_utils.to_categorical(y_train, num_classes=26)
y_test = np_utils.to_categorical(y_test, num_classes=26)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

print("Changing succeeded!")
os.system("pause")
#########################################################################
model = Sequential()
##1:64
model.add(Convolution2D(
		nb_filter = 64,
		nb_row = 3,
		nb_col = 3,
		border_mode = 'same',
		input_shape=(1,60,160)
						)
		)
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(
					pool_size = (2,2),
					strides = (2,2),
					border_mode = 'same',
					)
		)

##2:128
model.add(Convolution2D(128, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, border_mode = 'same'))

##3:256
model.add(Convolution2D(256, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, border_mode = 'same'))

##4:512
model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, border_mode = 'same'))

##5:512
model.add(Convolution2D(512, 3, 3, border_mode = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(2, 2, border_mode = 'same'))

#####FC
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))
########################
adam = Adam(lr = 1e-4)

########################
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=30, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('cnn_model.h5')   # HDF5文件，pip install h5py
print('\nSuccessfully saved as cnn_model.h5')



