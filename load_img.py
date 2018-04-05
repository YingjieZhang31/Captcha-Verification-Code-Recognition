import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

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