import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


img = os.listdir('./images')
length = len(img)
for i in range(length):
    img[i] = img[i][2]   //img[x][y]代表第x张图片的第y位字母
#first = img[0][0]
print (length)
for i in range(length):
    label = ord(img[i])-65
    print (label)#ord():查看字符asc码
    doc= open('out.txt','a')
    print(label,file=doc)
doc.close()

