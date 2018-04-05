import cv2
import numpy as np
from keras.models import load_model

character = ['A','B','C','D','E','F','G','H','I','J','K','L',
             'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

model1 = load_model(r'C:\Users\Administrator\Desktop\Side Project\Machine Learning\Keras\CNN_captcha验证码识别\model\cnn_model_1(good).h5')
model2 = load_model(r'C:\Users\Administrator\Desktop\Side Project\Machine Learning\Keras\CNN_captcha验证码识别\model\vgg2_model.h5')
model3 = load_model(r'C:\Users\Administrator\Desktop\Side Project\Machine Learning\Keras\CNN_captcha验证码识别\model\vgg3_model.h5')
model4 = load_model(r'C:\Users\Administrator\Desktop\Side Project\Machine Learning\Keras\CNN_captcha验证码识别\model\vgg_model_4.h5')

while(1):
    
    pic = input('输入要识别的验证码：')

    image = cv2.imread(pic,0)
    img = cv2.imread(pic,0)

    img = (img.reshape(1,1,60,160)).astype("float32")/255
    ###################################################
    predict = model1.predict_classes(img)
    tmp = predict[0]
    char = character[tmp]

    predict = model2.predict_classes(img)
    tmp = predict[0]
    char = char + character[tmp]

    predict = model3.predict_classes(img)
    tmp = predict[0]
    char = char + character[tmp]

    predict = model4.predict_classes(img)
    tmp = predict[0]
    char = char + character[tmp]

    print ('识别为：')
    print (char)

    cv2.imshow("test", image)
    cv2.waitKey(0)

    #predict = model.predict_classes(img)
    #print ('识别为：')
    #tmp = predict[0]
    #print (character[tmp])

    #cv2.imshow("test", image)
    #cv2.waitKey(0)
