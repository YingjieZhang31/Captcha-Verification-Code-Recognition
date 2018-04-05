
from captcha.image import ImageCaptcha
from random import sample

image = ImageCaptcha() 
characters =  list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

                   #QWERTYUIOPLKJHGFDSAZXCVBNM

def generate_data(digits_num, output, total):
    num = 0
    while(num<total):
        cur_cap = sample(characters, digits_num)
        cur_cap =''.join(cur_cap)
        _ = image.generate(cur_cap)
        image.write(cur_cap, output+cur_cap+".png")
        num += 1

generate_data(4, "images/", 50000)  #产生四个字符长度的验证码

