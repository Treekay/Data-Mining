import random
import string

import matplotlib.pyplot as  plt
import numpy as  np
from PIL import Image
from captcha.image import ImageCaptcha

from config.cfg import gen_char_set, MAX_CAPTCHA

number = string.digits
Alphabet = string.ascii_uppercase
alphabet = string.ascii_lowercase


def random_captcha_text(char_set=number + alphabet + Alphabet, captcha_size=MAX_CAPTCHA):
    """
    生成n位验证码
    :param char_set:  源数据集
    :param captcha_size: 生成长度 默认:4
    :return:  n位验证码
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(char_set=gen_char_set, size=(160, 64)):
    """
    使用ImageCaptcha库生成验证码样本
    :return:(captcha_text, captcha_image)
    """
    image = ImageCaptcha(width=size[0], height=size[1])
    captcha_text = random_captcha_text(char_set)
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def __show_captcha_img(text, image):
    """
    使用matplotlib来显示生成的图片
    """
    print("验证码图像channel:", image.shape)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


if __name__ == '__main__':
    ##展示验证码
    text, image = gen_captcha_text_and_image()
    __show_captcha_img(text, image)
