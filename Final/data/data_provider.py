import numpy as np

from config.cfg import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_CAPTCHA, CHAR_SET_LEN
from data.gen_captcha import gen_captcha_text_and_image
from utils.utils import rgb2gray, text2vec, increase_contrast


def get_next_batch(batch_size=128):
    """
    生成一个训练batch
    :param batch_size: batch大小
    :return: (image[],text[])
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = gen_captcha_text_and_image(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        image = rgb2gray(image)
        image = increase_contrast(image)
        batch_x[i, :] = image.flatten()
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y
