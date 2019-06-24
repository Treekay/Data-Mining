import os
import time

import numpy as np
import tensorflow as tf

from config.cfg import MAX_CAPTCHA, CHAR_SET_LEN, ckpt_path, IMAGE_WIDTH, IMAGE_HEIGHT
from data.gen_captcha import gen_captcha_text_and_image
from nn.cnn import crack_captcha_cnn, X, keep_prob
from utils.utils import rgb2gray, vec2text, increase_contrast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def crack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，输出预测文本
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)


def batch_hack_captcha():
    """
    批量生成验证码，然后再批量进行识别
    :return:
    """
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))  # 加载最后的模型参数设置

        stime = time.time()  # 开始时间
        task_cnt = 1000
        right_cnt = 0
        for i in range(task_cnt):
            text, image = gen_captcha_text_and_image(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            #  彩色图片转为灰度图片
            image = rgb2gray(image)
            #  提高灰度图片对比度
            image = increase_contrast(image)
            image = image.flatten()
            predict_text = crack_function(sess, predict, image)
            if text == predict_text:
                right_cnt += 1
            else:
                print("标记: {}  预测: {}".format(text, predict_text))
                pass

        print('task:', task_cnt, ' cost time:', (time.time() - stime), 's')
        print('right/total-----', right_cnt, '/', task_cnt)


if __name__ == '__main__':
    batch_hack_captcha()
    print('end...')
    pass
