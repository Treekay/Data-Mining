import numpy as np

from config.cfg import MAX_CAPTCHA, CHAR_SET_LEN


def char2pos(c):
    """
    字符转成位置信息
    :param c:  字符
    :return:  位置信息
    """
    if c == '_':
        k = 62
        return k

    k = ord(c) - 48  # 48 = ord('0')
    if k > 9:
        k = ord(c) - 55  # 55 = ord('A') - 10
        if k > 35:
            k = ord(c) - 61  # 61 = ord('a') - (10 + 26)
            if k > 61:
                raise ValueError('No Map')
    return k


def pos2char(char_idx):
    """
    根据位置信息转化为字符
    :param char_idx: 位置信息
    :return: 字符
    """
    if char_idx < 10:
        char_code = char_idx + 48  # 48 = ord('0')
    elif char_idx < 36:
        char_code = char_idx + 55  # 55 = ord('A') - 10
    elif char_idx < 62:
        char_code = char_idx + 61  # 61= ord('a') - 36
    elif char_idx == 62:
        char_code = ord('_')
    else:
        raise ValueError('error')

    return chr(char_code)


def rgb2gray(img):
    """
    把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    :param img: 彩色图像
    :return: 灰度图像
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def increase_contrast(image):
    """
    提升灰度图片对比度
    :param image: 灰度图像数据
    :return: 高对比度数据
    """
    max_value = image.max()
    min_value = image.min()
    d_value = max_value - min_value

    image -= min_value
    image /= d_value

    return image


def text2vec(text):
    """
    文本转向量
    :param text: 文本
    :return: 向量
    """
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长%d个字符' % MAX_CAPTCHA)

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    """
    向量转文本
    :param vec: 向量
    :return: 文本
    """
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN

        char_code = pos2char(char_idx)

        text.append(char_code)
    return "".join(text)


if __name__ == '__main__':
    t_text = 'Xo8P'
    t_vec = text2vec(t_text)
    print(t_vec)
    t_text = vec2text(t_vec)
    print(t_text)
