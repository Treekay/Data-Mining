"""
网络结构
"""
import math

import tensorflow as tf

from config.cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], 'X_input')
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN], 'Y_input')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """
    定义CNN
    """

    def weight_variable(shape) -> tf.Variable:
        """
        权重 初始化
        :param shape:
        :return:
        """
        initial = tf.random_normal(shape)
        return tf.Variable(w_alpha * initial)

    def bias_variable(shape) -> tf.Variable:
        """
        偏置项  初始化
        :param shape:
        :return:
        """
        initial = tf.random_normal(shape)
        return tf.Variable(b_alpha * initial)

    def variable_summaries(var: tf.Variable):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram('histogram', var)

    def conv2d(tensor, weight):
        """
        卷积  操作
        :param tensor: 张量
        :param weight: 权重
        :return:
        """
        return tf.nn.conv2d(tensor, weight, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(tensor):
        """
        2x2池化 操作
        :param tensor: 张量
        :return:
        """
        return tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def nn_layer(tensor, core_size, in_size, out_size, layer_name, activation_function=tf.nn.relu):
        """
        隐含层
        :param tensor: 特征数据
        :param core_size: 卷积核大小
        :param in_size: 输入数据的维度大小
        :param out_size: 输出数据的维度大小(=隐层神经元个数）
        :param layer_name: 命名空间
        :param activation_function: 激活函数（默认是relu)
        :return:
        """
        # 设置命名空间
        with tf.name_scope(layer_name):
            # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
            with tf.name_scope('weights'):
                weights = weight_variable([core_size, core_size, in_size, out_size])
                variable_summaries(weights)
            # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
            with tf.name_scope('biases'):
                biases = bias_variable([out_size])
                variable_summaries(biases)
            # 执行wx+b的线性计算，并且用直方图记录下来
            with tf.name_scope('linear_compute'):
                # 卷积操作
                preactivate = tf.nn.bias_add(conv2d(tensor, weights), biases)
                tf.summary.histogram('linear', preactivate)
            # 将线性输出经过激励函数，并将输出也用直方图记录下来
            activations = activation_function(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            # 池化操作
            pool = max_pool_2x2(activations)
            out = tf.nn.dropout(pool, keep_prob)
            # 返回隐含层的最终输出
            return out

    L1_NEU_NUM = 32
    L2_NEU_NUM = 64
    L3_NEU_NUM = 64
    L4_NEU_NUM = 256
    CONV_CORE_SIZE = 3
    MAX_POOL_NUM = 3
    FULL_LAYER_FEATURE_NUM = 1024

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        tf.summary.image('input', image_shaped_input, 6)

    # 3 conv layer
    conv1 = nn_layer(image_shaped_input, CONV_CORE_SIZE, 1, L1_NEU_NUM, 'layer_1', tf.nn.relu)

    conv2 = nn_layer(conv1, CONV_CORE_SIZE, L1_NEU_NUM, L2_NEU_NUM, 'layer_2', tf.nn.relu)

    conv3 = nn_layer(conv2, CONV_CORE_SIZE, L2_NEU_NUM, L3_NEU_NUM, 'layer_3', tf.nn.relu)

    # conv4 = nn_layer(conv3, CONV_CORE_SIZE, L3_NEU_NUM, L4_NEU_NUM, 'layer_4', tf.nn.relu)

    convF = conv3

    # Fully connected layer
    r = int(math.ceil(IMAGE_HEIGHT / (2 ** MAX_POOL_NUM)) * math.ceil(IMAGE_WIDTH / (2 ** MAX_POOL_NUM)) * L3_NEU_NUM)
    w_fc1 = weight_variable([r, FULL_LAYER_FEATURE_NUM])
    b_fc1 = bias_variable([FULL_LAYER_FEATURE_NUM])
    h_pool3_flat = tf.reshape(convF, [-1, w_fc1.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, w_fc1), b_fc1))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([FULL_LAYER_FEATURE_NUM, MAX_CAPTCHA * CHAR_SET_LEN])
    b_fc2 = bias_variable([MAX_CAPTCHA * CHAR_SET_LEN])
    out = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)  # 36*4

    return out
