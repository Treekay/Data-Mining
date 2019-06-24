import os

import tensorflow as tf

from config.cfg import MAX_CAPTCHA, CHAR_SET_LEN, tb_log_path, save_ckpt
from data.data_provider import get_next_batch
from nn.cnn import crack_captcha_cnn, Y, keep_prob, X

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_crack_captcha_cnn():
    """
    训练模型
    :return:
    """
    output = crack_captcha_cnn()
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(label, 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('my_monitor'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=label))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar('my_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    with tf.name_scope('my_monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('my_accuracy', accuracy)

    saver = tf.train.Saver()  # 将训练过程进行保存

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False
        )
    )

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tb_log_path, sess.graph)

    step = 0
    while True:
        batch_x, batch_y = get_next_batch(64)  # 64
        _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.95})
        print(step, 'loss:\t', loss_)

        step += 1

        # 每2000步保存一次实验结果
        if step % 2000 == 0:
            saver.save(sess, save_ckpt, global_step=step)

        # 在测试集上计算精度
        if step % 50 != 0:
            continue

        # 每50 step计算一次准确率，使用新生成的数据
        batch_x_test, batch_y_test = get_next_batch(256)  # 新生成的数据集个来做测试
        acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        print(step, 'acc---------------------------------\t', acc)

        # 终止条件
        if acc > 0.99:
            saver.save(sess, save_ckpt, global_step=step)
            break

        # 启用监控 tensor board
        summary = sess.run(merged, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        writer.add_summary(summary, step)


if __name__ == '__main__':
    train_crack_captcha_cnn()
    print('end...')
    pass
