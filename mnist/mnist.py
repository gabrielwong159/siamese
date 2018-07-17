import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
from tqdm import trange
from siamese import Siamese

mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)
model_path = 'model/siamese/model'


def train():
    learning_rate = 1e-4
    num_iterations = 10_000

    siamese = Siamese()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        variables_to_restore = slim.get_variables_to_restore(include=['siamese/conv'])
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, 'model/classifier/model')

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(siamese.loss)
        sess.run(tf.variables_initializer(optimizer.variables()))

        saver = tf.train.Saver()

        for i in range(num_iterations):
            x1, y1 = mnist.train.next_batch(128)
            x2, y2 = mnist.train.next_batch(128)

            x1 = np.reshape(x1, [-1, 28, 28, 1])
            x2 = np.reshape(x2, [-1, 28, 28, 1])

            y = (y1 == y2).astype(np.float32)
            feed_dict = {
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y_: y,
                siamese.keep_prob: 0.5,
            }

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict=feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'

            if i % 100 == 0:
                print(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                print('Model saved:', saver.save(sess, model_path))

        print('Finished:', saver.save(sess, model_path))


def test():
    test_data = 'data/labels'
    files = [join(test_data, f'{i}.png') for i in range(10)]
    truth = [cv2.imread(f, 0) / 255 for f in files]
    truth = np.array(truth).reshape([-1, 28, 28, 1])

    siamese = Siamese()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        x, y = mnist.test.images, mnist.test.labels
        x = np.reshape(x, [-1, 28, 28, 1])
        n = 0
        with trange(len(x)) as pbar:
            for i in pbar:
                image, label = x[i], y[i]
                feed_dict = {
                    siamese.x1: np.array([image for _ in truth]),
                    siamese.x2: truth,
                    siamese.keep_prob: 1.0,
                }
                sigmoid = sess.run([siamese.out], feed_dict=feed_dict)
                pred = np.argmax(sigmoid)
                n += (pred == label)

                if (i+1) % 100 == 0:
                    pbar.set_postfix(acc=n/i)
        print(n / i)


if __name__ == '__main__':
    # with tf.Graph().as_default():
    #    train()
    with tf.Graph().as_default():
        test()
