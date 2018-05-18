import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
import model

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
model_path = 'model/mnist/model.ckpt'


def train():
    learning_rate = 5e-4
    num_iterations = 500_000

    siamese = model.Siamese()
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 1e-2
    optimizer = tf.train.AdamOptimizer(learning_rate)  # 5e-4
    train_step = optimizer.minimize(siamese.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, model_path)

        for i in range(num_iterations):
            x1, y1 = mnist.train.next_batch(128)
            x2, y2 = mnist.train.next_batch(128)
            y = (y1 == y2).astype(np.float32)
            feed_dict = {
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y_: y,
            }

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict=feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'

            if i % 100 == 0:
                print(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                print('Model saved:', saver.save(sess, model_path))

        print('Finished:', saver.save(sess, model_path))


def test():
    test_data = 'data/mnist'
    files = [join(test_data, f'{i}.png') for i in range(10)]
    truth = [cv2.imread(f, 0).flatten() / 255 for f in files]

    siamese = model.Siamese()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        # get ground truth results
        labels = sess.run(siamese.o1, feed_dict={
            siamese.x1: truth,
        })

        # get test image results
        x, y = mnist.test.next_batch(10000)
        outputs = sess.run(siamese.o1, feed_dict={
            siamese.x1: x,
        })

        # get euclidean distance between both
        def pred(x):
            # L1 = np.sum(np.abs(labels - x), axis=-1)
            L2 = np.sum(np.square(labels - x), axis=-1)
            return np.argmin(L2)

        preds = np.apply_along_axis(pred, axis=-1, arr=outputs)
        print(np.sum(preds == y))


if __name__ == '__main__':
    # train()
    test()
