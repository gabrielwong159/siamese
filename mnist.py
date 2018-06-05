import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
import model

mnist = input_data.read_data_sets('data/mnist/MNIST_data/', one_hot=False)
model_path = 'model/mnist/model'


def train():
    learning_rate = 1e-4
    num_iterations = 20_000

    siamese = model.Siamese(height=28, width=28, model='mnist')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(siamese.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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
            }

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict=feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'

            if i % 100 == 0:
                print(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                print('Model saved:', saver.save(sess, model_path))

        print('Finished:', saver.save(sess, model_path))


def test():
    test_data = 'data/mnist/labels'
    files = [join(test_data, f'{i}.png') for i in range(10)]
    truth = [cv2.imread(f, 0) / 255 for f in files]
    truth = np.array(truth).reshape([-1, 28, 28, 1])

    siamese = model.Siamese(height=28, width=28, model='mnist')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        # get ground truth results
        labels = sess.run(siamese.o1, feed_dict={
            siamese.x1: truth,
        })

        x, y = mnist.test.images, mnist.test.labels
        x = np.reshape(x, [-1, 28, 28, 1])
        # divide a test batch of 10000 into 10*1000
        n = 0
        for i in range(10):
            batch = x[i*1000:(i+1)*1000]
            outputs = sess.run(siamese.o1, feed_dict={
                siamese.x1: batch,
            })

            def pred(x):
                L1 = np.sum(np.abs(labels - x), axis=-1)
                L2 = np.sum(np.square(labels - x), axis=-1)
                return np.argmin(L1)

            preds = np.apply_along_axis(pred, axis=-1, arr=outputs)
            n += np.sum(preds == y[i*1000:(i+1)*1000])
        print(n)


if __name__ == '__main__':
    with tf.Graph().as_default():
        train()
    with tf.Graph().as_default():
        test()
