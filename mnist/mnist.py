import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
from tqdm import tqdm, trange
from siamese import Siamese

mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)
model_path = 'model/siamese/model'
h, w, c = 28, 28, 1


def train():
    learning_rate = 1e-4
    num_iterations = 20_000

    siamese = Siamese()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(siamese.loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in trange(num_iterations):
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
                tqdm.write(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                tqdm.write('Model saved:', saver.save(sess, model_path))

        print('Finished:', saver.save(sess, model_path))


def test():
    test_data = 'data/labels'
    files = [join(test_data, f'{i}.png') for i in range(10)]
    ground = [cv2.imread(f, 0) / 255 for f in files]
    ground = np.array(ground).reshape([-1, 28, 28, 1])

    siamese = Siamese()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        ground_scores = sess.run(siamese.o1, feed_dict={
            siamese.x1: ground,
            siamese.keep_prob: 1.0,
        })

        x, y = mnist.test.images, mnist.test.labels
        x = np.reshape(x, [-1, h, w, c])

        n_correct = 0
        batch_size = 1000
        n_batches = len(x) // batch_size
        for i in trange(n_batches):
            batch_images, batch_labels = x[i::n_batches], y[i::n_batches]
            batch_scores = sess.run(siamese.o1, feed_dict={
                siamese.x1: batch_images,
                siamese.keep_prob: 1.0,
            })
            for score, label in zip(batch_scores, batch_labels):
                dist = np.sum(np.abs(ground_scores - score), axis=-1)
                pred = np.argmin(dist)
                n_correct += (pred == label)

        print(n_correct / len(x))


if __name__ == '__main__':
    #with tf.Graph().as_default():
    #   train()
    with tf.Graph().as_default():
       test()
