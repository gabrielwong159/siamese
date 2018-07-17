import tensorflow as tf
import tensorflow.contrib.slim as slim

h, w, c = 105, 105, 1


class Classifier(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.x = tf.placeholder(tf.float32, [None, h, w, c])
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese'):
            logits = self.network()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)

        probs = tf.nn.softmax(logits)
        preds = tf.argmax(probs, axis=-1)
        correct_prediction = tf.equal(preds, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.out = preds

    def network(self):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            net = slim.conv2d(self.x, 64, [10, 10], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 128, [7, 7], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [4, 4], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 256, [4, 4], scope='conv4')

            net = slim.flatten(net, scope='flat')

            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, self.keep_prob, scope='drop1')

            net = slim.fully_connected(net, self.n_classes, activation_fn=None, scope='out')
            return net


import numpy as np
from tqdm import trange


def train():
    learning_rate = 1e-4
    num_iterations = 20_000
    batch_size = 32

    images = np.load('data/images_background.npy')
    labels = np.array([[i for _ in range(20)] for i in range(len(images))])
    print(images.shape, labels.shape)
    n_classes = len(images)

    images = images.reshape([-1, h, w, c])
    labels = labels.reshape([-1])
    print(images.shape, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=10_000) \
                     .repeat(-1) \
                     .batch(batch_size) \
                     .prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    model = Classifier(n_classes)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    saver = tf.train.Saver()

    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with trange(num_iterations) as pbar:
            for i in pbar:
                x, y = sess.run(next_element)

                if i % 100 == 0:
                    loss, acc = sess.run([model.loss,model.accuracy], feed_dict={
                        model.x: x / 255,
                        model.y: y,
                        model.keep_prob: 1.0,
                    })
                    pbar.set_postfix(loss=loss, acc=acc)
                    losses.append(loss)

                if i % 500 == 0:
                    saver.save(sess, 'model/classifier/model')

                train_step.run(feed_dict={
                    model.x: x / 255,
                    model.y: y,
                    model.keep_prob: 0.5,
                })

        print(saver.save(sess, 'model/classifier/model'))
        np.save('losses', np.array(losses))


if __name__ == '__main__':
    train()
