import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS


class Siamese(object):
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, FLAGS.h, FLAGS.w, FLAGS.c])
        self.x2 = tf.placeholder(tf.float32, [None, FLAGS.h, FLAGS.w, FLAGS.c])
        self.y = tf.placeholder(tf.float32, [None])

        with tf.variable_scope(FLAGS.scope) as scope:
            o1 = self.network(self.x1)
            scope.reuse_variables()
            o2 = self.network(self.x2)
        dist = tf.sqrt(tf.reduce_sum(tf.square(o1 - o2), axis=-1))
        self.loss = self.loss_with_spring(dist, self.y)
        self.accuracy = self.compute_accuracy(dist, self.y)
        self.inference = dist
        self.out = o1

    def network(self, inputs):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(1e-2)):
            net = slim.conv2d(inputs, 64, [10, 10], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 128, [7, 7], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [4, 4], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 256, [4, 4], scope='pool4')

        net = slim.flatten(net, scope='flat')

        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(1e-4)):
            net = slim.fully_connected(net, 1024, activation_fn=None, scope='fc5')
        return net

    def loss_with_spring(self, dist, labels):
        margin = 5.0
        dist += 1e-6
        pos = labels * dist
        neg = (1.0 - labels) * tf.square(tf.maximum(0.0, margin - dist))
        return tf.reduce_mean(pos + neg)

    def compute_accuracy(self, dist, labels):
        preds = tf.cast(dist < 0.5, tf.float32)
        correct_prediction = tf.cast(tf.equal(labels, preds), tf.float32)
        return tf.reduce_mean(correct_prediction)
