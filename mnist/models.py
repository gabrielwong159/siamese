import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = tf.app.flags.FLAGS


class Classifier(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, FLAGS.h, FLAGS.w, FLAGS.c])
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope(FLAGS.scope):
            logits = self.network(self.x, self.keep_prob)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        self.accuracy = self.compute_accuracy(labels=self.y, logits=logits)
        self.inference = tf.nn.softmax(logits)  # return probs of each class

    def network(self, inputs, keep_prob):
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5]), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.conv2d(inputs, 32, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

        net = slim.flatten(net, scope='flat')

        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, keep_prob, scope='dropout3')

        net = slim.fully_connected(net, 10, activation_fn=None, scope='fc4')
        return net

    def compute_accuracy(self, labels, logits):
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(probs, axis=-1)
        correct_prediction = tf.equal(preds, labels)
        accuracies = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracies)


class Siamese(object):
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, FLAGS.h, FLAGS.w, FLAGS.c])
        self.x2 = tf.placeholder(tf.float32, [None, FLAGS.h, FLAGS.w, FLAGS.c])
        self.y = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope(FLAGS.scope) as scope:
            o1 = self.network(self.x1, self.keep_prob)
            scope.reuse_variables()
            o2 = self.network(self.x2, self.keep_prob)
        self.loss = self.loss_with_spring(o1, o2, self.y)
        self.out = o1
        self.inference = tf.sqrt(tf.reduce_sum(tf.square(o1 - o2), axis=-1))

    def network(self, inputs, keep_prob):
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]), \
             slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
            net = slim.conv2d(inputs, 32, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

        net = slim.flatten(net, scope='flat')

        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, keep_prob, scope='dropout3')

        net = slim.fully_connected(net, 1024, activation_fn=None, scope='fc4')
        return net

    def loss_with_spring(self, o1, o2, labels):
        margin = 5.0
        d = tf.sqrt(tf.reduce_sum(tf.square(o1 - o2), axis=-1))
        d += 1e-6
        pos = labels * d
        neg = (1.0 - labels) * tf.square(tf.maximum(0.0, margin - d))
        return tf.reduce_mean(pos + neg)
