import tensorflow as tf
import tensorflow.contrib.slim as slim

h, w, c = 105, 105, 1


class Siamese(object):
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, h, w, c])
        self.x2 = tf.placeholder(tf.float32, [None, h, w, c])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
        self.loss = self.loss_with_spring()

    def network(self, x):
        with slim.arg_scope([slim.conv2d],
                            padding='VALID',
                            weights_regularizer=slim.l2_regularizer(2e-4)):
            net = slim.conv2d(x, 64, [10, 10], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 128, [7, 7], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [4, 4], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 256, [4, 4], scope='conv4')

        net = slim.flatten(net, scope='flat')
        net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc1')
        return net

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
