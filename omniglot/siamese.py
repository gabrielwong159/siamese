import tensorflow as tf
import tensorflow.contrib.slim as slim

h = 105
w = 105
c = 1


class Siamese:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, h, w, c])
        self.x2 = tf.placeholder(tf.float32, [None, h, w, c])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self.network(self.x1, self.x2)
        labels = tf.expand_dims(self.y_, axis=-1)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        regularization_losses = tf.add_n(tf.losses.get_regularization_losses())
        self.loss = tf.reduce_mean(cross_entropy) + regularization_losses
        self.out = tf.nn.sigmoid(logits)

    def network(self, x1, x2):
        with tf.variable_scope('siamese') as scope:
            o1 = self.convnet(x1)
            scope.reuse_variables()
            o2 = self.convnet(x2)

        dist = tf.abs(o1 - o2)
        logits = slim.fully_connected(dist, 1, activation_fn=None)
        return logits

    def convnet(self, x):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(2e-4)):
            net = slim.conv2d(x, 64, [10, 10], padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 128, [7, 7], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 128, [4, 4], padding='VALID', scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 256, [4, 4], padding='VALID', scope='conv4')

        net = slim.flatten(net, scope='flat')
        net = slim.fully_connected(net, 4096, activation_fn=tf.nn.sigmoid, scope='fc1')
        return net
