import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


class Classifier(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self.network(self.x, self.keep_prob)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        
        self.loss = tf.reduce_mean(cross_entropy)
        self.accuracy = self.compute_accuracy(labels=self.y, logits=logits)
        self.inference = tf.nn.softmax(logits)  # return probs of each class
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()

    def network(self, inputs, keep_prob):
        with slim.arg_scope([slim.conv2d], kernel_size=[5, 5]):
            net = slim.conv2d(inputs, 32, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

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
