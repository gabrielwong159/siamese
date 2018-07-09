import tensorflow as tf
import tensorflow.contrib.slim as slim


class Siamese:
    def __init__(self, height, width, model):
        if model.lower() == 'mnist':
            network = self.mnist
        elif model.lower() == 'omniglot':
            network = self.omniglot
        else:
            raise ValueError('Unknown model name passed to Siamese object')

        self.x1 = tf.placeholder(tf.float32, [None, height, width, 1])
        self.x2 = tf.placeholder(tf.float32, [None, height, width, 1])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.o1 = network(self.x1)
            scope.reuse_variables()
            self.o2 = network(self.x2)
        dist = tf.abs(self.o1 - self.o2)

        logits = slim.fully_connected(dist, 1, activation_fn=None)
        labels = tf.expand_dims(self.y_, axis=-1)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy) + tf.add_n(slim.losses.get_regularization_losses())

        self.out = tf.nn.sigmoid(logits)

        # regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        # self.loss = self.loss_with_spring() + regularization_loss

    def mnist(self, x):
        net = slim.conv2d(x, 32, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.flatten(net, scope='flat')

        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, keep_prob=0.5, scope='drop1')

        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid, scope='fc2')
        return net

    def omniglot(self, x):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.1)):
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

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
