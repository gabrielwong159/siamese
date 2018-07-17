import tensorflow as tf
import tensorflow.contrib.slim as slim

h = 28
w = 28
c = 1


class Classifier(object):
    def __init__(self):
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
        net = slim.conv2d(self.x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.flatten(net, scope='flat')

        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, self.keep_prob, scope='drop1')

        net = slim.fully_connected(net, 10, activation_fn=None, scope='out')
        return net


def train():
    mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)

    learning_rate = 1e-4
    num_iterations = 20_000
    batch_size = 50

    model = Classifier()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with trange(num_iterations) as pbar:
            for i in pbar:
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([-1, h, w, c])

                if i % 100 == 0:
                    loss, acc = sess.run([model.loss, model.accuracy], feed_dict={
                        model.x: x,
                        model.y: y,
                        model.keep_prob: 1.0,
                    })
                    pbar.set_postfix(loss=loss, acc=acc)

                train_step.run(feed_dict={
                    model.x: x,
                    model.y: y,
                    model.keep_prob: 0.5,
                })

        print(saver.save(sess, 'model/classifier/model'))


def eval():
    mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)

    model = Classifier()
    saver = tf.train.Saver()

    images = mnist.test.images.reshape([-1, h, w, c])
    labels = mnist.test.labels

    with tf.Session() as sess:
        saver.restore(sess, 'model/classifier/model')

        accuracies = []
        n_iters = 10
        for i in trange(n_iters):
            x = images[i::n_iters]
            y = labels[i::n_iters]

            test_accuracy = model.accuracy.eval(feed_dict={
                model.x: x,
                model.y: y,
                model.keep_prob: 1.0
            })
            accuracies.append(test_accuracy)

    print(sum(accuracies) / len(accuracies))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    from tqdm import trange
    # train()
    eval()
