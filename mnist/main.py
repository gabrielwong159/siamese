import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
from tqdm import tqdm, trange
import flags
from models import Classifier, Siamese

FLAGS = tf.app.flags.FLAGS
mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)


def train_cls():
    tf.reset_default_graph()

    classifier = Classifier()
    optimizer = tf.train.AdamOptimizer(FLAGS.cls_lr)
    train_step = optimizer.minimize(classifier.loss)

    tf.summary.scalar('loss', classifier.loss)
    tf.summary.scalar('acc', classifier.accuracy)
    merged_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'classifier', 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'classifier', 'test'))
        sess.run(tf.global_variables_initializer())

        feed_shape = [-1, FLAGS.h, FLAGS.w, FLAGS.c]
        for i in trange(FLAGS.cls_iters, desc='Classifier training'):
            x, y = mnist.train.next_batch(FLAGS.cls_batch)
            x = x.reshape(feed_shape)
            _, summary = sess.run([train_step, merged_summaries], feed_dict={
                classifier.x: x,
                classifier.y: y,
                classifier.keep_prob: 0.5,
            })
            train_writer.add_summary(summary, i)

            if i % 500 == 0:
                summary = sess.run(merged_summaries, feed_dict={
                    classifier.x: mnist.test.images.reshape(feed_shape)[:5000],
                    classifier.y: mnist.test.labels[:5000],
                    classifier.keep_prob: 1.0,
                })
                test_writer.add_summary(summary, i)

        print('Training complete, model saved:', saver.save(sess, FLAGS.cls_model))


def test_cls():
    tf.reset_default_graph()

    classifier = Classifier()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.cls_model)

        n_iters = int(np.ceil(len(mnist.test.labels / FLAGS.cls_batch)))
        accuracies = []
        for i in trange(n_iters, desc='Classifier testing'):
            x = mnist.test.images[i::FLAGS.cls_batch]
            y = mnist.test.labels[i::FLAGS.cls_batch]
            accuracy = sess.run(classifier.accuracy, feed_dict={
                classifier.x: x.reshape([-1, FLAGS.h, FLAGS.w, FLAGS.c]),
                classifier.y: y,
                classifier.keep_prob: 1.0,
            })
            accuracies.append(accuracy)
    print('Final accuracy:', sum(accuracies) / len(accuracies))


def train_siamese(restore):
    tf.reset_default_graph()

    siamese = Siamese()
    optimizer = tf.train.AdamOptimizer(FLAGS.siamese_lr)
    train_step = optimizer.minimize(siamese.loss)

    tf.summary.scalar('loss', siamese.loss)
    merged_summaries = tf.summary.merge_all()

    restorer = tf.train.import_meta_graph(FLAGS.cls_model + '.meta', import_scope='siamese/conv')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'siamese', 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())
        if restore:
            restorer.restore(sess, FLAGS.cls_model)

        feed_shape = [-1, FLAGS.h, FLAGS.w, FLAGS.c]
        for i in trange(FLAGS.siamese_iters, desc='Siamese training'):
            x1, y1 = mnist.train.next_batch(FLAGS.siamese_batch)
            x2, y2 = mnist.train.next_batch(FLAGS.siamese_batch)

            x1 = np.reshape(x1, feed_shape)
            x2 = np.reshape(x2, feed_shape)
            y = (y1 == y2).astype(np.float32)

            _, loss, summary = sess.run([train_step, siamese.loss, merged_summaries], feed_dict={
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y: y,
                siamese.keep_prob: 0.5,
            })
            assert not np.isnan(loss), 'Model diverged with loss = NaN'
            train_writer.add_summary(summary, i)

        print('Training complete, model saved:', saver.save(sess, FLAGS.siamese_model))


def test_classification():
    tf.reset_default_graph()

    feed_shape = [-1, FLAGS.h, FLAGS.w, FLAGS.c]
    # use images from training set for comparison
    gt_images = []
    for i in range(FLAGS.n_classes):
        idx = np.argmax(mnist.train.labels == i)
        gt_images.append(mnist.train.images[idx])
    gt_images = np.array(gt_images).reshape(feed_shape)
    # prepare test set
    x, y = mnist.test.images, mnist.test.labels
    x = x.reshape(feed_shape)

    siamese = Siamese()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.siamese_model)

        gt_vals = sess.run(siamese.out, feed_dict={
            siamese.x1: gt_images,
            siamese.keep_prob: 1.0,
        })

        accuracies = []
        n_iters = int(np.ceil(len(x) / FLAGS.siamese_batch))
        for i in trange(n_iters, desc='Evaluating test images'):
            test_vals = sess.run(siamese.out, feed_dict={
                siamese.x1: x[i::n_iters],
                siamese.keep_prob: 1.0,
            })
            # evaluate batch by batch because the slicing rearranges the elements
            y_pred = []
            for val in test_vals:
                d = np.sum(np.abs(gt_vals - val), axis=1)
                y_pred.append(np.argmin(d))

            n_correct = np.sum(np.array(y_pred) == y[i::n_iters])
            accuracies.append(n_correct / len(y_pred))

    print('Final accuracy:', sum(accuracies) / len(accuracies))


if __name__ == '__main__':
    train_cls()
    test_cls()  # 0.9940597248524428
    train_siamese(restore=True)
    test_classification()  # 0.9923008041716308
