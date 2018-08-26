import numpy as np
import tensorflow as tf
import
from sklearn.metrics import confusion_matrix
from tqdm import trange
import flags
import data_loader as data
from model import Siamese
from utils import plot_confusion_matrix
FLAGS = tf.app.flags.FLAGS


def train():
    tf.reset_default_graph()

    dataset = data.get_dataset(train=True)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    siamese = Siamese()
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    train_step = optimizer.minimize(siamese.loss)

    tf.summary.scalar('loss', siamese.loss)
    tf.summary.scalar('acc', siamese.accuracy)
    merged_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in trange(FLAGS.n_iters):
            x1, x2, y = sess.run(next_element)
            _, loss, summary = sess.run([train_step, siamese.loss, merged_summaries], feed_dict={
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y: y,
            })
            assert not np.isnan(loss), 'Model diverged with loss = NaN'
            train_writer.add_summary(summary, i)

            if i % 1000 == 0:
                saver.save(sess, FLAGS.model_path)
        print('Training completed, model saved:', saver.save(sess, FLAGS.model_path))


def test():
    def parse_file(f):
        image = ~cv2.imread(f, 0)
        image = image / 255
        return np.expand_dims(image, axis=-1)

    files = data.get_files(train=False)
    files = files[:FLAGS.n_test_classes]  # subsample for n-way classification
    images = [[parse_file(f) for f in l] for l in files]
    gt_images = [l[0] for l in images]

    siamese = Siamese()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model_path)

        gt_vals = sess.run(siamese.out, feed_dict={
            siamese.x1: gt_images,
        })

        preds = []
        for i in range(len(images)):
            test_images = images[i][1:]
            test_vals = sess.run(siamese.out, feed_dict={
                siamese.x1: test_images,
            })

            test_preds = []
            for val in test_vals:
                d = np.sum(np.abs(gt_vals - val), axis=1)
                test_preds.append(np.argmin(d))
            preds.append(test_preds)

    y_true = [[i]*len(l) for i, l in enumerate(images)]
    y_true = np.array(y_true).flatten()
    y_pred = np.array(preds).flatten()
    cm = confusion_matrix(y_true, y_pred)

    tp = np.eye(len(cm)) * cm
    print('Total accuracy:', np.sum(tp) / np.sum(cm))
    plot_confusion_matrix(cm, np.arange(len(images)))


if __name__ == '__main__':
    train()
    test()
