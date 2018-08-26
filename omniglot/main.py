import numpy as np
import tensorflow as tf
from tqdm import trange
import flags
import data_loader as data
from model import Siamese
FLAGS = tf.app.flags.FLAGS


def train():
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
    pass


if __name__ == '__main__':
    train()
