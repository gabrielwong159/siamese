import cv2
import numpy as np
import tensorflow as tf
import itertools
import random
import os
from pathlib import Path

import data
import model

model_path = './model/model.ckpt'
num_iterations = 20000
batch_size = 800


def parse(f):
    return cv2.imread(f, 0).flatten() / 255


def test():
    sess = tf.Session()
    siamese = model.siamese(size=105**2)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print(sim_dist(sess, siamese), diff_dist(sess, siamese))


def diff_dist(sess, siamese):
    src1 = Path('data', 'images_evaluation', 'Angelic', 'character01')
    src2 = Path('data', 'images_evaluation', 'Angelic', 'character02')
    files1 = [str(src1 / f) for f in os.listdir(src1)]
    files2 = [str(src2 / f) for f in os.listdir(src2)]
    images1 = [parse(f) for f in files1]
    images2 = [parse(f) for f in files2]

    pairs = itertools.product(images1, images2)
    x1, x2 = zip(*pairs)

    o1 = sess.run(siamese.o1, feed_dict={siamese.x1: x1})
    o2 = sess.run(siamese.o1, feed_dict={siamese.x1: x2})

    sq_diff = np.square(o1 - o2)
    l2_norm = np.sum(sq_diff, axis=1)
    return np.min(l2_norm), np.max(l2_norm)


def sim_dist(sess, siamese):
    src = Path('data', 'images_evaluation', 'Angelic', 'character01')
    files = [str(src / f) for f in os.listdir(src)]
    images = [parse(f) for f in files]

    pairs = itertools.combinations(images, 2)
    x1, x2 = zip(*pairs)

    o1 = sess.run(siamese.o1, feed_dict={siamese.x1: x1})
    o2 = sess.run(siamese.o1, feed_dict={siamese.x1: x2})

    sq_diff = np.square(o1 - o2)
    l2_norm = np.sum(sq_diff, axis=1)
    return np.min(l2_norm), np.max(l2_norm)


def train():
    files = data.get_files()

    combine = []
    for alphabet in files:
        for character in files[alphabet]:
            combine.append([parse(f) for f in files[alphabet][character]])
    combine = combine[:500]

    same = []
    for charset in combine:
        same.extend(list(itertools.combinations(charset, 2)))

    diff = []
    for i, charset1 in enumerate(combine):
        for j, charset2 in enumerate(combine):
            if i == j: continue
            diff.extend(list(itertools.product(charset1, charset2)))
        if i % 100 == 0: print(i)

    sess = tf.Session()

    siamese = model.siamese(size=np.product(combine[0][0].shape))
    train_step = tf.train.AdamOptimizer(6e-5).minimize(siamese.loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(num_iterations):
        pairs, y_ = [], []
        for _ in range(batch_size):
            if random.choice(range(2)):
                pairs.append(random.choice(same))
                y_.append(True)
            else:
                pairs.append(random.choice(diff))
                y_.append(False)

        x1, x2 = zip(*pairs)

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
            siamese.x1: x1,
            siamese.x2: x2,
            siamese.y_: y_
        })

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if i % 100 == 0:
            print(i, loss_v, sim_dist(sess, siamese), diff_dist(sess, siamese))

    print(saver.save(sess, model_path))


if __name__ == '__main__':
    train()
    # test()
