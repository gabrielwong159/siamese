import numpy as np
import tensorflow as tf
import itertools
import random
from random import choice, choices, sample, shuffle
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from siamese import Siamese

random.seed(0)
tf.set_random_seed(0)

model_path = 'model/omniglot/model'
h, w, c = 105, 105, 1


def test(dist_type):
    if dist_type == 'L1': dist_fn = np.abs
    elif dist_type == 'L2': dist_fn = np.square
    else: raise ValueError("dist_type should be 'L1' or 'L2', given {}".format(dist_type))

    images = np.load('data/images_evaluation.npy')[:20] / 255
    images = np.expand_dims(images, axis=-1)
    ground = images[:, 0]

    siamese = Siamese()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        ground_scores = sess.run(siamese.o1, feed_dict={
            siamese.x1: ground,
            siamese.keep_prob: 1.0,
        })

        preds = [[] for _ in images]
        for i in trange(len(images), desc=dist_type):
            batch_scores = sess.run(siamese.o1, feed_dict={
                siamese.x1: images[i],
                siamese.keep_prob: 1.0,
            })

            for score in batch_scores:
                dist = np.sum(dist_fn(ground_scores - score), axis=-1)
                pred = np.argmin(dist)
                preds[i].append(pred)

    y_true = np.array([[i for _ in images[i]] for i in range(len(images))]).flatten()
    y_preds = np.array(preds).flatten()
    cm = confusion_matrix(y_true, y_preds)
    tp = np.eye(len(cm)) * cm
    print(dist_type, np.sum(tp) / np.sum(cm))
    plot_confusion_matrix(cm, np.arange(len(images)))


if __name__ == '__main__':
    with tf.Graph().as_default():
        train()
    with tf.Graph().as_default():
        test('L1')
    with tf.Graph().as_default():
        test('L2')

