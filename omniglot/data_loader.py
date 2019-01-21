import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import shuffle_and_repeat
import itertools
import random
import os
from os.path import join
FLAGS = tf.app.flags.FLAGS


def get_files(train, src='data'):
    if train:
        src = join(src, 'images_background')
    else:
        src = join(src, 'images_evaluation')

    alphabets = [join(src, f) for f in os.listdir(src)]
    characters = [join(a, f) for a in alphabets for f in os.listdir(a)]
    files = [[join(c, f) for f in os.listdir(c)] for c in characters]
    return files


def _parse_function(f1, f2, label):
    def file_to_img(f):
        image_string = tf.read_file(f)
        image_decoded = tf.image.decode_png(image_string)
        image_inverted = tf.bitwise.invert(image_decoded)
        image_resized = tf.reshape(image_inverted, (FLAGS.h, FLAGS.w, FLAGS.c))
        return image_resized / 255

    im1, im2 = map(file_to_img, [f1, f2])
    return im1, im2, label


def get_dataset(train, src='data'):
    files = get_files(train, src)
    pairs = []
    for l in files:  # for each folder of a character
        pairs.extend(itertools.combinations(l, 2))  # take all possible pairs
    # sub-sample the pairs
    pairs = random.sample(pairs, k=FLAGS.n_samples)
    pairs = [pair + (1.0,) for pair in pairs]  # append label=1.0 to all pairs

    for _ in range(len(pairs)):  # sample an equal number of dissimilar pairs
        c1, c2 = random.choices(files, k=2)  # choose two random characters
        f1, f2 = random.choice(c1), random.choice(c2)  # sample from each
        pairs.append((f1, f2, 0.0))

    f1, f2, labels = map(np.array, zip(*pairs))
    shuffle_idxs = np.arange(len(labels))
    np.random.shuffle(shuffle_idxs)
    f1 = f1[shuffle_idxs]
    f2 = f2[shuffle_idxs]
    labels = labels[shuffle_idxs]

    dataset = tf.data.Dataset.from_tensor_slices((f1, f2, labels)) \
                             .map(_parse_function, num_parallel_calls=8) \
                             .apply(shuffle_and_repeat(buffer_size=10_000)) \
                             .batch(FLAGS.batch_size) \
                             .prefetch(FLAGS.batch_size)
    return dataset
