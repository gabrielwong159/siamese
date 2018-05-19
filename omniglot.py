import numpy as np
import cv2
import tensorflow as tf
from os import listdir
from pathlib import Path
from itertools import combinations
from random import choice, choices, shuffle
import model

model_path = 'model/omniglot/model.ckpt'
width = 105
height = 105

"""
{
    "Alphabet_of_the_Magi": {
        "character01": ['a.png', 'b.png', ...],
        "character02": [...],
        ...
    },
    "Anglo-Saxon_Futhorc": {
        "character01": [...],
        ...    
    },
    ...
}
"""


def get_files(dataset='train', array=False):
    if dataset == 'train':
        src = Path('data', 'images_background')
    elif dataset == 'test':
        src = Path('data', 'images_evaluation')
    else:
        raise ValueError('Invalid dataset parameter provided')

    out = {}
    for alphabet in listdir(src):
        chars = {}
        for character in listdir(src / alphabet):
            folder = src / alphabet / character
            chars[character] = [str(folder / f) for f in listdir(folder)]
        out[alphabet] = chars

    if not array:
        return out

    # [n_alphabets, n_characters, n_samples]
    l = []
    for alphabet, characters in out.items():
        _l = []
        for char, filenames in characters.items():
            _l.append(filenames)

        l.append(_l)

    return l


def train():
    learning_rate = 5e-4
    num_iterations = 200_000
    batch_size = 16

    l = get_files('train', array=True)
    pairs = []
    for a in l:  # l: list of alphabets
        for c in a:  # a: list of characters
            pairs.extend(combinations(c, 2))  # grab pairs from c: list of samples

    n = len(pairs)
    for _ in range(n):
        a1, a2 = choices(l, k=2)
        c1, c2 = choice(a1), choice(a2)
        s1, s2 = choice(c1), choice(c2)
        pairs.append((s1, s2))

    pairs = [pair + (i < n,) for i, pair in enumerate(pairs)]
    shuffle(pairs)
    x1, x2, y_ = map(tf.constant, zip(*pairs))

    def _parse(x1, x2, y_):
        def str_to_img(s):
            image_string = tf.read_file(s)
            image_decoded = tf.image.decode_png(image_string)
            image_inverted = tf.bitwise.invert(image_decoded)
            image_resized = tf.reshape(image_inverted, (height, width, 1))
            return tf.divide(image_resized, 255)
        im1, im2 = map(str_to_img, (x1, x2))
        return im1, im2, y_

    dataset = tf.data.Dataset.from_tensor_slices((x1, x2, y_)) \
                             .map(_parse) \
                             .repeat(-1) \
                             .batch(batch_size) \
                             .prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    siamese = model.Siamese(height, width)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(siamese.loss)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(num_iterations):
            x1, x2, y_ = sess.run(next_element)
            feed_dict = {
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y_: y_,
            }

            _, loss_v = sess.run([train_step, siamese.loss], feed_dict=feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'

            if i % 100 == 0:
                print(f'step {i}: loss {loss_v}')

            if i % 1000 == 0:
                print('Model saved:', saver.save(sess, model_path))

        print('Finished:', saver.save(sess, model_path))


def test():
    d = get_files('test')
    truth = []
    for a in d:  # alphabet
        for c in d[a]:  # character
            image = cv2.bitwise_not(cv2.imread(d[a][c][0], 0))
            image = image.flatten() / 255
            truth.append((a, c, image))
    a, c, i = zip(*truth)

    siamese = model.Siamese(size)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        labels = sess.run(siamese.o1, feed_dict={
            siamese.x1: i,
        })

        test = cv2.imread(d['Manipuri']['character03'][3], 0)
        test = cv2.bitwise_not(test)
        test = test.flatten() / 255
        out = sess.run(siamese.o1, feed_dict={
            siamese.x1: [test],
        })[0]

        L2 = np.sum(np.square(labels - out), axis=-1)
        idx = np.argmin(L2)
        print(a[idx], c[idx])

        cv2.imshow('in', test.reshape((105, 105)))
        cv2.imshow('out', i[idx].reshape((105, 105)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
    # test()
