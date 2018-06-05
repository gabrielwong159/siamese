import numpy as np
import cv2
import tensorflow as tf
from os import listdir
from pathlib import Path
from itertools import combinations
from random import choice, choices, shuffle
from tqdm import tqdm, trange
import model

model_path = 'model/omniglot/model'
width = 105
height = 105


def get_files(dataset='train', array=False):
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

    if dataset == 'train':
        src = Path('data', 'omniglot', 'images_background')
    elif dataset == 'test':
        src = Path('data', 'omniglot', 'images_evaluation')
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


def train(resume=True):
    learning_rate = 5e-4
    num_iterations = 20_000
    batch_size = 32

    # grab all possible combinations of similar pairs
    l = get_files('train', array=True)
    pairs = []
    for a in l:  # l: list of alphabets
        for c in a:  # a: list of characters
            pairs.extend(combinations(c, 2))  # grab pairs from c: list of samples
    pairs = [pair + (1.0,) for pair in pairs]  # tag as similar

    # reduce the number of training points
    n_samples = 50_000
    pairs = choices(pairs, k=n_samples)

    n = len(pairs)  # use an equal number of same and diff pairs
    for _ in range(n):
        a1, a2 = choices(l, k=2)
        c1, c2 = choice(a1), choice(a2)
        s1, s2 = choice(c1), choice(c2)
        pairs.append((s1, s2, 0.0))

    def _parse(x1, x2, y_):
        def str_to_img(s):
            image_string = tf.read_file(s)
            image_decoded = tf.image.decode_png(image_string)
            image_inverted = tf.bitwise.invert(image_decoded)
            image_resized = tf.reshape(image_inverted, (height, width, 1))
            return tf.divide(image_resized, 255)

        im1, im2 = map(str_to_img, [x1, x2])
        return im1, im2, y_

    n_epochs = 400
    x1, x2, y_ = map(tf.constant, zip(*pairs))
    dataset = tf.data.Dataset.from_tensor_slices((x1, x2, y_)) \
                             .map(_parse) \
                             .shuffle(buffer_size=10_000) \
                             .repeat(n_epochs) \
                             .batch(batch_size) \
                             .prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    siamese = model.Siamese(height, width, model='omniglot')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(siamese.loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if resume:
            saver.restore(sess, model_path)
            print('Restored model from:', model_path)

        i = 0
        try:
            while True:
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

                if i+1 % 1000 == 0:
                    print('Model saved:', saver.save(sess, model_path))

                i += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            print('Finished:', saver.save(sess, model_path))


def test():
    d = get_files('test')
    truth = []
    for a in d:  # alphabet
        for c in d[a]:  # character
            image = cv2.bitwise_not(cv2.imread(d[a][c][0], 0))
            image = image.reshape([height, width, 1]) / 255
            truth.append((a, c, image))
    a, c, i = zip(*truth)

    siamese = model.Siamese(height, width, model='omniglot')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        ground_values = np.array([sess.run(siamese.o1, {siamese.x1: [_]}) for _ in tqdm(i, desc='Computing ground truth labels')])

        count = 0
        pbar = trange(10_000, postfix={'count': 0}, desc='Running test images')
        for i in pbar:
            _a = choice(list(d.keys()))
            _c = choice(list(d[_a].keys()))
            _i = choice(d[_a][_c])

            test = cv2.imread(_i, 0)
            test = cv2.bitwise_not(test)
            test = test.reshape([height, width, 1]) / 255
            out = sess.run(siamese.o1, feed_dict={
                siamese.x1: [test],
            })[0]

            L2 = np.sum(np.square(ground_values - out), axis=-1)
            idx = np.argmin(L2)
            if a[idx] == _a and c[idx] == _c:
                count += 1
                pbar.set_postfix({'count': count})


if __name__ == '__main__':
    with tf.Graph().as_default():
        train(False)
    with tf.Graph().as_default():
        test()
