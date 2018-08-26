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


def train():
    n_samples = 20_000
    learning_rate = 1e-5
    num_iterations = 500_000
    batch_size = 32

    filenames = np.load('data/images_background_filenames.npy')

    print(filenames.shape)

    pairs = []
    for class_filenames in filenames:
        pairs.extend(itertools.combinations(class_filenames, 2))

    pairs = sample(pairs, k=n_samples)
    pairs = [pair + (1.0,) for pair in pairs]
    for i in range(n_samples):
        class_1, class_2 = choices(filenames, k=2)
        im_1, im_2 = choice(class_1), choice(class_2)
        pairs.append((im_1, im_2, 0.0))

    def _parse(x1, x2, y_):
        def str_to_img(s):
            image_string = tf.read_file(s)
            image_decoded = tf.image.decode_png(image_string)
            image_resized = tf.reshape(image_decoded, (h, w, 1))
            return tf.divide(image_resized, 255)

        im1, im2 = map(str_to_img, [x1, x2])
        return im1, im2, y_

    x1, x2, y_ = map(np.array, zip(*pairs))
    dataset = tf.data.Dataset.from_tensor_slices((x1, x2, y_)) \
                             .map(_parse, num_parallel_calls=8) \
                             .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(pairs))) \
                             .batch(batch_size) \
                             .prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    siamese = Siamese()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(siamese.loss)
        sess.run(tf.variables_initializer(optimizer.variables()))

        saver = tf.train.Saver()

        min_loss = (-1, float('inf'))
        for i in trange(num_iterations):
            x1, x2, y_ = sess.run(next_element)
            feed_dict = {
                siamese.x1: x1,
                siamese.x2: x2,
                siamese.y_: y_,
                siamese.keep_prob: 0.5,
            }
            _, loss_v = sess.run([train_step, siamese.loss], feed_dict=feed_dict)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'
            if loss_v < min_loss[1]:
                min_loss = (i, loss_v)

            if i % 100 == 0:
                tqdm.write(f'step {i}: loss {loss_v} Minimum loss: {min_loss}')

            if (i+1) % 1000 == 0:
                tqdm.write('Model saved: {}'.format(saver.save(sess, model_path)))

        print('Finished:', saver.save(sess, model_path))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


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

