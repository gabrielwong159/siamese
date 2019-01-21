import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
from tqdm import trange
from models.classifier import Classifier

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=False, reshape=False)

model_path = 'model/classifier/model'
log_dir = 'logs/classifier'

learning_rate = 1e-4
batch_size = 50
n_iters = 20_000


def train():
    model = Classifier()
    saver = tf.train.Saver()
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(model.loss)
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(join(log_dir, 'test'), sess.graph)
        sess.run(tf.global_variables_initializer())
        
        for i in trange(n_iters):
            x, y, = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_step, model.summary], feed_dict={
                model.x: x,
                model.y: y,
                model.keep_prob: 0.5,
            })
            train_writer.add_summary(summary, i)
            
            if i % 500 == 0:
                summary = sess.run(model.summary, feed_dict={
                    model.x: mnist.test.images[:6000],
                    model.y: mnist.test.labels[:6000],
                    model.keep_prob: 1.0,
                })
                test_writer.add_summary(summary, i)
        print('Training complete, model saved at:', saver.save(sess, model_path))
        
        
if __name__ == '__main__':
    train()
