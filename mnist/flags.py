import tensorflow as tf

# Data
tf.app.flags.DEFINE_integer('h', 28, 'Image height')
tf.app.flags.DEFINE_integer('w', 28, 'Image width')
tf.app.flags.DEFINE_integer('c', 1, 'Image channels')
tf.app.flags.DEFINE_integer('n_classes', 10, 'Number of classes')

# Model definition
tf.app.flags.DEFINE_string('scope', 'siamese', 'Variable scope')

# Training hyperparameters
tf.app.flags.DEFINE_float('cls_lr', 1e-4, 'Classifier learning rate')
tf.app.flags.DEFINE_integer('cls_batch', 128, 'Classifier batch size')
tf.app.flags.DEFINE_integer('cls_iters', 10_000, 'Classifier training iterations')

tf.app.flags.DEFINE_float('siamese_lr', 1e-4, 'Siamese learning rate')
tf.app.flags.DEFINE_integer('siamese_batch', 128, 'Siamese batch size')
tf.app.flags.DEFINE_integer('siamese_iters', 20_000, 'Siamese training iterations')

# Directories
tf.app.flags.DEFINE_string('cls_model', 'model/cls/model', 'Path to saved Classifier')
tf.app.flags.DEFINE_string('siamese_model', 'model/siamese/model', 'Path to saved Siamese model')
tf.app.flags.DEFINE_string('summaries_dir', 'summaries', 'Path to save TF summaries')
