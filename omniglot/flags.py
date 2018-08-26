import tensorflow as tf

# Data
tf.app.flags.DEFINE_integer('h', 105, 'Image height')
tf.app.flags.DEFINE_integer('w', 105, 'Image width')
tf.app.flags.DEFINE_integer('c', 1, 'Image channels')

# Model definition
tf.app.flags.DEFINE_string('scope', 'siamese', 'Variable scope')

# Training hyperparameters
tf.app.flags.DEFINE_integer('n_samples', 20_000, 'Number of similar/dissimilar pairs for training')

tf.app.flags.DEFINE_float('lr', 1e-5, 'Siamese learning rate')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Siamese batch size')
tf.app.flags.DEFINE_integer('n_iters', 500_000, 'Siamese training iterations')

# Directories
tf.app.flags.DEFINE_string('model_path', 'model/siamese/model', 'Path to saved Siamese model')
tf.app.flags.DEFINE_string('summaries_dir', 'summaries', 'Path to save TF summaries')