from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from datasets import dataset_factory
from datasets import imagenet
from nets import mobilenet_v1
from preprocessing import preprocessing_factory
from preprocessing import inception_preprocessing
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib



slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', -1,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():


    """read the image, one single image"""
    """image = tf.image.decode_jpeg(image, channels=3)
    """
    """
    image = cv2.imread('train2.JPEG')
    image = tf.cast(image, tf.float32)
    image_size = mobilenet_v1.mobilenet_v1.default_image_size
    image = image_preprocessing_fn(image, image_size, image_size)
    image = tf.expand_dims(image, 0)
    """
    image_size = mobilenet_v1.mobilenet_v1.default_image_size
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    ####################
    # Define the model #
    ####################
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training = False)):
      logits, _, saved_out = mobilenet_v1.mobilenet_v1(processed_images, num_classes = 1001, is_training=False)
    #logits, _, saved_out = network_fn(image)
    probabilities = tf.nn.softmax(logits)

    """need to check how to load checkpoint"""
    init_fn = slim.assign_from_checkpoint_fn(
      os.path.join(FLAGS.checkpoint_path),
      slim.get_model_variables('MobilenetV1')
    )

    with tf.Session() as sess:
      init_fn(sess)
      np_image, np_logits, np_saved_out, np_probs= sess.run([image, logits, saved_out, probabilities])
      sorted_inds = [i[0] for i in sorted(enumerate(-np_logits[0][0:]), key=lambda x:x[1])]
    print("Preprocessed Image")
    print(np.shape(np_image))
    print("First depth-seperable layer output")
    print("Logits")
    print(np.shape(np_logits[0][0:]))
    print(np.argmax(np_logits))

    print("sorted inds")
    print(sorted_inds[0:5])
    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (np_probs[0][index] * 100, names[index]))


if __name__ == '__main__':
  tf.app.run()

