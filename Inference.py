# -*- coding: UTF-8 -*-
# Define graph automatically on cpu or on gpu.

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
from inception_utils import inception_arg_scope as arg_scope
from inception_v3 import inception_v3 as graph


class Inference():
    visual_loaded = False

    def _define_graph(self, num_classes=1001):
        with tf.Graph().as_default():
            with slim.arg_scope(arg_scope()):
                # A tensor of size [batch_size, height, width, channels], default
                # is 299 * 299.
                self.images = tf.placeholder(tf.float32, [1, 299, 299, 3])
                logits, _ = graph(
                    self.images,
                    num_classes=num_classes,
                    is_training=False,
                    spatial_squeeze=True,
                    scope=None)
                # Probably on class and prediction index.
                self.prob_on_class = tf.nn.softmax(logits)
                self.pred_op = tf.argmax(logits, axis=1)
                self.prob_op = tf.reduce_max(self.prob_on_class, axis=1)
                # Define topk predictor.
                self.values_tensor, self.indices_tensor = tf.nn.top_k(
                    self.prob_on_class, self.topk)

                self.variables_to_restore = slim.trainable_variables()
                # Generate sessions and get features op
                tf_config = tf.ConfigProto()
                if self.growth:
                    tf_config.gpu_options.allow_growth = True
                else:
                    tf_config.gpu_options.per_process_gpu_memory_fraction = self.memory_fraction

                # Definition of session.
                self.sess = tf.Session(config=tf_config)

    def load_model(self, model_path):
        # Restore model.
        self._define_graph()
        self.saver = tf.train.Saver(self.variables_to_restore)
        self.saver.restore(self.sess, model_path)
        self.visual_loaded = True

    def get_prediction(self, images):
        if self.visual_loaded:
            pred, prob = self.sess.run(
                [self.pred_op, self.prob_op], {self.images: images})
            pred = np.squeeze(pred)
            prob = np.squeeze(prob)
        else:
            print('Firstly please load pre-trained model.')
