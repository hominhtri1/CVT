import numpy as np
import tensorflow as tf

from model import task_module, model_helpers


class ClassificationModule(task_module.SemiSupervisedModule):
  def __init__(self, config, task_name, n_classes, inputs,
               encoder):
    super(ClassificationModule, self).__init__()
    self.task_name = task_name
    self.n_classes = n_classes
    self.labels = labels = tf.placeholder(tf.float32, [None, None],
                                          name=task_name + '_labels')

    class PredictionModule(object):
      def __init__(self, name, input_reprs, roll_direction=0, activate=True):
        self.name = name
        with tf.variable_scope(name + '/predictions'):
          self.pool = tf.layers.max_pooling1d(input_reprs, config.max_sentence_length, 1,
                                              padding='valid', name='pool')
          self.logits = tf.layers.dense(self.pool, n_classes, name='predict')

        targets = labels
        targets *= (1 - inputs.label_smoothing)
        targets += inputs.label_smoothing / n_classes
        self.loss = model_helpers.masked_ce_loss(
          self.logits, targets, inputs.mask, roll_direction=roll_direction)

    primary = PredictionModule('primary', encoder.bi_reprs)

    self.unsupervised_loss = 0
    self.supervised_loss = primary.loss
    self.probs = tf.nn.softmax(primary.logits)
    self.preds = tf.argmax(primary.logits, axis=-1)

  def update_feed_dict(self, feed, mb):
    labels = [e.labels for e in mb.examples]
    feed[self.labels] = np.eye(self.n_classes)[labels]
