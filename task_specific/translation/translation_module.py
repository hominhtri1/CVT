import numpy as np
import tensorflow as tf

from corpus_processing import minibatching
from model import task_module, model_helpers


class TranslationModule(task_module.SemiSupervisedModule):
  def __init__(self, config, task_name, n_classes, inputs,
               encoder):
    super(TranslationModule, self).__init__()
    self.task_name = task_name
    self.n_classes = n_classes
    self.words_tgt_in = words_tgt_in = tf.placeholder(tf.int64, [None, None],
                                                      name=task_name + '_words_tgt_in')
    self.words_tgt_out = words_tgt_out = tf.placeholder(tf.float32, [None, None, None],
                                                        name=task_name + '_words_tgt_out')
    self.size_src = size_src = tf.placeholder(tf.int64, [None],
                                              name=task_name + '_size_src')
    self.size_tgt = size_tgt = tf.placeholder(tf.int64, [None],
                                              name=task_name + '_size_tgt')

    class PredictionModule(object):
      def __init__(self, name, input_reprs, roll_direction=0, activate=True):
        self.name = name
        with tf.variable_scope(name + '/predictions'):
          decoder_state = tf.layers.dense(input_reprs, config.projection_size, name='encoder_to_decoder')

          with tf.variable_scope('word_embeddings'):
            word_embedding_matrix = tf.get_variable(
                'word_embedding_matrix',
                [config.en_vocab_size, config.word_embedding_size],
                dtype=tf.float32,
                initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
            word_embeddings = tf.nn.embedding_lookup(
                word_embedding_matrix, words_tgt_in)
            word_embeddings = tf.nn.dropout(word_embeddings, inputs.keep_prob)
            word_embeddings *= tf.get_variable('emb_scale', initializer=1.0)

          outputs, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.BasicRNNCell(config.projection_size),
            word_embeddings,
            initial_state=decoder_state,
            dtype=tf.float32,
            sequence_length=inputs.lengths,
            scope='predictlstm'
          )

          self.logits = tf.layers.dense(outputs, n_classes, name='predict')

        mask = minibatching.build_array([[1] * size_tgt[i] + [0] * (inputs.length[i] - size_tgt[i])
                                         for i in tf.shape(size_tgt)[0]])

        targets = words_tgt_out
        targets *= (1 - inputs.label_smoothing)
        targets += inputs.label_smoothing / n_classes
        self.loss = model_helpers.masked_ce_loss(
          self.logits, targets, mask)

    primary = PredictionModule('primary', encoder.bi_state)

    self.unsupervised_loss = primary.loss
    self.supervised_loss = primary.loss
    self.probs = tf.nn.softmax(primary.logits)
    self.preds = tf.argmax(primary.logits, axis=-1)

  def update_feed_dict(self, feed, mb):
    words_tgt_in = [e.words_tgt_in for e in mb.examples]
    feed[self.words_tgt_in] = words_tgt_in

    words_tgt_out = minibatching.build_array([e.words_tgt_out for e in mb.examples])
    feed[self.words_tgt_out] = np.eye(self.n_classes)[words_tgt_out]

    size_src = [e.size_src for e in mb.examples]
    feed[self.size_src] = size_src

    size_tgt = [e.size_tgt for e in mb.examples]
    feed[self.size_tgt] = size_tgt
