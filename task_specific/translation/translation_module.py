import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, LuongAttention

from base import utils, embeddings
from corpus_processing import minibatching
from model import task_module, model_helpers


class TranslationModule(task_module.SemiSupervisedModule):
  def __init__(self, config, task_name, n_classes, inputs,
               encoder):
    super(TranslationModule, self).__init__()
    self.task_name = task_name
    self.n_classes = n_classes
    self.words_tgt_in = words_tgt_in = tf.placeholder(tf.int32, [None, None],
                                                      name=task_name + '_words_tgt_in')
    self.words_tgt_out = words_tgt_out = tf.placeholder(tf.float32, [None, None, None],
                                                        name=task_name + '_words_tgt_out')
    self.size_src = size_sr = tf.placeholder(tf.int32, [None],
                                              name=task_name + '_size_src')
    self.size_tgt = size_tgt = tf.placeholder(tf.int32, [None],
                                              name=task_name + '_size_tgt')
    pretrained_embeddings_vi = utils.load_cpickle(config.word_embeddings_vi)

    class PredictionModule(object):
      def __init__(self, name, input_reprs, roll_direction=0, activate=True, is_translate=False, word_in=None, encoder_reprs=encoder.bi_reprs):
        self.name = name
        with tf.variable_scope(name + '/predictions'):
          #decoder_state = tf.layers.dense(input_reprs, config.projection_size, name='encoder_to_decoder')
          decoder_state = input_reprs

          with tf.variable_scope('word_embeddings_vi'):
            word_embedding_matrix = tf.get_variable(
                'word_embedding_matrix_vi', initializer=pretrained_embeddings_vi)
            if is_translate:
              word_embeddings = tf.nn.embedding_lookup(
                word_embedding_matrix, word_in)
            else:
              word_embeddings = tf.nn.embedding_lookup(
                word_embedding_matrix, words_tgt_in)
            word_embeddings = tf.nn.dropout(word_embeddings, inputs.keep_prob)
            word_embeddings *= tf.get_variable('emb_scale', initializer=1.0)

          decoder_lstm = model_helpers.lstm_cell(config.bidirectional_sizes[0], inputs.keep_prob,
                                      config.projection_size)

          decoder_output_layer = tf.layers.Dense(n_classes, name='predict')

          if not is_translate:
            attention_mechanism = LuongAttention(
              num_units=config.attention_units,
              memory=encoder_reprs,
              memory_sequence_length=size_sr,
              scale=True)
            attention_cell = AttentionWrapper(
              decoder_lstm,
              attention_mechanism,
              attention_layer_size=config.attention_units)

            batch_size = tf.shape(words_tgt_in)[0]
            decoder_initial_state = attention_cell.zero_state(
              dtype=tf.float32,
              batch_size=batch_size * config.beam_width)
            decoder_state = decoder_initial_state.clone(cell_state=decoder_state)

            helper = tf.contrib.seq2seq.TrainingHelper(
              word_embeddings,
              size_tgt)

            decoder = tf.contrib.seq2seq.BasicDecoder(
              attention_cell,
              helper,
              decoder_state,
              decoder_output_layer)

            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
              decoder)
            # swap_memory=True)

            self.logits = outputs.rnn_output
          else:
            if config.decode_mode == 'greedy':
              helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                word_embedding_matrix,
                [embeddings.START, embeddings.START],
                embeddings.END)

              decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_lstm,
                helper,
                decoder_state,
                decoder_output_layer)
            elif config.decode_mode == 'beam':
              encoder_reprs = tf.contrib.seq2seq.tile_batch(encoder_reprs, multiplier=config.beam_width)
              decoder_state = tf.contrib.seq2seq.tile_batch(decoder_state, multiplier=config.beam_width)
              size_src = tf.contrib.seq2seq.tile_batch(size_sr, multiplier=config.beam_width)

              attention_mechanism = LuongAttention(
                num_units=config.attention_units,
                memory=encoder_reprs,
                memory_sequence_length=size_src,
                scale=True)
              attention_cell = AttentionWrapper(
                decoder_lstm,
                attention_mechanism,
                attention_layer_size=config.attention_units)

              batch_size = 2
              decoder_initial_state = attention_cell.zero_state(
                dtype=tf.float32,
                batch_size=batch_size * config.beam_width)
              decoder_state = decoder_initial_state.clone(cell_state=decoder_state)

              #decoder_state = tf.contrib.seq2seq.tile_batch(
              #  decoder_state, multiplier=config.beam_width)

              decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=attention_cell,
                embedding=word_embedding_matrix,
                start_tokens=[embeddings.START, embeddings.START],
                end_token=embeddings.END,
                initial_state=decoder_state,
                beam_width=config.beam_width,
                output_layer=decoder_output_layer)

            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
              decoder,
              maximum_iterations=config.max_translate_length)
              #swap_memory=True)

            if config.decode_mode == 'greedy':
              self.sample_ids = outputs.sample_id
            elif config.decode_mode == 'beam':
              self.sample_ids = outputs.predicted_ids

          '''
          outputs, state = tf.nn.dynamic_rnn(
            model_helpers.lstm_cell(config.bidirectional_sizes[0], inputs.keep_prob,
                                    config.projection_size),
            word_embeddings,
            initial_state=decoder_state,
            dtype=tf.float32,
            sequence_length=size_tgt,
            scope='predictlstm'
          )
          '''

          self.state = state

          #self.logits = tf.layers.dense(outputs, n_classes, name='predict')
          #self.logits = tf.layers.dense(outputs.rnn_output, n_classes, name='predict')

        if is_translate:
          return

        targets = words_tgt_out
        targets *= (1 - inputs.label_smoothing)
        targets += inputs.label_smoothing / n_classes
        self.loss = model_helpers.masked_ce_loss(
          self.logits, targets, inputs.mask)

    primary = PredictionModule('primary', encoder.bi_state, encoder_reprs=encoder.bi_reprs)

    self.unsupervised_loss = primary.loss
    self.supervised_loss = primary.loss
    self.probs = tf.nn.softmax(primary.logits)
    self.preds = tf.argmax(primary.logits, axis=-1)

    ###

    self.word_in = tf.placeholder(tf.int32, [None, None], name=task_name + '_word_in')
    self.state_c_in = tf.placeholder(tf.float32, [None, None], name=task_name + '_state_c_in')
    self.state_h_in = tf.placeholder(tf.float32, [None, None], name=task_name + '_state_h_in')

    state_in = tf.nn.rnn_cell.LSTMStateTuple(self.state_c_in, self.state_h_in)

    translate_primary = PredictionModule('primary', state_in, is_translate=True, word_in=self.word_in, encoder_reprs=encoder.bi_reprs)

    #self.translate_preds = tf.argmax(translate_primary.logits, axis=-1)
    self.translate_preds = translate_primary.sample_ids
    self.translate_state = translate_primary.state

  def update_feed_dict(self, feed, mb):
    words_tgt_in = minibatching.build_array([e.words_tgt_in for e in mb.examples])
    feed[self.words_tgt_in] = words_tgt_in

    words_tgt_out = minibatching.build_array([e.words_tgt_out for e in mb.examples])
    feed[self.words_tgt_out] = np.eye(self.n_classes)[words_tgt_out]

    size_src = [e.size_src for e in mb.examples]
    feed[self.size_src] = size_src

    size_tgt = [e.size_tgt for e in mb.examples]
    feed[self.size_tgt] = size_tgt

  def update_feed_dict_translate(self, feed, word_in=None, state_in=None, size_tgt=None):
    #feed[self.word_in] = [[word_in]]
    feed[self.state_c_in] = [state_in.c[0], state_in.c[0]]
    feed[self.state_h_in] = [state_in.h[0], state_in.h[0]]
    #feed[self.size_tgt] = [1]
    feed[self.size_tgt] = [size_tgt, size_tgt]
