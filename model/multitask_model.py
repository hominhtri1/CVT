# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A multi-task and semi-supervised NLP model."""





import tensorflow as tf

from base import embeddings
from corpus_processing.minibatching import Dataset
from model import encoder
from model import shared_inputs
from task_specific.translation.translation_data import TranslationDataLoader


class Inference(object):
  def __init__(self, config, inputs, pretrained_embeddings, tasks):
    with tf.variable_scope('encoder'):
      self.encoder = encoder.Encoder(config, inputs, pretrained_embeddings)
    self.modules = {}
    for task in tasks:
      with tf.variable_scope(task.name):
        self.modules[task.name] = task.get_module(inputs, self.encoder)


class Model(object):
  def __init__(self, config, pretrained_embeddings, tasks):
    self._config = config
    self._tasks = tasks

    self._global_step, self._optimizer = self._get_optimizer()
    self._inputs = shared_inputs.Inputs(config)
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
      inference = Inference(config, self._inputs, pretrained_embeddings,
                            tasks)
      self._trainer = inference
      self._tester = inference
      self._teacher = inference
      if config.ema_test or config.ema_teacher:
        ema = tf.train.ExponentialMovingAverage(config.ema_decay)
        model_vars = tf.get_collection("trainable_variables", "model")
        ema_op = ema.apply(model_vars)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        def ema_getter(getter, name, *args, **kwargs):
          var = getter(name, *args, **kwargs)
          return ema.average(var)

        scope.set_custom_getter(ema_getter)
        inference_ema = Inference(
            config, self._inputs, pretrained_embeddings, tasks)
        if config.ema_teacher:
          self._teacher = inference_ema
        if config.ema_test:
          self._tester = inference_ema

    self._unlabeled_loss = self._get_consistency_loss(tasks)
    self._unlabeled_train_op = self._get_train_op(self._unlabeled_loss)
    self._labeled_train_ops = {}
    for task in self._tasks:
      task_loss = self._trainer.modules[task.name].supervised_loss
      self._labeled_train_ops[task.name] = self._get_train_op(task_loss)

  def _get_consistency_loss(self, tasks):
    return sum([self._trainer.modules[task.name].unsupervised_loss
                for task in tasks])

  def _get_optimizer(self):
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)
    warm_up_multiplier = (tf.minimum(tf.to_float(global_step),
                                     self._config.warm_up_steps)
                          / self._config.warm_up_steps)
    decay_multiplier = 1.0 / (1 + self._config.lr_decay *
                              tf.sqrt(tf.to_float(global_step)))
    lr = self._config.lr * warm_up_multiplier * decay_multiplier
    #optimizer = tf.train.MomentumOptimizer(lr, self._config.momentum)
    optimizer = tf.train.AdamOptimizer(self._config.tri_lr)
    return global_step, optimizer

  def _get_train_op(self, loss):
    grads, vs = list(zip(*self._optimizer.compute_gradients(loss)))
    grads, _ = tf.clip_by_global_norm(grads, self._config.grad_clip)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      return self._optimizer.apply_gradients(
          list(zip(grads, vs)), global_step=self._global_step)

  def _create_feed_dict(self, mb, model, is_training=True):
    feed = self._inputs.create_feed_dict(mb, is_training)
    if mb.task_name in model.modules:
      model.modules[mb.task_name].update_feed_dict(feed, mb)
    else:
      for module in list(model.modules.values()):
        module.update_feed_dict(feed, mb)
    return feed

  def train_unlabeled(self, sess, mb):
    return sess.run([self._unlabeled_train_op, self._unlabeled_loss],
                    feed_dict=self._create_feed_dict(mb, self._trainer))[1]

  def train_labeled(self, sess, mb):
    return sess.run([self._labeled_train_ops[mb.task_name],
                     self._trainer.modules[mb.task_name].supervised_loss,],
                    feed_dict=self._create_feed_dict(mb, self._trainer))[1]

  def run_teacher(self, sess, mb):
    result = sess.run({task.name: self._teacher.modules[task.name].probs
                       for task in self._tasks},
                      feed_dict=self._create_feed_dict(mb, self._teacher,
                                                       False))
    for task_name, probs in result.items():
      mb.teacher_predictions[task_name] = probs.astype('float16')

  def test(self, sess, mb):
    return sess.run(
        [self._tester.modules[mb.task_name].supervised_loss,
         self._tester.modules[mb.task_name].preds],
        feed_dict=self._create_feed_dict(mb, self._tester, False))

  def get_global_step(self, sess):
    return sess.run(self._global_step)

  def translate(self, sess, src):
    examples = TranslationDataLoader.get_examples_translate(self._config, src, 'translate')
    mb = Dataset.make_minibatch_translate(self._config, examples)

    state = sess.run(
        [self._tester.encoder.bi_state],
        feed_dict=self._create_feed_dict(mb, self._tester, False))
    state = state[0]

    tgt_list = []
    translate_module = self._tester.modules['translate']
    word_vocab_reversed_vi = embeddings.get_word_vocab_reversed_vi(self._config)
    cur_word = 2

    while True:
      word_out_arr, state = sess.run(
        [translate_module.translate_preds,
         translate_module.translate_state],
        feed_dict=translate_module.create_feed_dict_translate(cur_word, state))
      word_out = word_out_arr[0, 0]
      state = state[0]

      word_out_str = word_vocab_reversed_vi(word_out)
      tgt_list.append(word_out_str)

      if word_out_str == '<end>' or len(tgt_list) == 100:
        break

      cur_word = word_out

    tgt = ' '.join(str(x) for x in tgt_list)
    return tgt
