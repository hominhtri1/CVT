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

"""Run training and evaluation for CVT text models."""





import tensorflow as tf

from base import configure, embeddings
from base import utils
from training import trainer
from training import training_progress


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', '"train" or "eval')
tf.app.flags.DEFINE_string('model_name', 'default_model',
                           'A name identifying the model being '
                           'trained/evaluated')


def main():
  utils.heading('SETUP')
  config = configure.Config(mode=FLAGS.mode, model_name=FLAGS.model_name)
  config.write()
  if config.mode == 'encode':
    word_vocab = embeddings.get_word_vocab(config)
    sentence = "Squirrels , for example , would show up , look for the peanut , go away .".split()
    sentence = ([word_vocab[embeddings.normalize_word(w)] for w in sentence])
    print(sentence)
    return
  if config.mode == 'decode':
    word_vocab_reversed = embeddings.get_word_vocab_reversed(config)
    sentence = "25709 33 42 879 33 86 304 92 33 676 42 32 13406 33 273 445 34".split()
    sentence = ([word_vocab_reversed[int(w)] for w in sentence])
    print(sentence)
    return
  if config.mode == 'encode-vi':
    word_vocab_vi = embeddings.get_word_vocab_vi(config)
    print(len(word_vocab_vi))
    sentence = "Mỗi_một khoa_học_gia đều thuộc một nhóm nghiên_cứu , và mỗi nhóm đều nghiên_cứu rất nhiều đề_tài đa_dạng .".split()
    sentence = ([word_vocab_vi[embeddings.normalize_word(w)] for w in sentence])
    print(sentence)
    return
  if config.mode == 'decode-vi':
    word_vocab_reversed_vi = embeddings.get_word_vocab_reversed_vi(config)
    sentence = "8976 32085 129 178 17 261 381 5 7 195 261 129 381 60 37 2474 1903 6".split()
    sentence = ([word_vocab_reversed_vi[int(w)] for w in sentence])
    print(sentence)
    return
  if config.mode == 'embed':
    word_embeddings = embeddings.get_word_embeddings(config)
    word = 50
    embed = word_embeddings[word]
    print(' '.join(str(x) for x in embed))
    return
  if config.mode == 'embed-vi':
    word_embeddings_vi = embeddings.get_word_embeddings_vi(config)
    word = 50
    embed = word_embeddings_vi[word]
    print(' '.join(str(x) for x in embed))
    return
  with tf.Graph().as_default() as graph:
    model_trainer = trainer.Trainer(config)
    summary_writer = tf.summary.FileWriter(config.summaries_dir)
    checkpoints_saver = tf.train.Saver(max_to_keep=1)
    best_model_saver = tf.train.Saver(max_to_keep=1)
    init_op = tf.global_variables_initializer()
    graph.finalize()
    with tf.Session() as sess:
      sess.run(init_op)
      progress = training_progress.TrainingProgress(
          config, sess, checkpoints_saver, best_model_saver,
          config.mode == 'train')
      utils.log()
      if config.mode == 'train':
        #summary_writer.add_graph(sess.graph)
        utils.heading('START TRAINING ({:})'.format(config.model_name))
        model_trainer.train(sess, progress, summary_writer)
      elif config.mode == 'eval-train':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
            config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None, train_set=True)
      elif config.mode == 'eval-dev':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
            config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None, train_set=False)
      elif config.mode == 'infer':
        utils.heading('START INFER ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
            config.checkpoints_dir))
        model_trainer.infer(sess)
      elif config.mode == 'translate':
        utils.heading('START TRANSLATE ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
          config.checkpoints_dir))
        model_trainer.translate(sess)
      elif config.mode == 'eval-translate-train':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
          config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None, train_set=True, is_translate=True)
      elif config.mode == 'eval-translate-dev':
        utils.heading('RUN EVALUATION ({:})'.format(config.model_name))
        progress.best_model_saver.restore(sess, tf.train.latest_checkpoint(
          config.checkpoints_dir))
        model_trainer.evaluate_all_tasks(sess, summary_writer, None, train_set=False, is_translate=True)
      else:
        raise ValueError('Mode must be "train" or "eval"')


if __name__ == '__main__':
  main()
