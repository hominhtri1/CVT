import os
import tensorflow as tf

from base import embeddings
from corpus_processing import example, minibatching


class TranslationDataLoader(object):
  def __init__(self, config, name):
    self._config = config
    self._task_name = name
    self._raw_data_path = os.path.join(config.raw_data_topdir, name)
    self.label_mapping_path = os.path.join(
      config.preprocessed_data_topdir,
      name + '_label_mapping.pkl')

    if self.label_mapping:
      self._n_classes = len(set(self.label_mapping.values()))
    else:
      self._n_classes = None

  def get_dataset(self, split):
    if (split == 'train' and not self._config.for_preprocessing and
        tf.gfile.Exists(os.path.join(self._raw_data_path, 'train_subset.txt'))):
      split = 'train_subset'
    return minibatching.Dataset(
      self._config, self._get_examples(split), self._task_name)

  def get_sentence_tuples(self, split):
    tuples = []
    path = os.path.join(self._raw_data_path, split + '.txt')
    if not tf.gfile.Exists(path):
      if self._config.for_preprocessing:
        return []
      else:
        raise ValueError('Unable to load data from', path)

    f = tf.gfile.GFile(path, 'r')

    line_count = 0

    while (True):
      if split == 'train' and line_count == self._config.train_set_line_count:
        break

      line_src = f.readline()[:-1]

      if line_src == '':
        break

      line_count += 1

      line_tgt = f.readline()[:-1]
      f.readline()

      words_src = line_src.strip().split()
      words_tgt = line_tgt.strip().split()
      size_src = len(words_src)
      size_tgt = len(words_tgt) + 1

      tuples.append((words_src, words_tgt, size_src, size_tgt))

    f.close()

    return tuples

  @property
  def label_mapping(self):
    return {str(x): x for x in range(self._config.tgt_vocab_size)}

  def _get_examples(self, split):
    word_vocab = embeddings.get_word_vocab(self._config)
    word_vocab_vi = embeddings.get_word_vocab_vi(self._config)
    char_vocab = embeddings.get_char_vocab()
    examples = [
        TranslationExample(
            self._config, words_src, words_tgt, size_src, size_tgt,
            word_vocab, char_vocab, self._task_name, word_vocab_vi, split)
        for words_src, words_tgt, size_src, size_tgt in self.get_sentence_tuples(split)
    ]
    return examples

  @staticmethod
  def get_examples_translate(config, src, split):
    words_src = src.strip().split()
    size_src = len(words_src)
    words_tgt = []
    size_tgt = 1

    word_vocab = embeddings.get_word_vocab(config)
    word_vocab_vi = embeddings.get_word_vocab_vi(config)
    char_vocab = embeddings.get_char_vocab()

    examples = [
      TranslationExample(
        config, words_src, words_tgt, size_src, size_tgt,
        word_vocab, char_vocab, 'translate', word_vocab_vi, split)
    ]

    return examples

class TranslationExample(example.Example):
  def __init__(self, config, words_src, words_tgt, size_src, size_tgt,
               word_vocab, char_vocab, task_name, word_vocab_vi, split):
    super(TranslationExample, self).__init__(words_src, word_vocab, char_vocab, False, words_tgt, word_vocab_vi, split)

    #self.words = words_src
    self.size_src = size_src
    self.size_tgt = size_tgt

