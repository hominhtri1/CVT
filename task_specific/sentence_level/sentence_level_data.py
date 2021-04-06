import collections
import os
import tensorflow as tf

from base import embeddings, utils
from corpus_processing import minibatching, example


class SentenceClassificationDataLoader(object):
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

  def get_labeled_sentences(self, split):
    sentences = []
    path = os.path.join(self._raw_data_path, split + '.txt')
    if not tf.gfile.Exists(path):
      if self._config.for_preprocessing:
        return []
      else:
        raise ValueError('Unable to load data from', path)

    with tf.gfile.GFile(path, 'r') as f:
      for line in f:
        line = line.strip().split()
        words, tag = line[:-1], line[-1]
        sentences.append((words, tag))
    return sentences

  @property
  def label_mapping(self):
    return {'0': 0, '1': 1}

  def _get_examples(self, split):
    word_vocab = embeddings.get_word_vocab(self._config)
    char_vocab = embeddings.get_char_vocab()
    examples = [
        SentenceClassificationExample(
            self._config, words, tag,
            word_vocab, char_vocab, self.label_mapping, self._task_name)
        for words, tag in self.get_labeled_sentences(split)]
    return examples

class SentenceClassificationExample(example.Example):
  def __init__(self, config, words, original_tag,
               word_vocab, char_vocab, label_mapping, task_name):
    super(SentenceClassificationExample, self).__init__(words, word_vocab, char_vocab)

    self.labels = label_mapping[original_tag]