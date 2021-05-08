import abc

from corpus_processing import scorer


class TranslationWordLevelScorer(scorer.Scorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(TranslationWordLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_words = 0
    self._examples = []
    self._preds = []

  def update(self, examples, predictions, loss):
    super(TranslationWordLevelScorer, self).update(examples, predictions, loss)
    n_words = 0
    for example, preds in zip(examples, predictions):
      self._examples.append(example)
      self._preds.append(list(preds))
      n_words += len(example.words)
    self._total_loss += loss * n_words
    self._total_words += n_words

  def get_loss(self):
    return self._total_loss / max(1, self._total_words)
