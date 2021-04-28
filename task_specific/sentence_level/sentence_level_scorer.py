import abc

from corpus_processing import scorer


class SentenceLevelScorer(scorer.Scorer, metaclass=abc.ABCMeta):
  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._total_sentences = 0
    self._examples = []
    self._preds = []

  def update(self, examples, predictions, loss):
    super(SentenceLevelScorer, self).update(examples, predictions, loss)
    n_words = 0
    for example, preds in zip(examples, predictions):
      self._examples.append(example)
      self._preds.append(preds)
      n_words += len(example.words) - 2
    self._total_loss += loss
    self._total_sentences += len(examples)

  def get_loss(self):
    return self._total_loss / max(1, self._total_sentences)
