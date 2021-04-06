from task_specific.sentence_level import sentence_level_scorer


class SentenceAccuracyScorer(sentence_level_scorer.SentenceLevelScorer):
  def __init__(self):
    super(SentenceAccuracyScorer, self).__init__()

  def _get_results(self):
    correct, count = 0, 0
    for example, preds in zip(self._examples, self._preds):
      y_true, y_pred = example.labels, preds
      count += 1
      correct += (1 if y_pred == y_true else 0)
    return [
      ("accuracy", 100.0 * correct / count),
      ("loss", self.get_loss())
    ]