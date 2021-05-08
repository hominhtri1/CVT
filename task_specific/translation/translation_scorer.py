import nltk

from task_specific.translation import translation_word_level_scorer


class AccuracyScorer(translation_word_level_scorer.TranslationWordLevelScorer):
  def __init__(self, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label

  def _get_results(self):
    correct, count = 0, 0
    references, hypotheses = [], []
    for example, preds in zip(self._examples, self._preds):
      for y_true, y_pred in zip(example.words_tgt_out, preds):
        count += 1
        correct += (1 if y_pred == y_true and y_true != self._auto_fail_label
                    else 0)
      references.append([example.words_tgt_out])
      hypotheses.append(preds[:len(example.words_tgt_out)])
    return [
        ("accuracy", 100.0 * correct / count),
        ("loss", self.get_loss()),
        ("bleu", nltk.translate.bleu_score.corpus_bleu(references, hypotheses) * 100)
    ]
