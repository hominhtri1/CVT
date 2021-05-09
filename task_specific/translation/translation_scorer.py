import nltk
import numpy as np

from task_specific.translation import translation_word_level_scorer


class AccuracyScorer(translation_word_level_scorer.TranslationWordLevelScorer):
  def __init__(self, config, auto_fail_label=None):
    super(AccuracyScorer, self).__init__()
    self._auto_fail_label = auto_fail_label
    self._config = config

  def _get_results(self):
    correct, count = 0, 0
    references, hypotheses = [], []
    result_samples = np.random.choice(self._config.train_set_line_count, self._config.result_sample_line_count, replace=False)
    for id in result_samples:
      print(self._examples[id].words_tgt_out)
      print(self._preds[id])
      print()
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
