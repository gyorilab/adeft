import logging

from deft.util import contains_shortform
from deft.recognize import LongformRecognizer

logger = logging.getLogger('disambiguate')


class DeftDisambiguator(object):
    """Disambiguates longforms in texts for a particular shortform

    Checks first if the standard pattern is matched in the text. If it is
    matched for a unique recognizable longform return this longform. Otherwise
    use a classifier to predict the correct disambiguation for the longform. If
    no longform was detected by matching the standard pattern, return the
    longform the classifier predicted with highest probability. If more than
    one longform was matched using the standard pattern, return the longform
    with highest predicted probability among the matched longforms.

    Parameters
    ----------
    longform_recognizer : py:class:`deft.recognize.LongformRecognizer`
       recognizes longforms based on the standard pattern
    longform_classifier :  py:class:`deft.modeling.classify.LongformClassifier`
       machine learning model for disambiguating shortforms based upon context

    Attributes
    ----------
    shortform : str
        shortform to disambiguate. This is also an attribute of both the
        recognizer and classifier.
    """
    def __init__(self, longform_classifier, grounding_map):
        self.lf_classifier = longform_classifier
        self.shortform = longform_classifier.shortform
        self.lf_recognizer = LongformRecognizer(self.shortform,
                                                grounding_map)

    def disambiguate(self, texts):
        groundings = [self.lf_recognizer.recognize(text)
                      for text in texts]
        undetermined = [text for text, grounding in zip(texts, groundings)
                        if len(grounding) != 1]
        if undetermined:
            preds = self.lf_classifier.predict_proba(undetermined)

        result = [None]*len(texts)
        pred_index = 0
        for index, grounding in enumerate(groundings):
            if len(grounding) == 1:
                disamb = grounding.pop()
                result[index] = (disamb, {disamb: 1.0})
            elif groundings:
                unnormed = {label: prob
                            for label, prob in preds[pred_index].items()}
                norm_factor = sum(preds[pred_index].values())
                pred = {label: prob/norm_factor
                        for label, prob in unnormed.items()}
                disamb = max(pred.keys(),
                             key=lambda key: pred[key])
                result[index] = (disamb, pred)
                pred_index += 1
            else:
                pred = {label: prob
                        for label, prob in preds[pred_index].items()}
                disamb = max(pred.keys(),
                             key=lambda key: pred[key])
                result[index] = (disamb, pred)
                pred_index += 1
        return result
