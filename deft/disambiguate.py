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

    def disambiguate(self, text):
        if contains_shortform(text, self.shortform):
            longforms = self.lf_recognizer.recognize(text)
            if len(longforms) == 1:
                return longforms.pop()
        else:
            longforms = set()
        prediction = self.lf_classifier.predict_proba([text])[0]
        if not longforms:
            return max(prediction.keys(),
                       key=lambda key: prediction[key])
        else:
            return max(prediction[label] for label in longforms)
