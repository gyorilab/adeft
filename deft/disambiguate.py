import logging

from deft.util import contains_shortform

logger = logging.getLogger('disambiguate')


class Disambiguator(object):
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
    def __init__(self, longform_recognizer, longform_classifier):
        self.lf_recognizer = longform_recognizer
        self.lf_classifier = longform_classifier
        assert longform_recognizer.shortform == longform_recognizer.shortform
        assert longform_classifier.shortform == longform_classifier.shortform
        self.shortform = longform_recognizer.shortform

    def disambiguate(self, text):
        if contains_shortform(text, self.shortform):
            longforms = self.lf_recognizer.recognize(text)
            if len(longforms) == 1:
                return longforms.pop()
            else:
                prediction = self.lf_classifier.predict_proba([text])[0]
                if not longforms:
                    return max(prediction.iterkeys(),
                               key=lambda key: prediction[key])
                else:
                    return max(prediction[label] for label in longforms)
