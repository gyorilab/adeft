import os
import json
import logging

from deft.locations import MODELS_PATH
from deft.recognize import LongformRecognizer
from deft.modeling.classify import load_model

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
    longform_classifier :  py:class:`deft.modeling.classify.LongformClassifier`
       machine learning model for disambiguating shortforms based upon context

    grounding_map : dict
        Dictionary mapping longforms to their groundings

    names : dict
        dictionary mapping groundings to standardized names

    Attributes
    ----------
    shortform : str
        shortform to disambiguate

    lf_recognizer : py:class:`deft.recognize.LongformRecognizer`
        recognizer to disambiguate by searching for a defining pattern

    labels : set
        set of labels classifier is able to predict
    """
    def __init__(self, longform_classifier, grounding_map, names):
        self.lf_classifier = longform_classifier
        self.shortform = longform_classifier.shortform
        self.lf_recognizer = LongformRecognizer(self.shortform,
                                                grounding_map)
        self.names = names
        self.labels = set(grounding_map.values())

    def disambiguate(self, texts):
        """Return disambiguations for a list of texts

        Parameters
        ----------
        texts : list of str
            fulltexts to disambiguate shortform

        Returns
        -------
        result : list of tuple
            Disambiguations for text. For each text the corresponding
            disambiguation is a tuple of three elements. A grounding,
            a standardized name for the grounding, and a dictionary
            containing predicted probabilities for different groundings
        """
        # First disambiguate based on searching for defining patterns
        groundings = [self.lf_recognizer.recognize(text)
                      for text in texts]
        # For texts without a defining pattern or with inconsistent
        # defining patterns, use the longform classifier.
        undetermined = [text for text, grounding in zip(texts, groundings)
                        if len(grounding) != 1]
        if undetermined:
            preds = self.lf_classifier.predict_proba(undetermined)

        result = [None]*len(texts)
        # each time we have to use a prediction from the longform classifier
        # this is incremented so we can keep track of which prediction is to
        # be used next
        pred_index = 0
        for index, grounding in enumerate(groundings):
            if len(grounding) == 1:
                # if an unambiguous defining pattern exists, use this
                # as the disambiguation. set the probability of this
                # grounding to one
                disamb = grounding.pop()
                pred = {label: 0. for label in self.labels}
                pred[disamb] = 1.0
                result[index] = (disamb, self.names.get(disamb), pred)
            elif grounding:
                # if inconsistent defining patterns exist, disambiguate
                # to the one with highest predicted probability. Set the
                # probability of the multiple groundings to sum to one
                unnormed = {label: preds[pred_index][label] if
                            label in grounding else 0.
                            for label in self.labels}
                norm_factor = sum(unnormed.values())
                pred = {label: prob/norm_factor
                        for label, prob in unnormed.items()}
                disamb = max(pred.keys(),
                             key=lambda key: pred[key])
                result[index] = (disamb, self.names.get(disamb), pred)
                pred_index += 1
            else:
                # otherwise use the longform classifier directly
                pred = {label: prob
                        for label, prob in preds[pred_index].items()}
                disamb = max(pred.keys(),
                             key=lambda key: pred[key])
                result[index] = (disamb, self.names.get(disamb), pred)
                pred_index += 1
        return result


def load_disambiguator(shortform, models_path=MODELS_PATH):
    """Returns deft disambiguator loaded from models directory

    Parameters
    ----------
    shortform : str
        Shortform to disambiguate
    models_path : Optional[str]
        Path to models directory. Defaults to deft pretrained models loaded
        from s3. User has the option to specify a path to another directory
        to use custom models
    """
    model = load_model(os.path.join(MODELS_PATH, shortform,
                                    shortform.lower() + '_model.gz'))
    with open(os.path.join(MODELS_PATH, shortform,
                           shortform.lower() + '_grounding_map.json')) as f:
        grounding_map = json.load(f)
    with open(os.path.join(MODELS_PATH, shortform,
                           shortform.lower() + '_names.json')) as f:
        names = json.load(f)
    return DeftDisambiguator(model, grounding_map, names)
