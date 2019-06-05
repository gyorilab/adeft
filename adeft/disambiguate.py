import os
import json
import logging

from adeft.locations import MODELS_PATH
from adeft.recognize import DeftRecognizer
from adeft.modeling.classify import load_model
from adeft.download import get_available_models

logger = logging.getLogger(__file__)


class DeftDisambiguator(object):
    """Disambiguates a particular shortform in a list of texts

    Parameters
    ----------
    classifier :  py:class:`adeft.modeling.classify.DeftClassifier`
       machine learning model for disambiguating shortforms based upon context

    grounding_dict : dict
        Dictionary mapping shortforms to grounding_map dictionaries mapping
        longforms to groundings

    names : dict
        dictionary mapping groundings to canonical names

    Attributes
    ----------
    shortform : str
        shortform to disambiguate

    recognizer : py:class:`adeft.recognize.DeftRecognizer`
        recognizer to disambiguate by searching for a defining pattern

    labels : set
        set of labels classifier is able to predict
    """
    def __init__(self, classifier, grounding_dict, names):
        self.classifier = classifier
        self.shortforms = classifier.shortforms
        self.recognizers = [DeftRecognizer(shortform,
                                           grounding_map)
                            for shortform,
                            grounding_map in grounding_dict.items()]
        self.names = names
        self.labels = set(value for grounding_map in grounding_dict.values()
                          for value in grounding_map.values())
        self.pos_labels = classifier.pos_labels

    def disambiguate(self, texts):
        """Return disambiguations for a list of texts

        First checks for defining patterns (DP) within a text. If there is
        an unambiguous match to a longform with a defining pattern, considers
        this to be the correct disambiguation with confidence 1.0.
        If no defining pattern is found, uses a logistic regression model to
        predict the correct disambiguation. If there were multiple longforms
        with different groundings found with a defining pattern, disambiguates
        to the grounding among these with highest predicted probability. If no
        defining pattern was found, disambiguates to the grounding with highest
        predicted probability.

        Parameters
        ----------
        texts : str or list of str
            fulltext or list of fulltexts in which to disambiguate shortform

        Returns
        -------
        result : list of tuple
            Disambiguations for text. For each text the corresponding
            disambiguation is a tuple of three elements. A grounding,
            a canonical name associated with the grounding, and a dictionary
            containing predicted probabilities for each possible grounding
        """
        # Handle case where a single string is passed
        if isinstance(texts, str):
            return self.disambiguate([texts])[0]
        # First disambiguate based on searching for defining patterns
        groundings = []
        for text in texts:
            grounding = set()
            for recognizer in self.recognizers:
                grounding.update(recognizer.recognize(text))
            groundings.append(grounding)
        # For texts without a defining pattern or with inconsistent
        # defining patterns, use the longform classifier.
        undetermined = [text for text, grounding in zip(texts, groundings)
                        if len(grounding) != 1]
        if undetermined:
            preds = self.classifier.predict_proba(undetermined)

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

    def info(self):
        """Get information about disambiguator and its performance.

        Displays disambiguations model is able to produce. Shows class
        balance of disambiguation labels in the models training data and
        crossvalidated F1 score, precision, and recall on training data.
        Classification metrics are given by the weighted average of these
        metrics over positive labels, weighted by number of examples in
        each class in test data. Positive labels are appended with *s in
        the displayed info. Classification metrics may not be available
        depending upon how model was trained.
        """
        if len(self.shortforms) > 1:
            readable_shortforms = (','.join(self.shortforms[:-1]) + ', and ' +
                                   self.shortforms[-1])
        else:
            readable_shortforms = self.shortforms[0]
        output = 'Disambiguation model for %s\n\n' % readable_shortforms
        output += 'Produces the disambiguations:\n'
        for grounding, name in self.names.items():
            pos = '*' if grounding in self.pos_labels else ''
            output += '\t%s%s\t%s\n' % (name, pos, grounding)
        output += '\n'
        if not (hasattr(self.classifier, 'stats') and self.classifier.stats):
            output += 'Model statistics are not available.'
            return output

        model_stats = self.classifier.stats
        output += 'Training data had class balance:\n'
        label_distribution = model_stats['label_distribution']
        for grounding, count in sorted(label_distribution.items(),
                                       key=lambda x: - x[1]):
            name = (self.names[grounding]
                    if grounding in self.names else 'Ungrounded')
            pos = '*' if grounding in self.pos_labels else ''
            output += '\t%s%s\t%s\n' % (name, pos, count)
        output += '\n'
        output += 'Classification Metrics:\n'
        f1 = round(model_stats['f1']['mean'], 5)
        output += '\tF1 score:\t%s\n' % f1

        precision = round(model_stats['precision']['mean'], 5)
        output += '\tPrecision:\t%s\n' % precision

        recall = round(model_stats['recall']['mean'], 5)
        output += '\tRecall:\t\t%s\n' % recall
        output += '\n'

        output += 'See Docstring for explanation of metrics.\n'
        return output


def load_disambiguator(shortform, models_path=MODELS_PATH):
    """Returns deft disambiguator loaded from models directory

    Parameters
    ----------
    model_name : str
        Model_Name to disambiguate
    models_path : Optional[str]
        Path to models directory. Defaults to deft's pretrained models.
        Users have the option to specify a path to another directory to use
        custom models.
    """
    available = get_available_models()
    try:
        model_name = available[shortform]
    except KeyError:
        logger.error('No model available for shortform %s' % shortform)
        return None

    model = load_model(os.path.join(models_path, model_name,
                                    model_name + '_model.gz'))
    with open(os.path.join(models_path, model_name,
                           model_name + '_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(models_path, model_name,
                           model_name + '_names.json')) as f:
        names = json.load(f)
    return DeftDisambiguator(model, grounding_dict, names)
