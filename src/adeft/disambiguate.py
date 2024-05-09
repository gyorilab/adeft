"""Implements classes to disambiguate shortforms given text context."""

import os
import json
import logging
import numpy as np
from hashlib import md5


from adeft.locations import ADEFT_MODELS_PATH
from adeft.recognize import AdeftRecognizer
from adeft.modeling.classify import load_model
from adeft.download import get_available_models

logger = logging.getLogger(__file__)


class AdeftDisambiguator(object):
    """Disambiguates a particular shortform in a list of texts

    Parameters
    ----------
    classifier :  py:class:`adeft.modeling.classify.AdeftClassifier`
        Machine learning model for disambiguating shortforms based upon context
    grounding_dict : dict
        Dictionary mapping shortforms to grounding_map dictionaries mapping
        longforms to groundings
    names : dict
        Dictionary mapping groundings to canonical names

    Attributes
    ----------
    shortforms : list of str
        Shortforms to disambiguate
    recognizers : list of py:class:`adeft.recognize.AdeftRecognizer`
        A list of recognizers, one for each shortform, to disambiguate by
        searching for a defining pattern.
    labels : set
        Set of labels that the classifier is able to predict.
    pos_labels : list of str
        List of labels of interest. Only these are considered when
        calculating the micro averaged f1 score for a classifier.
    """
    def __init__(self, classifier, grounding_dict, names):
        self.classifier = classifier
        self.shortforms = classifier.shortforms
        self.recognizers = [AdeftRecognizer(shortform,
                                            grounding_map)
                            for shortform,
                            grounding_map in grounding_dict.items()]
        self.grounding_dict = grounding_dict
        self.names = names
        self.labels = (set(value for grounding_map in grounding_dict.values()
                           for value in grounding_map.values()) |
                       set(classifier.estimator.classes_))
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
        result : tuple or list of tuple
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
                grounding.update({x['grounding']
                                  for x in recognizer.recognize(text)})
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

    def update_pos_labels(self, pos_labels):
        """Update which labels are considered pos_labels

        Micro-averaged precision, recall, and f1 scores are also updated.

        Warning: If this method is called on a disambiguator trained with a
        a version prior to 0.10.0, global precision, recall, and f1 will be set
        to NaN. Older disambiguators must be retrained to update positive
        labels and recompute model statistics.

        Parameters
        ----------
        pos_labels : list
            list of strs. Should be a subset of the labels produced by the
            underlying classifier. Check the labels attribute of the
            AdeftDisambiguator to see which labels are produced.
        """
        labels = list(self.labels)
        stats = self.classifier.stats
        confusion = self.classifier.confusion_info
        if stats is not None and confusion is not None:
            num_splits = len(confusion[labels[0]][labels[0]])
            TP = np.zeros(num_splits, dtype=int)
            FP = np.zeros(num_splits, dtype=int)
            FN = np.zeros(num_splits, dtype=int)
            for label1 in self.labels:
                for label2 in self.labels:
                    row = np.array(confusion[label1][label2])
                    if label1 == label2 and label1 in pos_labels:
                        TP += row
                    if label1 != label2 and label1 in pos_labels:
                        FN += row
                    if label1 != label2 and label2 in pos_labels:
                        FP += row
            Pr = TP/(TP + FP)
            Rc = TP/(TP + FN)
            Pr[Pr == float('inf')] = 0.
            Rc[Rc == float('inf')] = 0.
            F1 = 2/(1/Pr + 1/Rc)
            stats['f1']['mean'] = np.round(np.mean(F1), 6)
            stats['f1']['std'] = np.round(np.std(F1), 6)
            stats['precision']['mean'] = np.round(np.mean(Pr), 6)
            stats['precision']['std'] = np.round(np.std(Pr), 6)
            stats['recall']['mean'] = np.round(np.mean(Rc), 6)
            stats['recall']['std'] = np.round(np.std(Rc), 6)
        elif (stats is not None and
              set(pos_labels) != set(self.pos_labels)):
            stats['f1']['mean'] = float('nan')
            stats['f1']['std'] = float('nan')
            stats['precision']['mean'] = float('nan')
            stats['precision']['std'] = float('nan')
            stats['recall']['mean'] = float('nan')
            stats['recall']['std'] = float('nan')
        self.classifier.stats = stats
        self.classifier.pos_labels = list(pos_labels)
        self.pos_labels = list(pos_labels)

    def modify_groundings(self, new_groundings=None, new_names=None):
        """Update groundings and standardized names

        Modify groundings and standard names for the disambiguator without
        retraining. Cannot map two existing groundings to a single new
        grounding, as this leads to a nontrivial change in the model rather
        than just a relabeling.

        Parameters
        ----------
        new_groundings : Optional[dict]
            Dictionary mapping a subset of previous groundings to updated
            groundings. If None, no groundings are modified. Default: None

        new_names : Optional[dict]
            Dictionary mapping a subset of previous groundings to updated
            names. If None, no names are modified. Default: None
        """
        if new_names is not None:
            # Check if keys in new_names are a subset of current groundings
            if not (set(new_names.keys()) <=
                    set(self.names.keys())):
                raise ValueError('Keys of new names are not a subset of'
                                 ' the current groundings')
            # Update names in names dictionary. Keep groundings the same
            self.names = {grounding: new_names[grounding]
                          if grounding in new_names
                          else name
                          for grounding, name in self.names.items()}

        if new_groundings is not None:
            # Check if keys in new_groundings are a subset of
            # current groundings
            if not (set(new_groundings.keys()) <=
                    set(self.names.keys())):
                raise ValueError('Keys of new groundings are not a subset of'
                                 ' the current groundings')
            # Update keys of names dictionary to new groundings
            names = {(new_groundings[grounding]
                      if grounding in new_groundings
                      else grounding): name
                     for grounding, name in self.names.items()}
            # Check that two previously distinct labels have not been merged
            if len(names) != len(self.names):
                raise ValueError('Previously distinct groundings have been'
                                 ' merged')
            self.names = names
            # Update groundings in grounding_dict
            self.grounding_dict = {shortform:
                                   {phrase:
                                    (new_groundings[grounding]
                                     if grounding in new_groundings
                                     else grounding)
                                    for phrase, grounding in
                                    grounding_map.items()}
                                   for shortform, grounding_map
                                   in self.grounding_dict.items()}
            # Update positive labels in disambiguator
            self.pos_labels = [new_groundings[grounding]
                               if grounding in new_groundings
                               else grounding
                               for grounding in self.pos_labels]
            # Update classifier
            classifier = self.classifier
            # Update positive labels
            classifier.pos_labels = self.pos_labels
            # Updated class labels. (This will change the labels for the
            # predictions the classifier makes)
            for index, label in enumerate(classifier.estimator.classes_):
                if label in new_groundings:
                    new_label = new_groundings[label]
                    classifier.estimator.classes_[index] = new_label
            # Update labels in model statistics so info can be updated
            if hasattr(classifier, 'stats') and classifier.stats:
                label_dist = classifier.stats['label_distribution']
                label_dist = {(new_groundings[label]
                               if label in new_groundings
                               else label):
                              count
                              for label, count in label_dist.items()}
                classifier.stats['label_distribution'] = label_dist
                classifier.stats = {new_groundings[label]
                                    if label in new_groundings else label:
                                    value for label, value in
                                    classifier.stats.items()}

    def dump(self, model_name, path=None):
        """Save disambiguator to disk

        Parameters
        ----------
        model_name : str
            Model files will be saved in directory with this name.
        path : Optional[str]
            Path where model is to be stored. Defaults to current directory.
            Default: None
        """
        if path is None:
            path = os.getcwd()

        grounding_dict = self.grounding_dict
        names = self.names
        classifier = self.classifier

        model_path = os.path.join(path, model_name)
        # Create model directory if it does not already exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        classifier.dump_model(os.path.join(model_path,
                                           '%s_model.gz' % model_name))
        with open(os.path.join(model_path,
                               '%s_grounding_dict.json'
                               % model_name), 'w') as f:
            json.dump(grounding_dict, f)
        with open(os.path.join(model_path, '%s_names.json'
                               % model_name), 'w') as f:
            json.dump(names, f)

    def version(self):
        """Returns version string for disambiguator

        Returns
        -------
        str
            String of the form
            <adeft_version>::<timestamp>::<hash>
            where <hash> is the md5 hash of the grounding_dict
            jsonified with sorted keys.
        """
        model = self.classifier
        try:
            timestamp = model.timestamp
            adeft_version = model.version
        except AttributeError:
            logger.warning('Information is not available to calculate'
                           ' model version')
            return None
        gdict_json = json.dumps(self.grounding_dict, sort_keys=True)
        gdict_hash = md5(gdict_json.encode('utf-8')).hexdigest()
        return '%s::%s::%s' % (adeft_version, timestamp, gdict_hash)

    def info(self):
        """Get information about disambiguator and its performance.

        Displays disambiguations model is able to produce. Shows class
        balance of disambiguation labels in the models training data and
        crossvalidated F1 score, precision, and recall on training data.
        Classification metrics for multi-label data are calculated by taking
        the micro-average over the positive labels. This means the metrics
        are calculated globally by counting the total true positives,
        false negatives, and false positives. Positive labels are starred in
        in the displayed output. F1, Precision, and Recall are also shown for
        for each label separately. Classification metrics may not be available
        depending upon how the model was trained.

        Returns
        -------
        str
            A string representing the information about the disambigutor.
        """
        if len(self.shortforms) > 1:
            readable_shortforms = (','.join(self.shortforms[:-1]) + ', and ' +
                                   self.shortforms[-1])
        else:
            readable_shortforms = self.shortforms[0]
        output = 'Disambiguation model for %s\n\n' % readable_shortforms
        output += 'Produces the disambiguations:\n'
        for grounding, name in sorted(self.names.items(), key=lambda x: x[1]):
            pos = '*' if grounding in self.pos_labels else ''
            output += '\t%s%s\t%s\n' % (name, pos, grounding)
        output += '\n'
        if not (hasattr(self.classifier, 'stats') and self.classifier.stats):
            output += 'Model statistics are not available.'
            return output

        model_stats = self.classifier.stats
        output += 'Class level metrics:\n'
        output += '--------------------\n'
        label_distribution = model_stats['label_distribution']
        # number of digits after the decimal place to report when
        # displaying value of a metric
        metric_digits = 5
        name_pad = max((len(val) for val in self.names.values()))
        count_pad = max(len(str(count)) for count
                        in label_distribution.values())
        metric_pad = metric_digits + 2
        header = '%s\t%s\t%s\n' % ('Grounding'.ljust(name_pad),
                                   'Count'.ljust(count_pad),
                                   'F1'.ljust(metric_pad))
        output += header
        for grounding, count in sorted(label_distribution.items(),
                                       key=lambda x: - x[1]):
            name = (self.names[grounding]
                    if grounding in self.names else 'Ungrounded')
            pos = '*' if grounding in self.pos_labels else ''
            try:
                f1 = round(model_stats[grounding]['f1']['mean'], metric_digits)
            except KeyError:
                f1 = ''
            output += '%s%s\t%s\t%s\n' % (name.rjust(name_pad), pos,
                                          str(count).rjust(count_pad),
                                          str(f1).rjust(metric_pad))
        output += '\n'
        output += 'Global Metrics:\n'
        output += '-----------------\n'
        f1 = round(model_stats['f1']['mean'], 5)
        output += '\tF1 score:\t%s\n' % f1

        precision = round(model_stats['precision']['mean'], 5)
        output += '\tPrecision:\t%s\n' % precision

        recall = round(model_stats['recall']['mean'], 5)
        output += '\tRecall:\t\t%s\n' % recall
        output += '\n'

        output += '* Positive labels\n'
        output += 'See Docstring for explanation\n'
        return output


def load_disambiguator(shortform, path=ADEFT_MODELS_PATH):
    """Returns adeft disambiguator loaded from models directory

    Searches folder specified by path for a disambiguation model
    that can disambiguate the given shortform and returns this
    model

    Parameters
    ----------
    shortform : str
        Shortform to disambiguate.
    path : Optional[str]
        Path to models directory. Defaults to adeft's pretrained models.
        Users have the option to specify a path to another directory to use
        custom models.

    Returns
    -------
    py:class:`adeft.disambiguate.AdeftDisambiguator`
        A disambiguator that was loaded from a file. Returns None if there
        are no disambiguation models in the supplied folder that can
        disambiguate the given shortform
    """
    available = get_available_models(path=path)
    try:
        model_name = available[shortform]
    except KeyError:
        logger.error('No model available for shortform %s' % shortform)
        return None

    output = load_disambiguator_directly(os.path.join(path, model_name))
    return output


def load_disambiguator_directly(path):
    """Returns disambiguator located at path

    Parameters
    ----------
    path : str
        Path to a disambiguation model. Must be a path to a directory
       <model_name> containing the files
       <model_name>_model.gz, <model_name>_grounding_dict.json,
       <model_name>_names.json

    Returns
    -------
    py:class:`adeft.disambiguate.AdeftDisambiguator`
        A disambiguation model loaded from folder specified by path
    """
    model_name = os.path.basename(os.path.abspath(path))
    model = load_model(os.path.join(path, model_name + '_model.gz'))
    with open(os.path.join(path, model_name + '_grounding_dict.json')) as f:
        grounding_dict = json.load(f)
    with open(os.path.join(path, model_name + '_names.json')) as f:
        names = json.load(f)
    output = AdeftDisambiguator(model, grounding_dict, names)
    return output
