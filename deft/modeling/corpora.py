from nltk.tokenize import sent_tokenize

from deft.util import contains_shortform


class CorpusBuilder(object):
    """Class for generating corpora

    Parameters
    ----------
    lfr : :py:class:`deft.recognize.LongformRecognizer`
        A recognizer that can find longforms by matching the standard pattern

    Attributes
    ----------
    shortform : str
        build corpus for this shortform. this is taken from the objects
        longform_recognizer and included as an attribute for convenience

    corpus : list of tuple
       List of pairs of the form (<text>, <label>) that can be used as training
       data for classification algorithms
    """
    __slots__ = ['lfr', 'shortform', 'corpus']

    def __init__(self, longform_recognizer):
        self.lfr = longform_recognizer
        self.shortform = longform_recognizer.shortform
        self.corpus = set([])

    def get_from_texts(self, texts):
        for text in texts:
            data_points = self._process_text(text)
            if data_points:
                self.corpus.update(data_points)

    def _process_text(self, text):
        """Returns training data and label corresponding to text if found

        The training text corresponding to an input text is obtained by
        stripping out all sentences that containing a longform defined with
        the standard pattern that can be recognized. It is possible that
        longforms are matched with the standard pattern. In this case, multiple
        datapoints are returned each with different labels but the same
        training text.

        Parameters
        ----------
        text : str
            fulltext to build datapoint from if possible

        Returns
        -------
        list of tuple | None
            Returns None if no label can be found by matching the standard
            pattern. Otherwise, returns a list of pairs containing the training
            text and a label for each label appearing in the input text
            matching the standard pattern.
        """
        if not contains_shortform(text, self.shortform):
            return None
        else:
            sentences = sent_tokenize(text)
            training_sentences = []
            labels = []
            for sentence in sentences:
                if contains_shortform(sentence, self.shortform):
                    longform = self.lfr.recognize(sentence)
                    if longform:
                        sentence = sentence.replace(' (%s)' % self.shortform,
                                                    '')
                        labels.append(longform)
                training_sentences.append(sentence)
        if not labels:
            return None
        else:
            training_text = ' '.join(training_sentences)
            return [(training_text, label) for label in labels]
