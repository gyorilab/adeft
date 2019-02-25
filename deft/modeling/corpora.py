from deft.recognize import DeftRecognizer


class DeftCorpusBuilder(object):
    """Class for generating corpora

    Parameters
    ----------
    shortform : str
        Shortform to disambiguate

    grounding_map : dict of str: str
        Dictionary mapping longform texts to their groundings

    Attributes
    ----------
    shortform : str
        build corpus for this shortform. this is taken from the objects
        longform_recognizer and included as an attribute for convenience

    dr : py:class`deft.recoginze.DeftRecognizer`
        Recognizes longforms for shortform by finding defining patterns (DP)
    """
    def __init__(self, shortform, grounding_map):
        self.shortform = shortform
        self.grounding_map = grounding_map
        self.dr = DeftRecognizer(shortform, grounding_map)

    def build_from_texts(self, texts):
        """Build corpus from a list of texts

        Parameters
        ----------
        texts : list of str
            List of texts to build corpus from

        Returns
        -------
        corpus : list of tuple
            Contains tuples for each text in the input list which contains
            a defining pattern. Multiple tuples correspond to  texts with
            multiple defining patterns for longforms with different groundings.
            The first element of each tuple contains the training text with all
            defining patterns replaced with only the shortform. The second
            element contains the groundings for longforms matched with a
            defining pattern.
        """
        corpus = set()
        for text in texts:
            data_points = self._process_text(text)
            if data_points:
                corpus.update(data_points)
        return corpus

    def _process_text(self, text):
        """Returns training data and label corresponding to text if found

        The training text corresponding to an input text is obtained by
        stripping out all occurences of (<shortform>). It is possible that
        longforms are matched with the standard pattern. In this case, multiple
        datapoints are returned each with different labels but the same
        training text.

        Parameters
        ----------
        text : str
            fulltext to build datapoint from if possible

        Returns
        -------
        datapoints : list of tuple | None
            Returns None if no label can be found by matching the standard
            pattern. Otherwise, returns a list of pairs containing the training
            text and a label for each label appearing in the input text
            matching the standard pattern.
        """
        groundings = self.dr.recognize(text)
        if not groundings:
            return None
        training_text = self.dr.strip_defining_patterns(text)
        datapoints = [(training_text, grounding)
                      for grounding in groundings]
        return datapoints
