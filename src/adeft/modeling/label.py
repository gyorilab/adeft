from adeft.recognize import AdeftRecognizer


class AdeftLabeler(object):
    """Class for labeling corpora

    Parameters
    ----------
    grounding_dict : dict of dict of str
        Dictionary mapping shortforms to grounding_map dictionaries mapping
        longforms to groundings

    Attributes
    ----------
    recognizers : list of py:class`adeft.recognize.AdeftRecognizer`
        List of recognizers for each shortform to be considered. Each
        recognizer identifies longforms for a shortform by finding defining
        matches to a defining pattern (DP)
    """
    def __init__(self, grounding_dict):
        self.grounding_dict = grounding_dict
        self.recognizers = [AdeftRecognizer(shortform, grounding_map)
                            for shortform, grounding_map
                            in grounding_dict.items()]

    def build_from_texts(self, text_tuples):
        """Build labeled corpus from a list of texts

        Labels texts based on defining patterns (DPs)

        Parameters
        ----------
        text_tuples : list of tuple
            List of two element tuples whose first elements are texts from
            which we seek to build a corpus and whose second elements are
            identifiers associated with the texts. Each text should have a
            unique identifier associated to it.

        Returns
        -------
        corpus : list
            Contains a tuple for each text in the input list which contains
            a defining pattern. Multiple tuples correspond to texts with
            multiple defining patterns for longforms with different groundings.
            The first element of each tuple contains a training text with all
            defining patterns replaced with only the shortform. The second
            element contains a grounding label for the desired shortform within
            the training text that was identified through a defining
            pattern. The third element contains the identifier for the given
            training text.
        """
        corpus = []
        for text, identifier in text_tuples:
            data_points = self._process_text(text)
            if data_points:
                corpus.extend((*data_point, identifier)
                              for data_point in data_points)
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
            Fulltext to build datapoint from, if possible.

        Returns
        -------
        datapoints : list of tuple or None
            Returns None if no label can be found by matching the standard
            pattern. Otherwise, returns a list of pairs containing the training
            text and a label for each label appearing in the input text
            matching the standard pattern.
        """
        groundings = set()
        for recognizer in self.recognizers:
            groundings.update({x['grounding']
                               for x in recognizer.recognize(text)})
        if not groundings:
            return None
        for recognizer in self.recognizers:
            text = recognizer.strip_defining_patterns(text)
        datapoints = [(text, grounding) for grounding in groundings]
        return datapoints
