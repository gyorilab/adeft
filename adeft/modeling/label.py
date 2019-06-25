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

    def build_from_texts(self, texts):
        """Build labeled corpus from a list of texts

        Labels texts based on defining patterns (DPs)

        Parameters
        ----------
        texts : list of str
            List of texts to build corpus from

        Returns
        -------
        corpus : list of tuple
            Contains tuples for each text in the input list which contains
            a defining pattern. Multiple tuples correspond to texts with
            multiple defining patterns for longforms with different groundings.
            The first element of each tuple contains the training text with all
            defining patterns replaced with only the shortform. The second
            element contains the groundings for longforms matched with a
            defining pattern.
        """
        corpus = []
        for text in texts:
            data_points = self._process_text(text)
            if data_points:
                corpus.extend(data_points)
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
            groundings.update(recognizer.recognize(text))
        if not groundings:
            return None
        for recognizer in self.recognizers:
            text = recognizer.strip_defining_patterns(text)
        datapoints = [(text, grounding) for grounding in groundings]
        return datapoints
