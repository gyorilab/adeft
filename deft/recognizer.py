from deft.extraction import Processor


class TrieNode(object):
    __slots__ = ['concept', 'children']

    def __init__(self, concept=None):
        self.concept = concept
        self.children = {}


class Recognizer(object):
    __slots__ = ['shortform', 'concept_map', '_trie', '_processor']

    def __init__(self, shortform, concept_map, exclude=None):
        self.shortform = shortform
        self.concept_map = concept_map
        self._trie = self._init_trie(concept_map)
        self._processor = Processor(shortform, exclude)

    def recognize(self, text):
        candidates = self._processor.extract(text)
        concepts = set([self._search(candidate[::-1])
                        for candidate in candidates])
        if len(concepts) > 1:
            # replace with proper logging
            print('??????')
        return concepts.pop()

    def _init_trie(self, concept_map):
        root = TrieNode()
        for longform, concept in self.concept_map.items():
            current = root
            for index, token in enumerate(longform):
                if token not in current.children:
                    if index == len(longform) - 1:
                        new = TrieNode(concept)
                    else:
                        new = TrieNode()
                    current.children[token] = new
                    current = new
                else:
                    current = current.children[token]
        return root

    def _search(self, tokens):
        current = self._trie
        for token in tokens:
            if token not in current.children:
                break
            if current.children[token].concept is not None:
                return current.children[token].concept
            else:
                current = current.children[token]
        else:
            return None
