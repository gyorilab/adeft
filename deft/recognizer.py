from deft.extraction import Processor


class TrieNode(object):
        __slots__ = ['concept', 'children']

        def __init__(self, concept=None, children={}):
            self.concept = concept
            self.children = children


class Recognizer(object):
    __slots__ = ['shortform', 'concept_map', '_trie']

    def __init__(self, shortform, concept_map):
        self.shortform = shortform
        self.concept_map = concept_map
        self._trie = self._init_trie(concept_map)

    def _init_trie(self, concept_map):
        root = TrieNode()
        for longform, concept in self.concept_map.items():
            print('******', len(root.children))
            current = root
            for index, token in enumerate(longform):
                print(token)
                print(index)
                print(current is root)
                print(current.children)
                print(root.children)
                if token not in current.children:
                    print('*')
                    if index == len(longform) - 1:
                        print('&')
                        new = TrieNode(concept)
                    else:
                        print('&&')
                        new = TrieNode()
                    current.children[token] = new
                    current = new
                else:
                    print('**')
                    current = current.children[token]
        return root
