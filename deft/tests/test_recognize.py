from deft.recognize import Recognizer

grounding_map = {('reticulum', 'endoplasm'): 'MESH:D004721',
                 ('receptor', 'estrogen'): 'HGNC:3467',
                 ('alpha', 'receptor', 'estrogen'): 'HGNC:3467',
                 ('reticular', 'endoplasm'): 'MESH:D004721',
                 ('room', 'emerg'): 'MESH:D004631'}


example1 = ('The growth of estrogen receptor (ER)-positive breast cancer'
            ' is inhibited by all-trans-retinoinc acid (RA).',
            'HGNC:3467')

example2 = ('Assembly of alpha- and beta-subunits in the endoplasmic'
            ' reticulum is a prerequisite for the structural and'
            ' functional maturation of oligomeric P-type ATPases. In'
            ' Xenopus oocytes, overexpressed, unassembled alpha- and'
            ' beta-subunits of Xenopus Na, K-ATPase are retained in the'
            ' endoplasmic reticulum (ER) and are degraded with different'
            ' kinetics.',
            'MESH:D004721')

example3 = ('For women, mandatory HMO programs reduce some types of'
            ' non emergency room (ER) use, and increase reported unmet'
            ' need for medical care.',
            'MESH:D004631')

example4 = ('We have analyzed interaction of coactivators with the'
            ' wild-type estrogen receptor alpha (ER), HEG0, and a mutant,'
            'L546P-HEG0, which is constitutively active in several'
            ' transiently transfected cells and a HeLa line that stably'
            ' propagates an estrogen-sensitive reporter gene',
            'HGNC:3467')

example5 = ('A number of studies showed that chemotherapeutic benefits'
            ' may result from targeting the endoplasmic reticular (ER)'
            ' stress signaling pathway',
            'MESH:D004721')


def test_init():
    """Test that the recognizers internal trie is initialized correctly"""
    recognizer = Recognizer('ER', grounding_map)
    trie = recognizer._trie
    for longform, grounding in grounding_map.items():
        current = trie
        for index, token in enumerate(longform):
            assert token in current.children
            if index < len(longform) - 1:
                assert current.children[token].grounding is None
            else:
                assert current.children[token].grounding == grounding
            current = current.children[token]


def test_search():
    """Test that searching for a longform in the trie works correctly"""
    recognizer = Recognizer('ER', grounding_map)
    example = (('room', 'emerg', 'non', 'of', 'type', 'some',
                'reduc', 'program', 'hmo', 'mandatori', ',', 'women', 'for'),
               'MESH:D004631')
    assert recognizer._search(example[0]) == example[1]


def test_recognizer():
    """Test the recognizer end to end"""
    recognizer = Recognizer('ER', grounding_map)
    for text, grounding in [example1, example2, example3, example4, example5]:
        assert recognizer.recognize(text) == grounding
