from adeft.nlp import stem, word_tokenize
from adeft.recognize import AdeftRecognizer, OneShotRecognizer, SearchTrie


grounding_map = {'endoplasmic reticulum': 'MESH:D004721',
                 'estrogen receptor': 'HGNC:3467',
                 'estrogen receptor alpha': 'HGNC:3467',
                 'endoplasmic reticular': 'MESH:D004721',
                 'emergency room': 'ungrounded',
                 'extra room': 'ungrounded'}


example1 = ('The growth of estrogen receptor (ER)-positive breast cancer'
            ' is inhibited by all-trans-retinoinc acid (RA).',
            'HGNC:3467')

example2 = ('In Xenopus oocytes, overexpressed, unassembled alpha- and'
            ' beta-subunits of Xenopus Na, K-ATPase are retained in the'
            ' endoplasmic reticulum (ER) and are degraded with different'
            ' kinetics.',
            'MESH:D004721')

example3 = ('For women, mandatory HMO programs reduce some types of'
            ' non emergency room (ER) use, and increase reported unmet'
            ' need for medical care.',
            'ungrounded')

example4 = ('We have analyzed interaction of coactivators with the'
            ' wild-type estrogen receptor alpha (ER), HEG0, and a mutant,'
            'L546P-HEG0, which is constitutively active in several'
            ' transiently transfected cells and a HeLa line that stably'
            ' propagates an estrogen-sensitive reporter gene.',
            'HGNC:3467')

example5 = ('A number of studies showed that chemotherapeutic benefits'
            ' may result from targeting the endoplasmic reticular (ER)'
            ' stress signaling pathway',
            'MESH:D004721')


def test_init():
    """Test that the recognizers internal trie is initialized correctly"""
    trie = SearchTrie(grounding_map, token_map=stem)._trie
    for longform, grounding in grounding_map.items():
        edges = tuple(stem(token)
                      for token, _ in word_tokenize(longform))[::-1]
        current = trie
        for index, token in enumerate(edges):
            assert token in current.children
            if index < len(edges) - 1:
                assert current.children[token].data is None
            else:
                assert current.children[token].data == longform
            current = current.children[token]


def test_search():
    """Test that searching for a longform in the trie works correctly"""
    rec = AdeftRecognizer('ER', grounding_map)
    example = ['for', 'women', ',', 'mandatory', 'hmo', 'programs', 'reduce',
               'some', 'types', 'of', 'non', 'emergency', 'room']
    result = rec._search(example)
    assert result == {'longform': 'emergency room'}


def test_recognizer():
    """Test the recognizer end to end"""
    rec = AdeftRecognizer('ER', grounding_map)
    for text, expected in [example1, example2, example3, example4, example5]:
        result = rec.recognize(text)
        assert result.pop()['grounding'] == expected

    # Case where defining pattern appears at the start of the fragment
    assert not rec.recognize('(ER) stress')


def test_strip_defining_patterns():
    rec = AdeftRecognizer('ER', grounding_map)
    test_cases = ['The endoplasmic reticulum (ER) is a transmembrane',
                  'The endoplasmic reticulum (ER)is a transmembrane',
                  'The endoplasmic reticulum (ER)-is a transmembrane']
    results = (['The ER is a transmembrane']*2 +
               ['The ER -is a transmembrane'])

    for case, result in zip(test_cases, results):
        assert rec.strip_defining_patterns(case) == result

    null_case = 'Newly developed extended release (ER) medications'
    null_result = 'Newly developed extended release ER medications'
    assert rec.strip_defining_patterns(null_case) == null_result


def test_one_shot_recognizer():
    example6 = ('A number of studies have assessed the relationship between'
                ' beta-2 adrenergic receptor (ADRB2) gene polymorphisms'
                ' and asthma risk', 'beta-2 adrenergic receptor', 'ADRB2')
    example7 = ('Mutation PRECEPT and Regulon Precept, which use Bayesian'
                ' statistics to characterize predictors of cellular phenotypes'
                ' to guide therapeutic strategies (PRECEPTS)',
                'predictors of cellular phenotypes to guide therapeutic'
                ' strategies', 'PRECEPTS')
    example8 = ('This is a test sentence for the OneShotRecognizer class'
                ' of Acromine based Disambiguation of Entities from Text'
                ' (ADEFT)',
                'Acromine based Disambiguation of Entities from Text',
                'ADEFT')
    example9 = ('Hormones as diverse as adiponectin (ADP) and thromboxane'
                'A2 (TXA2) are mentioned in this sentence.', 'adiponectin',
                'ADP')
    example10 = ('Hormones as diverse as adiponectin (ADP) and thromboxane'
                 ' A2 (TXA2) are mentioned in this sentence.',
                 'thromboxane A2', 'TXA2')
    for text, result, shortform in [example6, example7, example8, example9,
                                    example10]:
        rec = OneShotRecognizer(shortform)
        longform_set = {x['longform_text'] for x in rec.recognize(text)}
        assert longform_set.pop() == result
