from deft.resources import stopwords
from deft.util import contains_shortform, get_candidate_fragments,\
    strip_defining_patterns


text1 = ('Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) generates executable models of pathway dynamics'
         ' from natural language.')

stripped1 = ('Integrated Network and Dynamical Reasoning Assembler'
             ' generates executable models of pathway dynamics'
             ' from natural language.')
result1 = [['integrated',  'network',  'and',  'dynamical',  'reasoning',
           'assembler']]


text2 = ('The Indonesian Debt Restructuring Agency (INDRA) was'
         ' established by the Jakarta Initiative in 1998.')
result2 = [['the',  'indonesian',  'debt',  'restructuring',  'agency']]

text3 = ('An Indonesian Debt Restructuring Agency (INDRA) was'
         ' established to provide foreign-exchange cover for'
         ' Indonesian corporations with foreign currency denominated'
         ' debt.')
result3 = [['an',  'indonesian',  'debt',  'restructuring',  'agency']]

text4 = 'Interior Natural Desert Reclamation and Afforestation (INDRA)'
result4 = [['interior',  'natural',  'desert',  'reclamation',  'and',
            'afforestation']]

text5 = ('Interior Natural Desert Reclamation and Afforestation (INDRA)'
         ' is not the Integrated Network and Dynamical Reasoning'
         ' Assembler (INDRA). Neither of these is the Indonesian Debt'
         ' Restructuring Agency (INDRA).')
result5 = [['interior', 'natural', 'desert', 'reclamation', 'and',
            'afforestation'],
           ['is', 'not', 'the', 'integrated', 'network', 'and', 'dynamical',
            'reasoning', 'assembler'],
           ['neither', 'of', 'these', 'is', 'the', 'indonesian', 'debt',
            'restructuring', 'agency']]


def test_contains_shortform():
    assert contains_shortform(text1, 'INDRA')
    assert not contains_shortform(text1, 'DD')


def test_strip_defining_patterns():
    stripped_text = strip_defining_patterns(text1, 'INDRA')
    assert stripped_text == stripped1


def test_get_candidate_fragments():
    """Test extraction of maximal longform candidate from text
    """
    # Test with no excluded words
    for text, result in zip([text1, text2, text3, text4, text5],
                            [result1, result2, result3, result4, result5]):
        fragments = get_candidate_fragments(text, 'INDRA')
        assert fragments == result

    # Case where pattern is at start of the sentence
    fragments1 = get_candidate_fragments('(INDRA) is an ambiguous acronym',
                                         'INDRA')
    assert not fragments1
    # Case where pattern is not found
    assert not get_candidate_fragments('Integrated Network'
                                       'and dynamical reasoning assembler',
                                       'INDRA')

    # Test with excluded words
    fragments2 = get_candidate_fragments(text1, 'INDRA',
                                         exclude=stopwords)
    assert fragments2[0] == ['dynamical',  'reasoning', 'assembler']

    fragment3 = get_candidate_fragments('Is (INDRA) ambiguous?',
                                        'INDRA', exclude=stopwords)
    assert not fragment3
