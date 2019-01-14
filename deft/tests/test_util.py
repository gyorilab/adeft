from deft.resources import stopwords
from deft.util import contains_shortform, get_max_candidate_longform


sentence1 = ('Integrated Network and Dynamical Reasoning Assembler'
             ' (INDRA) generates executable models of pathway dynamics'
             ' from natural language.')
result1 = 'integrated network and dynamical reasoning assembler'


sentence2 = ('The Indonesian Debt Restructuring Agency (INDRA) was'
             ' established by the Jakarta Initiative in 1998.')
result2 = 'the indonesian debt restructuring agency'

sentence3 = ('An Indonesian Debt Restructuring Agency (INDRA) was'
             ' established to provide foreign-exchange cover for'
             ' Indonesian corporations with foreign currency denominated'
             ' debt.')
result3 = 'an indonesian debt restructuring agency'

sentence4 = 'Interior Natural Desert Reclamation and Afforestation (INDRA)'
result4 = 'interior natural desert reclamation and afforestation'


sentence5 = '(INDRA) (INDRA (INDRA (INDRA)))'
sentence6 = 'The (INDRA) (INDRA (INDRA (INDRA)))'
sentence7 = 'Integrated Network and Dynamical Reasoning assembler'


def test_contains_shortform():
    assert contains_shortform(sentence1, 'INDRA')
    assert not contains_shortform(sentence1, 'DD')


def test_get_candidates():
    """Test extraction of maximal longform candidate from sentence
    """
    # Test with no excluded words
    test_cases = [(sentence1, ['integrated', 'network', 'and', 'dynamical',
                               'reasoning', 'assembler']),
                  (sentence2, ['the', 'indonesian', 'debt', 'restructuring',
                               'agency']),
                  (sentence3, ['an', 'indonesian', 'debt', 'restructuring',
                               'agency']),
                  (sentence4, ['interior', 'natural', 'desert', 'reclamation',
                               'and', 'afforestation'])]

    for sentence, result in test_cases:
        assert get_max_candidate_longform(sentence, 'INDRA') == result

    # Case where pattern is at start of the sentence
    assert get_max_candidate_longform('(INDRA) is an ambiguous acronym',
                                      'INDRA') is None
    # Case where pattern is not found
    assert get_max_candidate_longform('Integrated Network'
                                      'and dynamical reasoning assembler',
                                      'INDRA') is None

    # Test with excluded words
    candidate = get_max_candidate_longform(sentence1, 'INDRA',
                                           exclude=stopwords)
    assert candidate == ['dynamical',  'reasoning', 'assembler']
    assert get_max_candidate_longform('Is (INDRA) ambiguous?',
                                      'INDRA', exclude=stopwords) is None
