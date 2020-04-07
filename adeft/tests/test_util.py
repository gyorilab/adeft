from adeft.util import get_candidate, get_candidate_fragments


text1 = ('Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) generates executable models of pathway dynamics'
         ' from natural language.')

stripped1 = ('Integrated Network and Dynamical Reasoning Assembler'
             ' generates executable models of pathway dynamics'
             ' from natural language.')
result1 = [['Integrated',  'Network',  'and',  'Dynamical',  'Reasoning',
           'Assembler']]


text2 = ('The Indonesian Debt Restructuring Agency (INDRA) was'
         ' established by the Jakarta Initiative in 1998.')
result2 = [['The',  'Indonesian',  'Debt',  'Restructuring',  'Agency']]

text3 = ('An Indonesian Debt Restructuring Agency (INDRA) was'
         ' established to provide foreign-exchange cover for'
         ' Indonesian corporations with foreign currency denominated'
         ' debt.')
result3 = [['An',  'Indonesian',  'Debt',  'Restructuring',  'Agency']]

text4 = 'Interior Natural Desert Reclamation and Afforestation (INDRA)'
result4 = [['Interior',  'Natural',  'Desert',  'Reclamation',  'and',
            'Afforestation']]

text5 = ('Interior Natural Desert Reclamation and Afforestation (INDRA)'
         ' is not the Integrated Network and Dynamical Reasoning'
         ' Assembler (INDRA). Neither of these is the Indonesian Debt'
         ' Restructuring Agency (INDRA).')
result5 = [['Interior', 'Natural', 'Desert', 'Reclamation', 'and',
            'Afforestation'],
           ['is', 'not', 'the', 'Integrated', 'Network', 'and', 'Dynamical',
            'Reasoning', 'Assembler'],
           ['Neither', 'of', 'these', 'is', 'the', 'Indonesian', 'Debt',
            'Restructuring', 'Agency']]

stopwords = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at', 'from',
                 'into', 'to', 'for', 'on', 'by', 'be', 'being', 'been', 'am',
                 'is', 'are', 'was', 'were', 'in', 'that', 'as'])


def test_get_candidate_fragments():
    """Test extraction of maximal longform candidate from text
    """
    for text, result in zip([text1, text2, text3, text4, text5],
                            [result1, result2, result3, result4, result5]):
        fragments = get_candidate_fragments(text, 'INDRA')
        candidates = [get_candidate(fragment)[0] for fragment in fragments]
        assert candidates == result

    # Case where pattern is at start of the sentence
    fragments1 = get_candidate_fragments(' (INDRA) is an ambiguous acronym',
                                         'INDRA')
    assert not fragments1
    # Case where pattern is not found
    assert not get_candidate_fragments('Integrated Network'
                                       'and dynamical reasoning assembler',
                                       'INDRA')
